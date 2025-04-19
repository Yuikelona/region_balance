import torch
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image
import math
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# --- AttentionStore 和 register_attention_control ---
class AttentionStore:
    # Important: Modify __call__ or add logic to handle reset correctly per model call if needed
    def __init__(self):
        self.step_store = self.get_empty_store()
        # self.attention_store = {} # Not needed if only using step_store per call
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self._is_active = True # Flag to enable/disable storing

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if not self._is_active: return attn

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Adjust size check based on expected SDXL attention sizes if needed
        if is_cross and attn.shape[1] <= 64 ** 2:
            # Store a clone BUT WITHOUT DETACH during grad calculation phase
            self.step_store[key].append(attn.clone()) # REMOVED .detach()
        return attn

    def get_step_store(self):
        # Return the captured store for the current step/call
        store = self.step_store
        self.step_store = self.get_empty_store() # Clear after getting
        self.cur_att_layer = 0 # Reset layer count
        return store

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self._is_active = True # Ensure active on reset

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.num_att_layers == -1: # Should be set by register_attention_control
             return attn

        # Simplified call logic: just forward and let get_step_store handle reset
        processed_attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        # Reset happens when get_step_store is called
        # if self.cur_att_layer == self.num_att_layers:
        #     self.cur_att_layer = 0
        #     self.cur_step += 1
        return processed_attn

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs): # Accept kwargs
            # ... (rest of the forward logic from original register_attention_control) ...
            # --- Start copy from original ---
            is_cross = encoder_hidden_states is not None
            residual = hidden_states

            # Compatibility check for different attention block versions
            if hasattr(self, 'spatial_norm') and self.spatial_norm is not None:
                 hidden_states = self.spatial_norm(hidden_states, temb) # temb might be needed

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            # attention_mask handling might differ slightly in newer diffusers
            # attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size) if hasattr(self, 'prepare_attention_mask') else attention_mask

            if hasattr(self, 'group_norm') and self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            # Handling for norm_encoder_hidden_states based on attention processor type might be needed
            elif hasattr(self, 'norm_cross') and self.norm_cross:
                 encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            # Using scaled_dot_product_attention is common now
            if hasattr(F, 'scaled_dot_product_attention') and not hasattr(self, 'scale'): # Check if it's likely using SDPA
                 # Newer diffusers might use SDPA directly or via AttentionProcessor
                 # This hook might need adjustment depending on the exact AttentionProcessor used
                 # For simplicity, let's assume the older baddbmm path for hooking,
                 # but be aware this might need changes for newer diffusers versions.

                 # Fallback to baddbmm style calculation for hooking if scale exists
                 query = self.head_to_batch_dim(query)
                 key = self.head_to_batch_dim(key)
                 value = self.head_to_batch_dim(value)

                 attention_probs = torch.baddbmm(
                     torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                     query,
                     key.transpose(-1, -2),
                     beta=0,
                     alpha=self.scale,
                 )
                 attention_probs = attention_probs.softmax(dim=-1)
                 attention_probs = attention_probs.to(value.dtype)

                 # Hooking point
                 attention_probs = controller(attention_probs, is_cross, place_in_unet)

                 hidden_states = torch.bmm(attention_probs, value)
                 hidden_states = self.batch_to_head_dim(hidden_states)

            else: # Assume older or non-SDPA path compatible with original hook
                 query = self.head_to_batch_dim(query)
                 key = self.head_to_batch_dim(key)
                 value = self.head_to_batch_dim(value)

                 attention_scores = torch.baddbmm(
                     torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                     query,
                     key.transpose(-1, -2),
                     beta=0,
                     alpha=self.scale,
                 )
                 # Apply attention mask if needed (might be handled differently)
                 # if attention_mask is not None:
                 #    attention_scores = attention_scores + attention_mask

                 attention_probs = attention_scores.softmax(dim=-1)
                 del attention_scores
                 attention_probs = attention_probs.to(value.dtype)

                 # Hooking point
                 attention_probs = controller(attention_probs, is_cross, place_in_unet)

                 hidden_states = torch.bmm(attention_probs, value)
                 hidden_states = self.batch_to_head_dim(hidden_states)


            # linear proj
            hidden_states = to_out(hidden_states)

            # residual connection
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if hasattr(self, 'residual_connection') and self.residual_connection:
                 hidden_states = hidden_states + residual
            if hasattr(self, 'rescale_output_factor'):
                 hidden_states = hidden_states / self.rescale_output_factor
            # --- End copy from original ---

            return hidden_states
        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0
        def reset(self): pass
        def activate(self): pass
        def deactivate(self): pass

    if controller is None:
        controller = DummyController()

    # Use a more robust check for attention blocks
    attention_classes = []
    try:
        from diffusers.models.attention_processor import Attention
        attention_classes.append(Attention)
    except ImportError:
        print("Warning: diffusers.models.attention_processor.Attention not found.")
    try:
        # Older diffusers might have CrossAttention here, adjust if needed
        from diffusers.models.attention import BasicTransformerBlock
        # Check within BasicTransformerBlock for the actual attention module if needed
        # This part highly depends on the diffusers version
        print("Warning: register_attention_control might need adjustments for BasicTransformerBlock structure.")
    except ImportError:
        pass
    if not attention_classes:
         print("ERROR: Could not find compatible Attention class in diffusers.")
         # Fallback attempt (less reliable)
         attention_class_name = 'Attention'
    else:
         attention_class_name = attention_classes[0].__name__


    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == attention_class_name: # Use found class name
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            # Iterate through named children to ensure order consistency (though less critical here)
            for name, net__ in net_.named_children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net_name, net_module in sub_nets:
        if "down_blocks" in net_name:
            cross_att_count += register_recr(net_module, 0, "down")
        elif "up_blocks" in net_name:
            cross_att_count += register_recr(net_module, 0, "up")
        elif "mid_block" in net_name:
            cross_att_count += register_recr(net_module, 0, "mid")

    if hasattr(controller, 'num_att_layers'):
        print(f"Registered {cross_att_count} attention layers.")
        controller.num_att_layers = cross_att_count
    else:
         print("Warning: Controller does not have num_att_layers attribute.")


# --- Balancer Class ---
class AnimeStyleBalancer:
    def __init__(self,
                 animagine_unet: UNet2DConditionModel,
                 pony_unet: UNet2DConditionModel,
                 controller: AttentionStore,
                 vae: AutoencoderKL,
                 tokenizer: CLIPTokenizer, # Tokenizer 1 (CLIP ViT-L)
                 text_encoder: CLIPTextModel, # Text Encoder 1 (CLIP ViT-L)
                 tokenizer_2: CLIPTokenizer, # Tokenizer 2 (OpenCLIP ViT-bigG)
                 text_encoder_2: CLIPTextModelWithProjection, # Text Encoder 2 (OpenCLIP ViT-bigG)
                 scheduler, # Pass the scheduler instance
                 scale_factor=0.1, # Adjusted default
                 scale_range=(1.0, 0.5), # Adjusted default
                 loss_weight_style=0.5, # Weight for style loss
                 loss_weight_semantic=0.5, # Weight for semantic loss
                 attn_res=32, # Default attention resolution for loss
                 ):
        self.unet1 = animagine_unet
        self.unet2 = pony_unet
        self.controller = controller
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_2 = text_encoder_2
        self.scheduler = scheduler
        self.scale_factor = scale_factor
        self.scale_range = scale_range
        self.loss_weight_style = loss_weight_style
        self.loss_weight_semantic = loss_weight_semantic
        self.attn_res = attn_res # Store attention resolution

        self.device = self.unet1.device
        self.vae_scale_factor = vae.config.scaling_factor

        # Register controllers (assuming they haven't been registered outside)
        # It's often better to register outside and pass the controller instance
        # register_attention_control(self.unet1, self.controller)
        # register_attention_control(self.unet2, self.controller)


    def _encode_prompt_sdxl(self, prompt, prompt_2, negative_prompt, negative_prompt_2, device, num_images_per_prompt=1, do_classifier_free_guidance=True):
        # Helper to encode prompts for SDXL using both text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                print(f"Warning: The following part of your input was truncated: {removed_text}")

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0] # Use last_hidden_state for text_embeds
            prompt_embeds = prompt_embeds.hidden_states[-2] # Usually second to last hidden state

            # Negative prompts
            if do_classifier_free_guidance:
                uncond_tokens = [negative_prompt] * len(prompt) if isinstance(negative_prompt, str) else negative_prompt
                max_length = text_input_ids.shape[-1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds_output = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
                negative_prompt_embeds = negative_prompt_embeds_output.hidden_states[-2]
                negative_pooled_prompt_embeds = negative_prompt_embeds_output[0]

                # Repeat embeds for multiple images per prompt
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                # For classifier-free guidance
                seq_len = negative_prompt_embeds.shape[1]
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

            prompt_embeds_list.append(prompt_embeds)

        # Concatenate embeds from both encoders
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, device):
         # Helper function for SDXL time IDs
         add_time_ids = list(original_size + crops_coords_top_left + target_size)
         add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
         return add_time_ids

    def get_attention_maps_for_loss(self, latents, t, prompt_embeds, added_cond_kwargs):
        """Gets attention maps from conditional forward pass only, for loss calculation."""
        attn_maps = {}
        # Ensure controller is active and reset
        self.controller.activate()
        self.controller.reset()

        # Model 1 (Animagine) - Conditional only
        _ = self.unet1(latents, t, encoder_hidden_states=prompt_embeds[1:], # Index 1 for conditional
                       added_cond_kwargs={k: v[1:] for k, v in added_cond_kwargs.items()}).sample
        attn_maps['model1'] = self.controller.get_step_store() # Get and reset store

        # Model 2 (Pony) - Conditional only
        self.controller.reset() # Reset again just in case
        _ = self.unet2(latents, t, encoder_hidden_states=prompt_embeds[1:], # Index 1 for conditional
                       added_cond_kwargs={k: v[1:] for k, v in added_cond_kwargs.items()}).sample
        attn_maps['model2'] = self.controller.get_step_store() # Get and reset store

        # Deactivate controller after use for loss to avoid capturing during noise prediction
        self.controller.deactivate()
        return attn_maps['model1'], attn_maps['model2']


    @torch.no_grad()
    def get_noise_pred_no_grad(self, latents, t, prompt_embeds, added_cond_kwargs, guidance_scale):
        """Gets noise predictions from both models without gradients."""
        # Ensure controller is inactive during noise prediction
        self.controller.deactivate()

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        t_input = t # timestep is usually scalar or matched shape

        # Model 1 (Animagine)
        noise_pred1 = self.unet1(latent_model_input, t_input,
                                 encoder_hidden_states=prompt_embeds,
                                 added_cond_kwargs=added_cond_kwargs).sample
        noise_pred_uncond1, noise_pred_text1 = noise_pred1.chunk(2)
        noise_pred_cfg1 = noise_pred_uncond1 + guidance_scale * (noise_pred_text1 - noise_pred_uncond1)

        # Model 2 (Pony)
        noise_pred2 = self.unet2(latent_model_input, t_input,
                                 encoder_hidden_states=prompt_embeds,
                                 added_cond_kwargs=added_cond_kwargs).sample
        noise_pred_uncond2, noise_pred_text2 = noise_pred2.chunk(2)
        noise_pred_cfg2 = noise_pred_uncond2 + guidance_scale * (noise_pred_text2 - noise_pred_uncond2)

        return noise_pred_cfg1, noise_pred_cfg2

    def compose_noise_pred(self, noise_pred1, noise_pred2, influence1, influence2):
        """Combine noise predictions using influence maps."""
        # Ensure influence maps match spatial dimensions
        h, w = noise_pred1.shape[-2:]
        influence1 = torch.nn.functional.interpolate(influence1, size=(h, w), mode='bicubic', align_corners=False) # Use bicubic for potentially smoother maps
        influence2 = torch.nn.functional.interpolate(influence2, size=(h, w), mode='bicubic', align_corners=False)

        concat_map = torch.cat([influence1, influence2], dim=1) # [B, 2, H, W]
        softmax_map = torch.nn.functional.softmax(input=concat_map, dim=1)

        confidence_map1 = softmax_map[:, 0:1, :, :] # [B, 1, H, W]
        confidence_map2 = softmax_map[:, 1:2, :, :] # [B, 1, H, W]

        # No need to repeat channels, broadcasting handles it
        combined_noise_pred = confidence_map1 * noise_pred1 + confidence_map2 * noise_pred2
        return combined_noise_pred

    def aggregate_cross_attention_for_loss(self, attention_store, res):
        """ Aggregates cross-attention from the store, focusing on the conditional part. """
        out = []
        num_pixels = res ** 2
        if not attention_store: return None # Handle empty store

        # We only ran the conditional part, so batch size is 1 in the stored attention
        batch_size_stored = 1 # Because get_attention_maps_for_loss runs cond only
        select_idx = 0 # Index within the stored batch (which is only the conditional part)

        for location in ["down", "mid", "up"]: # Check all locations
            key = f"{location}_cross"
            if key in attention_store:
                for item in attention_store[key]:
                    # item shape: [heads * batch_size_stored, seq_len, num_tokens]
                    # seq_len here is num_pixels (e.g., 1024 for 32x32)
                    if item.shape[1] == num_pixels:
                        # Reshape: [batch_size_stored, num_heads, res, res, num_tokens]
                        # Note: Shape might vary slightly based on diffusers version / attention processor
                        # Assuming head dimension is first after splitting batch
                        num_heads = item.shape[0] // batch_size_stored
                        try:
                            cross_maps = item.reshape(batch_size_stored, num_heads, res, res, item.shape[-1])[select_idx]
                            # Sum across heads or average? Summing might emphasize stronger heads. Avg is safer.
                            cross_maps_avg_heads = cross_maps.mean(dim=0) # Avg over heads [res, res, num_tokens]
                            out.append(cross_maps_avg_heads)
                        except Exception as e:
                             print(f"Warning: Reshaping error in aggregate_cross_attention: {e}, item shape: {item.shape}, target res: {res}")
                             continue # Skip problematic items

        if not out:
            # print(f"Warning: No suitable cross-attention maps found for res {res}.")
            return None

        # Stack and average across layers/blocks
        out_tensor = torch.stack(out, dim=0) # [num_layers, res, res, num_tokens]
        avg_attn = out_tensor.mean(dim=0) # Average over layers [res, res, num_tokens]
        return avg_attn # Return CPU tensor? Loss calculation needs device tensor. Keep on device.

    def compute_style_coherence_loss(self, attn_map1, attn_map2):
        """Computes KL divergence between spatial attention distributions."""
        if attn_map1 is None or attn_map2 is None:
            return torch.tensor(0.0, device=self.device)

        # Sum across tokens to get spatial distribution
        dist1 = attn_map1.sum(dim=-1) # [res, res]
        dist2 = attn_map2.sum(dim=-1) # [res, res]

        # Normalize (ensure non-negative before sum)
        dist1 = torch.clamp(dist1, min=0.0)
        dist2 = torch.clamp(dist2, min=0.0)
        dist1 = dist1 / (dist1.sum() + 1e-10)
        dist2 = dist2 / (dist2.sum() + 1e-10)

        # Symmetric KL divergence
        kl_div1 = torch.sum(dist1 * (torch.log(dist1 + 1e-10) - torch.log(dist2 + 1e-10)))
        kl_div2 = torch.sum(dist2 * (torch.log(dist2 + 1e-10) - torch.log(dist1 + 1e-10)))

        return (kl_div1 + kl_div2) / 2.0

    def compute_semantic_consistency_loss(self, attn_map1, attn_map2, token_indices):
        """Computes 1 - correlation for specified token attentions."""
        if attn_map1 is None or attn_map2 is None or not token_indices:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        num_valid_tokens = 0

        max_tokens = attn_map1.shape[-1]

        for token_idx in token_indices:
             if token_idx >= max_tokens: continue # Skip if index out of bounds

             attn1 = attn_map1[:, :, token_idx].flatten() # [res*res]
             attn2 = attn_map2[:, :, token_idx].flatten() # [res*res]

             # Normalize variance for correlation
             if attn1.std() > 1e-6 and attn2.std() > 1e-6: # Avoid NaN correlation
                 # Using cosine similarity might be more stable than correlation
                 # Cosine Similarity = (A . B) / (||A|| * ||B||)
                 cos_sim = F.cosine_similarity(attn1.unsqueeze(0), attn2.unsqueeze(0))
                 loss += 1.0 - cos_sim # Maximize similarity -> minimize 1 - sim
                 num_valid_tokens += 1

        if num_valid_tokens == 0: return torch.tensor(0.0, device=self.device)
        return loss / num_valid_tokens


    def compute_total_loss(self, attn_store1, attn_store2, token_indices):
        """Computes the combined loss."""
        # Aggregate attention maps specifically for loss calculation resolution
        attn_map1 = self.aggregate_cross_attention_for_loss(attn_store1, res=self.attn_res)
        attn_map2 = self.aggregate_cross_attention_for_loss(attn_store2, res=self.attn_res)

        style_loss = self.compute_style_coherence_loss(attn_map1, attn_map2)
        semantic_loss = self.compute_semantic_consistency_loss(attn_map1, attn_map2, token_indices)

        total_loss = (self.loss_weight_style * style_loss +
                      self.loss_weight_semantic * semantic_loss)
        return total_loss

    def sample(self,
               prompt: str,
               prompt_2: str = None,
               height: int = 1024,
               width: int = 1024,
               num_inference_steps: int = 30,
               guidance_scale: float = 7.0,
               negative_prompt: str = "",
               negative_prompt_2: str = None,
               generator: torch.Generator = None,
               num_images_per_prompt: int = 1,
               ):
        """Main sampling loop with dynamic balancing."""
        batch_size = 1 # Currently supports batch size 1 for balancer logic

        # 1. Encode Prompts
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt_sdxl(
            prompt=[prompt] * batch_size,
            prompt_2=[prompt_2 or prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            negative_prompt_2=[negative_prompt_2 or negative_prompt] * batch_size,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0
        )

        # Get token indices for semantic loss (use first tokenizer)
        # Exclude special tokens like <|startoftext|>, <|endoftext|>, padding
        text_input_ids = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True).input_ids
        special_token_ids = {self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        token_indices = [i for i, token_id in enumerate(text_input_ids) if token_id not in special_token_ids]


        # 2. Prepare Timesteps and Latents
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet1.config.in_channels
        latents = torch.randn(
            (batch_size * num_images_per_prompt, num_channels_latents, height // 8, width // 8),
            generator=generator, device=self.device, dtype=prompt_embeds.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma # Scale initial noise

        # Prepare added conditions for SDXL
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype, device=self.device)
        if guidance_scale > 1.0:
            add_time_ids = torch.cat([add_time_ids] * 2) # Repeat for CFG

        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

        # 3. Initialize Influence Maps
        latent_h, latent_w = latents.shape[-2:]
        influence1 = torch.randn((batch_size * num_images_per_prompt, 1, latent_h, latent_w), device=self.device, generator=generator, dtype=latents.dtype) * 0.01 # Start near zero
        influence2 = torch.randn((batch_size * num_images_per_prompt, 1, latent_h, latent_w), device=self.device, generator=generator, dtype=latents.dtype) * 0.01 # Start near zero

        # Scale factor scheduling
        scale_factors = np.linspace(self.scale_range[0], self.scale_range[1], len(timesteps))

        # Define wrapper functions inside sample or as helper methods if preferred
        # These wrappers ensure the call signature matches what checkpoint expects
        # and handles the .sample attribute access.
        def unet1_checkpoint_wrapper(latent_input, timestep, encoder_hidden_states, added_cond_kwargs):
            # Accesses self.unet1 from the outer scope
            return self.unet1(latent_input, timestep, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample

        def unet2_checkpoint_wrapper(latent_input, timestep, encoder_hidden_states, added_cond_kwargs):
            # Accesses self.unet2 from the outer scope
            return self.unet2(latent_input, timestep, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample

         # 4. Denoising Loop
        for i, t in enumerate(tqdm(timesteps)):
            # --- Balancer Update Step ---
            influence1_detached = influence1.detach().clone().requires_grad_(True)
            influence2_detached = influence2.detach().clone().requires_grad_(True)
            latents_for_grad = latents.detach()

            with torch.enable_grad():
                # 1. Get noise predictions (values needed, no grad path here yet is ok)
                noise_pred1_no_grad, noise_pred2_no_grad = self.get_noise_pred_no_grad(
                     latents_for_grad, t, prompt_embeds, added_cond_kwargs, guidance_scale
                )

                # 2. Compose noise using influence maps (THIS introduces grad path)
                noise_pred_grad = self.compose_noise_pred(
                    noise_pred1_no_grad, noise_pred2_no_grad, influence1_detached, influence2_detached
                )

                # 3. Simulate the NEXT latent state based on composed noise
                if not hasattr(self.scheduler, 'alphas_cumprod'):
                     raise AttributeError("Scheduler must have 'alphas_cumprod' attribute for manual step simulation.")

                # Ensure t is an integer index if scheduler expects it
                current_t_index = t.item() if isinstance(t, torch.Tensor) else t
                # Handle potential indexing issues if t comes from timesteps directly
                if current_t_index >= len(self.scheduler.alphas_cumprod):
                    # Fallback or error for out-of-bounds timestep index
                    print(f"Warning: Timestep index {current_t_index} out of range for alphas_cumprod. Using last value.")
                    current_t_index = len(self.scheduler.alphas_cumprod) - 1 # Use last valid index as fallback

                alpha_prod_t = self.scheduler.alphas_cumprod[current_t_index]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latents_for_grad - beta_prod_t ** (0.5) * noise_pred_grad) / alpha_prod_t ** (0.5)

                if i + 1 < len(timesteps):
                    t_prev_index = timesteps[i+1].item() if isinstance(timesteps[i+1], torch.Tensor) else timesteps[i+1]
                    if t_prev_index >= len(self.scheduler.alphas_cumprod):
                         t_prev_index = len(self.scheduler.alphas_cumprod) - 1
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev_index]
                else:
                    alpha_prod_t_prev = self.scheduler.config.final_alpha_cumprod if hasattr(self.scheduler.config,'final_alpha_cumprod') else 0.0

                pred_epsilon = noise_pred_grad
                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
                x_prev_simulated = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

                scaled_x_prev_simulated = self.scheduler.scale_model_input(x_prev_simulated, t)

                # 4. Get attention maps using CHECKPOINT
                #    Pass arguments clearly to the wrapper via checkpoint
                self.controller.activate()
                self.controller.reset()
                # --- MODIFICATION START ---
                _ = checkpoint(unet1_checkpoint_wrapper,            # Function to checkpoint
                               scaled_x_prev_simulated,           # Arg 1 for wrapper
                               t,                                 # Arg 2 for wrapper
                               prompt_embeds[1:],                 # Arg 3 for wrapper (Cond only embeds)
                               {k: v[1:] for k, v in added_cond_kwargs.items()}, # Arg 4 for wrapper (Cond only kwargs)
                               use_reentrant=False)               # Recommended setting
                # --- MODIFICATION END ---
                attn_store1 = self.controller.get_step_store()

                self.controller.reset()
                # --- MODIFICATION START ---
                _ = checkpoint(unet2_checkpoint_wrapper,            # Function to checkpoint
                               scaled_x_prev_simulated,           # Arg 1 for wrapper
                               t,                                 # Arg 2 for wrapper
                               prompt_embeds[1:],                 # Arg 3 for wrapper (Cond only embeds)
                               {k: v[1:] for k, v in added_cond_kwargs.items()}, # Arg 4 for wrapper (Cond only kwargs)
                               use_reentrant=False)               # Recommended setting
                # --- MODIFICATION END ---
                attn_store2 = self.controller.get_step_store()
                self.controller.deactivate()

                # 5. Compute loss based on attention from simulated state
                loss = self.compute_total_loss(attn_store1, attn_store2, token_indices)

                # 6. Compute gradients
                if loss.requires_grad:
                    grad1, grad2 = torch.autograd.grad(loss, [influence1_detached, influence2_detached], retain_graph=False, allow_unused=True)
                else:
                    grad1, grad2 = None, None

            # --- End of Balancer Update Step ---

            # Update influence maps (outside grad context)
            step_size = self.scale_factor * scale_factors[i]
            if grad1 is not None:
                influence1 = influence1 - step_size * grad1.detach()
            if grad2 is not None:
                influence2 = influence2 - step_size * grad2.detach()

            # --- Denoising Step ---
            # Combine noise using *updated* influence maps
            noise_pred = self.compose_noise_pred(
                noise_pred1_no_grad, noise_pred2_no_grad, influence1, influence2
            )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
            # --- End of Denoising Step ---

        # 5. Decode Latents
        # latents = latents / self.vae_scale_factor # Handled by decode? Check pipeline code. Diffusers decode usually handles this.
        latents = 1 / self.vae_scale_factor * latents # Manual scaling often needed before decode
        image = self.vae.decode(latents.to(self.vae.dtype)).sample # Ensure dtype match
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy() # Added .detach()
        image = Image.fromarray((image * 255).round().astype(np.uint8)[0]) # Select first image if batch > 1

        return image


# --- Loading and Main ---
def load_sdxl_models(animagine_repo="cagliostrolab/animagine-xl-3.1",
                     pony_repo="cagliostrolab/animagine-xl-4.0",
                     dtype=torch.float16,
                     device="cuda"):
    """Loads all necessary components for SDXL."""

    # Load UNets
    animagine_unet = UNet2DConditionModel.from_pretrained(animagine_repo, subfolder="unet", torch_dtype=dtype).to(device)
    pony_unet = UNet2DConditionModel.from_pretrained(pony_repo, subfolder="unet", torch_dtype=dtype).to(device)

    # Load VAE (use one, e.g., from Animagine)
    vae = AutoencoderKL.from_pretrained(animagine_repo, subfolder="vae", torch_dtype=dtype).to(device)

    # Load Tokenizers and Text Encoders (SDXL requires two sets)
    tokenizer = CLIPTokenizer.from_pretrained(animagine_repo, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(animagine_repo, subfolder="text_encoder", torch_dtype=dtype).to(device)
    tokenizer_2 = CLIPTokenizer.from_pretrained(animagine_repo, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(animagine_repo, subfolder="text_encoder_2", torch_dtype=dtype).to(device)

    # Load Scheduler (use one, e.g., from Animagine)
    scheduler = DDIMScheduler.from_pretrained(animagine_repo, subfolder="scheduler") # Or EulerDiscreteScheduler, etc.

    return (animagine_unet, pony_unet, vae,
            tokenizer, text_encoder, tokenizer_2, text_encoder_2,
            scheduler)

"""Example usage"""
device = "cuda"
dtype = torch.float16
# Load models
(animagine_unet, pony_unet, vae,
  tokenizer, text_encoder, tokenizer_2, text_encoder_2,
  scheduler) = load_sdxl_models(dtype=dtype, device=device)

# Initialize attention controller
controller = AttentionStore()

# Register attention controller with models (IMPORTANT!)
print("Registering attention controller for Animagine UNet...")
register_attention_control(animagine_unet, controller)
print("Registering attention controller for Pony UNet...")
register_attention_control(pony_unet, controller)
# Note: register_attention_control modifies the UNets in-place

# Initialize balancer
balancer = AnimeStyleBalancer(
    animagine_unet=animagine_unet,
    pony_unet=pony_unet,
    controller=controller,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    tokenizer_2=tokenizer_2,
    text_encoder_2=text_encoder_2,
    scheduler=scheduler,
    scale_factor=0,         # Tune this
    scale_range=(0.5, 0.2),   # Tune this
    loss_weight_style=0.4,    # Tune this
    loss_weight_semantic=0.6, # Tune this
    attn_res=32               # Match expected attention map size (sqrt(latent_pixels))
)

# Generate an image
prompt = "A close up view of a child admiring the fireworks on the sky at night"
prompt_2 = prompt # SDXL often uses the same prompt for both encoders, but can differ
negative_prompt = "worst quality, low quality, bad hands, deformed limbs, extra limbs, blurry, text, watermark, signature"
negative_prompt_2 = negative_prompt

generator = torch.Generator(device=device).manual_seed(1111) # Seed for reproducibility

print("Starting image generation...")
image = balancer.sample(
    prompt=prompt,
    prompt_2=prompt_2,
    negative_prompt=negative_prompt,
    negative_prompt_2=negative_prompt_2,
    num_inference_steps=10, # Fewer steps often work well with advanced schedulers
    guidance_scale=7.0,
    height=1024, # Use SDXL native resolution
    width=1024,
    generator=generator,
)

# Save the image
output_filename = "fused_generation_sdxl_v2.png"
image.save(output_filename)

# show image
try:
    from IPython.display import display
    print("图像如下:")
    display(image)
except ImportError:
    pass

print(f"Generation complete! Image saved as {output_filename}")

