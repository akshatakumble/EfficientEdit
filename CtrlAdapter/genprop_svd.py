import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List

# Imports from your file structure
from unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from controlnet import ControlNetModel
from adapter_spatial_temporal import AdapterSpatioTemporal
from utils_svd import _convert_to_karras, sample_svd_sigmas_timesteps

class MaskPredictionDecoder(nn.Module):
    """
    GenProp Mask Prediction Decoder (MPD).
    Predicts the edited region mask from UNet features.
    """
    def __init__(self, in_channels=1280, out_channels=1, num_frames=25):
        super().__init__()
        self.num_frames = num_frames
        
        # Modified decoder to handle low channel counts (like 4 for latents)
        # We project up to 32 channels first, then apply Norm/Act.
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), # 8 groups for 32 channels is safe
            nn.SiLU(),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Output [0, 1] probability map
        )

    def forward(self, hidden_states):
        # hidden_states: [Batch, Frames, C, H, W] (from UNet output)
        b, f, c, h, w = hidden_states.shape
        
        # Reshape to [Batch, C, Frames, H, W] for 3D Conv
        x = hidden_states.permute(0, 2, 1, 3, 4)
        
        mask_logits = self.decoder(x) # [B, 1, F, H, W]
        
        # Permute back to [B, F, 1, H, W] for easier loss calc
        return mask_logits.permute(0, 2, 1, 3, 4)

class GenPropSVDXT(nn.Module):
    def __init__(
        self,
        unet: UNetSpatioTemporalConditionModel,
        controlnet: ControlNetModel,
        adapters: nn.ModuleList,
        num_inference_steps: int = 25
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.adapters = adapters
        self.num_inference_steps = num_inference_steps
        
        # Freeze backbone and ControlNet
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.adapters.requires_grad_(True)
        
        # MPD is trainable
        self.mpd = MaskPredictionDecoder(in_channels=4, num_frames=unet.config.num_frames)
        self.mpd.requires_grad_(True) 

        # Zero Convolutions
        self.zero_convs = nn.ModuleList([
            self._make_zero_conv(320),
            self._make_zero_conv(640),
            self._make_zero_conv(1280) 
        ])

        # SVD Sigmas
        sigmas_svd = _convert_to_karras(num_intervals=200000) 
        sigmas_svd = np.flip(sigmas_svd).copy()
        self.register_buffer("sigmas_svd", torch.from_numpy(sigmas_svd).to(dtype=torch.float32))

    def _make_zero_conv(self, channels):
        conv = nn.Conv2d(channels, channels, 1)
        conv.weight.data.zero_()
        conv.bias.data.zero_()
        return conv

    def get_inverse_timesteps(self, batch_size, device):
        # Calculate u/sigma for a single item (B=1)
        u_single, sigmas_single = sample_svd_sigmas_timesteps(1, self.sigmas_svd.cpu(), self.num_inference_steps)
        
        # Ensure single outputs are 0D or 1D tensors
        if isinstance(u_single, torch.Tensor):
            u_single = u_single.flatten().clone().detach().to(device)
            sigmas_single = sigmas_single.flatten().clone().detach().to(device)
        else:
            u_single = torch.tensor(u_single, device=device)
            sigmas_single = torch.tensor(sigmas_single, device=device)
        
        # Explicitly repeat tensors to match batch_size (B)
        u = u_single.repeat(batch_size)
        sigmas = sigmas_single.repeat(batch_size)
        
        t_controlnet = (u * 1000).round().long()
        t_svd = (0.25 * sigmas.log())
        
        # Ensure t_svd is 1D tensor of size B
        if t_svd.dim() == 0: t_svd = t_svd.unsqueeze(0)
            
        return t_svd, t_controlnet, sigmas

    def extract_control_features(self, video_pixels, t_controlnet, batch_size, num_frames, h, w, dtype):
        """
        Runs ControlNet to extract features (The SCE part).
        """
        video_pixels_flat = video_pixels.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, 3, h * 8, w * 8)
        
        dummy_latents = torch.zeros(batch_size * num_frames, 4, h, w, device=video_pixels.device, dtype=dtype)
        
        down_res, mid_res = self.controlnet(
            sample=dummy_latents,
            timestep=t_controlnet.repeat_interleave(num_frames),
            encoder_hidden_states=torch.zeros(batch_size * num_frames, 77, 768).to(video_pixels.device, dtype),
            controlnet_cond=video_pixels_flat,
            return_dict=False,
            skip_conv_in=True,
            skip_time_emb=True
        )
        return down_res, mid_res


    def adapt_features(self, down_res, mid_res, timesteps_svd, image_embeddings, num_frames: int):
        """
        Run Ctrl-Adapter to project ControlNet features into SVD UNet space.

        Returns:
            A list/tuple of additional residuals, same length as `down_res`,
            where each element is 5D [B, C, F, H, W].

        The UNet will internally flatten these as
            [B, C, F, H, W] -> [(B * F), C, H, W]
        using its own (batch_size, num_frames) derived from the input video
        latents, so we must match that B and F here.
        """
        device = down_res[0].device
        dtype = down_res[0].dtype

        # -------------------------------------------------------------
        # 1. Infer TRUE batch_size and num_frames from ControlNet features
        # -------------------------------------------------------------
        batch_frames = down_res[0].shape[0]              # = B * F_control
        batch_size = image_embeddings.shape[0]           # = B (CLIP batch)

        if batch_frames % batch_size != 0:
            raise ValueError(
                f"down_res batch ({batch_frames}) not divisible "
                f"by image_embeddings batch ({batch_size})"
            )

        num_frames_eff = batch_frames // batch_size      # F_control

        # -------------------------------------------------------------
        # 2. Normalize timesteps_svd -> [batch_frames]
        # -------------------------------------------------------------
        temb = timesteps_svd
        if not torch.is_tensor(temb):
            temb = torch.tensor(temb, device=device, dtype=torch.float32)
        else:
            temb = temb.to(device=device, dtype=torch.float32)

        temb = temb.flatten()
        n = temb.numel()

        if n == 1:
            # single scalar -> same t for all B*F
            temb_bf = temb.repeat(batch_frames)
        elif n == batch_size:
            # one per video -> repeat over frames
            temb_bf = temb.repeat_interleave(num_frames_eff)
        elif n == num_frames_eff:
            # one per frame (B=1 case in sampling)
            if batch_size == 1:
                temb_bf = temb  # [F] == [batch_frames]
            else:
                temb_bf = temb.view(1, num_frames_eff).repeat(batch_size, 1).reshape(-1)
        elif n == batch_frames:
            # already [B*F]
            temb_bf = temb
        else:
            raise ValueError(
                f"Unexpected timesteps_svd length {n}. Expected 1, "
                f"batch_size ({batch_size}), num_frames ({num_frames_eff}), "
                f"or batch_frames ({batch_frames})."
            )

        temb_bf = temb_bf.to(dtype=dtype)

        # -------------------------------------------------------------
        # 3. Expand image_embeddings to [B*F, seq_len, dim]
        # -------------------------------------------------------------
        encoder_hidden_states = image_embeddings.to(device=device, dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames_eff, dim=0)

        # -------------------------------------------------------------
        # 4. Adapt chosen ControlNet levels
        # -------------------------------------------------------------
        controlnet_indices = [2, 5, 8]  # must match len(self.adapters) & self.zero_convs
        assert len(controlnet_indices) == len(self.adapters) == len(self.zero_convs)

        idx_to_adapted_4d = {}

        for adapter_idx, dn_idx in enumerate(controlnet_indices):
            base_feat = down_res[dn_idx]  # [B*F, C_in, H_in, W_in]

            adapted = self.adapters[adapter_idx](
                hidden_states=base_feat,
                num_frames=num_frames_eff,
                timestep=temb_bf,
                encoder_hidden_states=encoder_hidden_states,
            )  # [B*F, C_out, H', W']

            adapted = self.zero_convs[adapter_idx](adapted)  # still [B*F, C_out, H', W']

            # Match spatial size to ControlNet feature at dn_idx (defensive)
            target = base_feat
            if adapted.shape[-2:] != target.shape[-2:]:
                adapted = F.interpolate(
                    adapted,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Match channels to ControlNet feature at dn_idx (defensive)
            if adapted.shape[1] != target.shape[1]:
                if adapted.shape[1] > target.shape[1]:
                    adapted = adapted[:, : target.shape[1], ...]
                else:
                    pad_ch = target.shape[1] - adapted.shape[1]
                    pad = torch.zeros(
                        adapted.shape[0],
                        pad_ch,
                        *adapted.shape[2:],
                        device=device,
                        dtype=adapted.dtype,
                    )
                    adapted = torch.cat([adapted, pad], dim=1)

            idx_to_adapted_4d[dn_idx] = adapted  # [B*F, C, H, W]

        # -------------------------------------------------------------
        # 5. Build full list as 5D [B, C, F, H, W]
        #    so that UNet can flatten using its own (batch_size, num_frames).
        # -------------------------------------------------------------
        aligned_residuals_5d = []

        for dn_idx, base_feat in enumerate(down_res):
            # base_feat: [B*F, C_b, H_b, W_b]
            bf, c_b, h_b, w_b = base_feat.shape

            if dn_idx in idx_to_adapted_4d:
                adapted_4d = idx_to_adapted_4d[dn_idx]  # [B*F, C_b, H_b, W_b]
            else:
                adapted_4d = torch.zeros_like(base_feat)

            # reshape [B*F, C, H, W] -> [B, C, F, H, W]
            adapted_5d = adapted_4d.view(batch_size, num_frames_eff, c_b, h_b, w_b)
            adapted_5d = adapted_5d.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            aligned_residuals_5d.append(adapted_5d)

        return aligned_residuals_5d



    def forward(
        self,
        noisy_latents,
        timesteps_svd,
        t_controlnet,
        image_embeddings,
        original_video_pixels,
        added_time_ids,
    ):
        noisy_latents_unet = noisy_latents.permute(0, 2, 1, 3, 4)
        batch_size, num_frames, _, h, w = noisy_latents_unet.shape
        dtype = noisy_latents.dtype

        # 2. SCE (ControlNet)
        down_res, mid_res = self.extract_control_features(
            original_video_pixels, t_controlnet, batch_size, num_frames, h, w, dtype
        )

        # 3. Adapter
        adapted_residuals = self.adapt_features(down_res, mid_res, timesteps_svd, image_embeddings, num_frames)

        # 4. SVD UNet
        unet_output = self.unet(
            sample=noisy_latents_unet, 
            timestep=timesteps_svd,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=adapted_residuals,
            return_dict=True
        )
        
        # 5. Mask Prediction (MPD)
        mask_pred = self.mpd(unet_output.sample) 

        return unet_output.sample, mask_pred