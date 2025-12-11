import argparse
import os
import gc
import shutil
import subprocess
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import numpy as np
import imageio
import wandb

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# Import local modules
from dataset import GenPropDataset
from genprop_svd import GenPropSVDXT
from unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from controlnet import ControlNetModel
from adapter_spatial_temporal import AdapterSpatioTemporal

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GenProp with Ctrl-Adapter on SVD-XT")
    
    # Data & Directories
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--val_output_dir", type=str, default="./val_videos")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    
    # Model Params
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=25)
    
    # Validation & Saving
    parser.add_argument("--validate_every_n_epochs", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Steps for validation sampling")
    
    # Remote Storage (Rclone)
    parser.add_argument("--rclone_remote_name", type=str, default=None)
    parser.add_argument("--rclone_remote_dir", type=str, default="GenProp_Runs")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="genprop")
    parser.add_argument("--wandb_run", type=str, default="run-01")
    
    # GenProp Loss Weights
    parser.add_argument("--lambda_mask", type=float, default=2.0)
    parser.add_argument("--beta_grad", type=float, default=1.0)
    parser.add_argument("--gamma_mpd", type=float, default=1.0)

    return parser.parse_args()

def rclone_upload(local_path, remote_name, remote_dir):
    """Uploads a folder to remote and deletes local copy."""
    if not remote_name:
        return
    
    dir_name = os.path.basename(local_path)
    remote_path = f"{remote_name}:{remote_dir}/{dir_name}"
    
    logger.info(f"Uploading {local_path} to {remote_path} via rclone...")
    try:
        subprocess.run(["rclone", "copy", local_path, remote_path], check=True)
        logger.info("Upload successful.")
        shutil.rmtree(local_path)
        logger.info(f"Deleted local checkpoint: {local_path}")
    except Exception as e:
        logger.error(f"Rclone failed: {e}")

def save_videos(frames, path, fps=8):
    """frames: [F, C, H, W] tensor in [0, 1]"""
    frames = (frames * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    imageio.mimsave(path, frames, fps=fps)

def main():
    args = parse_args()
    
    # 1. Accelerate Setup
    acc_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=acc_project_config
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.val_output_dir, exist_ok=True)
    
    set_seed(args.seed)

    # 2. Init WandB
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
#            init_kwargs={"wandb": {"name": args.wandb_run}}
        )

    # 3. Load Models
    pretrained_model_name = "stabilityai/stable-video-diffusion-img2vid-xt"
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_name, subfolder="vae")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name, subfolder="image_encoder")
    
    unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile")
    
    # Adapters
    # SVD-XT down blocks map to 320, 640, 1280
    adapters = nn.ModuleList([
        AdapterSpatioTemporal(in_channels=320, out_channels=320, num_layers=1),
        AdapterSpatioTemporal(in_channels=640, out_channels=640, num_layers=1),
        AdapterSpatioTemporal(in_channels=1280, out_channels=1280, num_layers=1)
    ])

    model = GenPropSVDXT(unet, controlnet, adapters, num_inference_steps=25)
    
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    vae.to(accelerator.device)
    image_encoder.to(accelerator.device)

    # 4. Optimizer
    params_to_optimize = list(model.adapters.parameters()) + list(model.mpd.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)

    # 5. Dataset
    train_dataset = GenPropDataset(
        data_root=args.data_root,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_dataset = GenPropDataset(
        data_root=args.val_dir,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 6. Prepare
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # SVD Noise Scheduler
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

    # 7. Training Loop
    global_step = 0
    torch.cuda.empty_cache()
    gc.collect()
    
    # Clip stats for normalization
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(accelerator.device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(accelerator.device)

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch["edited_video"].to(accelerator.device)
                original_pixels = batch["original_video"].to(accelerator.device)
                mask = batch["mask_video"].to(accelerator.device)
                
                batch_size, channels, num_frames, h, w = pixel_values.shape

                # --- Prepare Latents ---
                pixel_values_reshaped = pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, channels, h, w)
                latents = vae.encode(pixel_values_reshaped).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.reshape(batch_size, num_frames, 4, h // 8, w // 8).permute(0, 2, 1, 3, 4) # [B, C, F, H_lat, W_lat]

                # --- Prepare Conditional Image Latents (SVD 8-channel Input) ---
                first_frames = pixel_values[:, :, 0, :, :] # [B, C, H, W]
                # Encode first frame (4D input expected)
                cond_latents = vae.encode(first_frames).latent_dist.mode() # [B, 4, h/8, w/8]
                cond_latents = cond_latents * vae.config.scaling_factor
                # Repeat for temporal dimension to match video latents shape [B, 4, F, h, w]
                cond_latents = cond_latents.unsqueeze(2).repeat(1, 1, num_frames, 1, 1) # [B, C, F, H_lat, W_lat]

                # --- Prepare CLIP Embeddings ---
                first_frames_norm = (first_frames + 1.0) / 2.0
                clip_input = F.interpolate(first_frames_norm, size=(224, 224), mode="bicubic", align_corners=False)
                clip_input = (clip_input - clip_mean) / clip_std
                image_embeddings = image_encoder(clip_input).image_embeds.unsqueeze(1)

                # --- Time IDs ---
                added_time_ids = torch.tensor([6, 127, 0.02]).to(accelerator.device).repeat(batch_size, 1)

                # --- Noise Sampling ---
                unwrapped_model = accelerator.unwrap_model(model)
                t_svd, t_controlnet, sigmas = unwrapped_model.get_inverse_timesteps(batch_size, accelerator.device)
                
                noise = torch.randn_like(latents)
                sigmas_reshaped = sigmas.view(batch_size, 1, 1, 1, 1)
                noisy_latents = latents + noise * sigmas_reshaped # [B, C, F, H_lat, W_lat]
                inp_noisy_latents = noisy_latents / ((sigmas_reshaped**2 + 1) ** 0.5)

                # --- Concatenate for SVD Input (8 Channels) ---
                unet_input = torch.cat([inp_noisy_latents, cond_latents], dim=1)

                # --- Forward ---
                original_video_norm = (original_pixels + 1.0) / 2.0
                
                model_pred_raw, mask_pred = model(
                    noisy_latents=unet_input, 
                    timesteps_svd=t_svd,
                    t_controlnet=t_controlnet,
                    image_embeddings=image_embeddings,
                    original_video_pixels=original_video_norm,
                    added_time_ids=added_time_ids
                )
                
                # --- FIX: Align model_pred shape [B, F, C, H, W] to target shape [B, C, F, H, W] ---
                model_pred = model_pred_raw.permute(0, 2, 1, 3, 4)

                # --- Loss ---
                # Diffusion Loss
                alpha = 1 / ((sigmas_reshaped**2 + 1) ** 0.5)
                sigma_coeff = sigmas_reshaped / ((sigmas_reshaped**2 + 1) ** 0.5)
                target_v = alpha * noise - sigma_coeff * latents # [B, C, F, H_lat, W_lat]
                
                # --- MASK INTERPOLATION FIX ---
                mask_flat = mask.permute(0, 2, 1, 3, 4).reshape(-1, 1, h, w)
                
                mask_latent = F.interpolate(
                    mask_flat, 
                    size=(latents.shape[-2], latents.shape[-1]), 
                    mode="nearest" 
                )
                mask_latent = mask_latent.reshape(batch_size, num_frames, 1, latents.shape[-2], latents.shape[-1]).permute(0, 2, 1, 3, 4)
                
                loss_pixel = F.mse_loss(model_pred, target_v, reduction="none")
                loss_mask = (loss_pixel * mask_latent).mean()
                loss_non_mask = (loss_pixel * (1 - mask_latent)).mean()
                loss_diffusion = loss_non_mask + args.lambda_mask * loss_mask

                # MPD Loss
                mask_target_mpd = F.interpolate(
                    mask_flat, 
                    size=mask_pred.shape[-2:], 
                    mode="nearest"
                ).reshape(batch_size, num_frames, 1, *mask_pred.shape[-2:])  # [B, F, 1, H, W]
                
                loss_mpd = F.mse_loss(mask_pred, mask_target_mpd)

                # Gradient Loss
                perturbation = torch.randn_like(original_video_norm) * 0.01 * mask
                perturbed_input = original_video_norm + perturbation
                
                dtype = unet_input.dtype 
                
                down_orig, _ = unwrapped_model.extract_control_features(
                    original_video_norm, t_controlnet, batch_size, num_frames, h//8, w//8, dtype
                )
                down_pert, _ = unwrapped_model.extract_control_features(
                    perturbed_input, t_controlnet, batch_size, num_frames, h//8, w//8, dtype
                )
                
                loss_grad = 0
                for idx in [0, 4, 7]: 
                    loss_grad += F.mse_loss(down_orig[idx], down_pert[idx])

                loss = loss_diffusion + args.gamma_mpd * loss_mpd + args.beta_grad * loss_grad

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                accelerator.log({
                    "loss": loss.item(), "loss_mask": loss_mask.item(), "loss_grad": loss_grad.item()
                }, step=global_step)

        # --- Validation (Full SVD Sampling) ---
        if (epoch + 1) % args.validate_every_n_epochs == 0:
            if accelerator.is_main_process:
                logger.info(f"Running Validation for Epoch {epoch+1}")
                model.eval()
                unwrapped_model = accelerator.unwrap_model(model)
                
                with torch.no_grad():
                    for i, batch in enumerate(val_dataloader):
                        if i >= 4: break # Generate 4 videos max
                        
                        orig = batch["original_video"].to(accelerator.device)
                        orig_norm = (orig + 1.0) / 2.0 
                        
                        target_video = batch["edited_video"].to(accelerator.device) 
                        first_frame = (target_video[:, :, 0, :, :] + 1.0) / 2.0 
                        
                        # Encode First Frame (Condition)
                        clip_input = F.interpolate(first_frame, size=(224, 224), mode="bicubic", align_corners=False)
                        clip_input = (clip_input - clip_mean) / clip_std
                        image_embeddings = image_encoder(clip_input).image_embeds.unsqueeze(1)
                        
                        # ------------------------------------------------------------------
                        # Prepare conditional latents (first frame of the edited video)
                        # ------------------------------------------------------------------
                        image_latents_raw = vae.encode(target_video[:, :, 0, :, :]).latent_dist.mode()
                        image_latents_raw = image_latents_raw * vae.config.scaling_factor

                        # --- FIX: ensure latents are 4D [B, 4, H, W] ---
                        # Some VAEs output [B, 4, T_lat, H, W], so collapse T_lat.
                        if image_latents_raw.dim() == 5:
                            # take first temporal slice (recommended)
                            image_latents_raw = image_latents_raw[:, :, 0, :, :]
                            # OR: average across time: image_latents_raw = image_latents_raw.mean(dim=2)

                        # At this point: image_latents_raw is [B, 4, H_lat, W_lat]

                        # --- Expand into a video of length args.num_frames ---
                        # Result: [B, 4, F, H_lat, W_lat]
                        cond_latents = image_latents_raw.unsqueeze(2).repeat(1, 1, args.num_frames, 1, 1)
                        latents      = image_latents_raw.unsqueeze(2).repeat(1, 1, args.num_frames, 1, 1)

                        # ------------------------------------------------------------------
                        # Prepare noise for diffusion
                        # ------------------------------------------------------------------
                        noise = torch.randn_like(latents)

                        # set scheduler timesteps
                        noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)

                        # initial noise injection uses the *first* scheduler timestep
                        initial_timestep = noise_scheduler.timesteps[0].to(accelerator.device).unsqueeze(0)

                        latents = noise_scheduler.add_noise(latents, noise, initial_timestep)

                        # time ids (unchanged)
                        added_time_ids = torch.tensor([6, 127, 0.02]).to(accelerator.device).repeat(1, 1)


                        # Sampling Loop
                        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
                            u_val = 1.0 - (t.item() / noise_scheduler.config.num_train_timesteps)
                            t_cnet = torch.tensor([(1.0 - u_val) * 1000]).long().to(accelerator.device).clamp(0, 999)
                            
                            down_res, mid_res = unwrapped_model.extract_control_features(
                                orig_norm, t_cnet, 1, args.num_frames, args.height//8, args.width//8, orig.dtype
                            )
                            
                            t_for_adapter = t.to(accelerator.device)  # scalar timestep
                            adapted_residuals = unwrapped_model.adapt_features(
                                down_res, mid_res, t_for_adapter, image_embeddings, args.num_frames
                            )
                            latent_model_input = noise_scheduler.scale_model_input(latents, t)
                            # latent_model_input: [B, C_lat, F, H_lat, W_lat]
                            # cond_latents:      [B, C_cond, F, H_lat, W_lat]

                            B, C_lat, F_, H_lat, W_lat = latent_model_input.shape
                            B2, C_cond, F2, H_cond, W_cond = cond_latents.shape
                            assert B == B2 and F_ == F2 and H_lat == H_cond and W_lat == W_cond, \
                                f"Shape mismatch between noise latents {latent_model_input.shape} and cond_latents {cond_latents.shape}"

                            # If for any reason channels are 1, expand to 4 (defensive)
                            if C_lat == 1:
                                latent_model_input = latent_model_input.repeat(1, 4, 1, 1, 1)
                                C_lat = 4
                            if C_cond == 1:
                                cond_latents = cond_latents.repeat(1, 4, 1, 1, 1)
                                C_cond = 4

                            # Now we expect 4 + 4 = 8 channels for SVD-XT
                            unet_input = torch.cat([latent_model_input, cond_latents], dim=1)  # [B, C_lat + C_cond, F, H_lat, W_lat]
                            assert unet_input.shape[1] == unwrapped_model.unet.config.in_channels, \
                                f"UNet in_channels={unwrapped_model.unet.config.in_channels}, but got {unet_input.shape[1]}"

                            # UNet expects [B, F, C, H, W]
                            unet_input_permuted = unet_input.permute(0, 2, 1, 3, 4)  # [B, F, 8, H_lat, W_lat]
                            
                            
                            #print("\nUNet sample:", unet_input_permuted.shape)             # expect [B, F, 8, H, W]
                            #print("\nAdapted[0]:", adapted_residuals[0].shape)
                            

                            noise_pred_raw = unwrapped_model.unet(
                                sample=unet_input_permuted,
                                timestep=t.unsqueeze(0),
                                encoder_hidden_states=image_embeddings,
                                added_time_ids=added_time_ids,
                                down_block_additional_residuals=adapted_residuals,
                                return_dict=False
                            )[0]
                            # Permute prediction back to [B, C, F, H, W] for scheduler step
                            noise_pred = noise_pred_raw.permute(0, 2, 1, 3, 4)
                            
                            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                        latents = latents / vae.config.scaling_factor
                        B, C, F_frames, H_lat, W_lat = latents.shape
                        latents_reshaped = latents.permute(0, 2, 1, 3, 4).reshape(B * F_frames, C, H_lat, W_lat)
                        frames = vae.decode(latents_reshaped, num_frames=F_frames).sample
                        
                        save_path = os.path.join(args.val_output_dir, f"epoch_{epoch+1}_sample_{i}.mp4")
                        frames = frames.reshape(args.num_frames, 3, args.height, args.width)
                        save_videos((frames + 1.0) / 2.0, save_path)
                
                model.train()

        # Saving & Uploading
        if (epoch + 1) % args.save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
                accelerator.save_state(save_path)
                
                if args.rclone_remote_name:
                    rclone_upload(save_path, args.rclone_remote_name, args.rclone_remote_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()