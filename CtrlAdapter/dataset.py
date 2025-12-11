import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path

# Try importing decord for efficient video loading
try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge('torch')
except ImportError:
    raise ImportError("Please install decord: pip install decord")

class GenPropDataset(Dataset):
    """
    Dataset for GenProp training.
    Expected structure:
    data_root/
        +-- original_videos/  (mp4)
        +-- edited_videos/    (mp4)
        +-- masks/            (mp4)
    
    Assumes filenames match across folders (e.g., vid1.mp4 in all three).
    """
    def __init__(
        self,
        data_root: str,
        width: int = 512,
        height: int = 512,
        num_frames: int = 25,
        sample_rate: int = 1,
    ):
        self.data_root = Path(data_root)
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        
        self.original_dir = self.data_root / "original_videos"
        self.edited_dir = self.data_root / "edited_videos"
        self.mask_dir = self.data_root / "masks"
        
        # Get list of video files
        # We assume edited_videos is the source of truth for filenames
        self.video_files = sorted([
            f.name for f in self.edited_dir.glob("*.mp4")
            if (self.original_dir / f.name).exists() and (self.mask_dir / f.name).exists()
        ])
        
        print(f"Found {len(self.video_files)} valid triplets in {data_root}")

        # Transforms
        # Resize logic: We resize to the smallest edge matches target, then crop
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(min(height, width)), 
            transforms.CenterCrop((height, width)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        # Mask transforms (No normalization, just [0, 1])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(min(height, width)),
            transforms.CenterCrop((height, width)),
        ])

    def __len__(self):
        return len(self.video_files)

    def _load_video(self, path):
        """Loads video and returns frames tensor [F, C, H, W]"""
        vr = VideoReader(str(path), ctx=cpu(0))
        total_frames = len(vr)
        
        # Sampling logic
        # If video is shorter than needed, loop it. If longer, sample strides.
        required_frames = self.num_frames * self.sample_rate
        
        if total_frames >= required_frames:
            # Random start index for temporal augmentation
            max_start_idx = total_frames - required_frames
            start_idx = random.randint(0, max_start_idx)
            frame_indices = np.arange(start_idx, start_idx + required_frames, self.sample_rate)
        else:
            # Not enough frames, repeat the indices
            frame_indices = np.arange(0, total_frames)
            # Pad by repeating the last frame
            remaining = self.num_frames - len(frame_indices)
            last_idx = frame_indices[-1]
            padding = [last_idx] * remaining
            frame_indices = np.concatenate([frame_indices, padding])
            # Handle sample rate edge case roughly by just taking first N
            frame_indices = frame_indices[:self.num_frames]

        # Get batch of frames
        frames = vr.get_batch(frame_indices) # [F, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0 # [F, C, H, W], range [0, 1]
        
        return frames

    def __getitem__(self, idx):
        filename = self.video_files[idx]
        
        edited_path = self.edited_dir / filename
        original_path = self.original_dir / filename
        mask_path = self.mask_dir / filename
        
        try:
            # Load all three
            edited_frames = self._load_video(edited_path)
            original_frames = self._load_video(original_path)
            mask_frames = self._load_video(mask_path)
            
            # Masks might be RGB, convert to grayscale 1 channel
            if mask_frames.shape[1] == 3:
                mask_frames = mask_frames[:, 0:1, :, :] # Take R channel
            
            # Ensure lengths match (in case of slight video file discrepancies)
            min_len = min(len(edited_frames), len(original_frames), len(mask_frames))
            edited_frames = edited_frames[:min_len]
            original_frames = original_frames[:min_len]
            mask_frames = mask_frames[:min_len]
            
            # Apply Spatial Transforms
            # Note: transforms expect [C, H, W], we have [F, C, H, W]
            # Normalize changes [0, 1] -> [-1, 1] for RGB
            edited_frames = self.pixel_transforms(edited_frames)
            original_frames = self.pixel_transforms(original_frames)
            mask_frames = self.mask_transforms(mask_frames)
            
            # Binarize mask (Soft edge masks are okay for RA loss, but usually we want binary for hard masking)
            # GenProp uses soft masks for effects, so we keep gradients but maybe threshold slightly
            mask_frames = (mask_frames > 0.5).float() 

            # Output Structure: [C, F, H, W] for Diffusers Pipeline usually
            # But the train script expects [B, C, F, H, W] after DataLoader batching
            # So here we return [C, F, H, W]
            
            return {
                "edited_video": edited_frames.permute(1, 0, 2, 3),   # [C, F, H, W]
                "original_video": original_frames.permute(1, 0, 2, 3), # [C, F, H, W]
                "mask_video": mask_frames.permute(1, 0, 2, 3),       # [1, F, H, W]
                "filename": filename
            }
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Fallback to random item to avoid crashing training
            return self.__getitem__(random.randint(0, len(self) - 1))