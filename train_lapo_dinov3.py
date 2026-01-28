"""
LAPO Training with DINOv3 Features

Pipeline: 64x64 -> upscale 224x224 -> DINOv3 patch pooling -> LAPO (feature space)
"""
import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange

from src.nn import LAPODINOv3
from src.dinov3_encoder import DINOv3Encoder
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSLAPOHFDataset,
    get_grad_norm,
    get_optim_groups,
    set_seed,
    upscale_image,
    preprocess_for_dinov2,  # ImageNet normalization works for DINOv3 too
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LAPOConfig:
    # Training parameters (matching lapo_hf.yaml)
    num_epochs: int = 10
    batch_size: int = 512
    future_obs_offset: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    grad_norm: Optional[float] = None
    
    # DINOv3 settings
    use_dinov3: bool = True
    dinov3_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    upscale_size: int = 224
    freeze_dinov3: bool = True
    pooling_mode: str = "topk_variance"  # 'avg', 'adaptive_4x4', 'flatten', 'cls', 'topk_variance'
    topk: int = 32  # Number of patches to keep for topk_variance mode
    
    # Model architecture
    feature_dim: int = 225792  # Will be overridden by encoder.output_dim * frame_stack
    latent_action_dim: int = 128
    act_head_dim: int = 1024
    act_head_dropout: float = 0.0
    obs_head_dim: int = 1024
    obs_head_dropout: float = 0.0
    frame_stack: int = 3
    
    # Data source configuration  
    use_hf_dataset: bool = True
    data_path: str = "data/test.hdf5"
    hf_dataset_name: str = "EpicPinkPenguin/visual_distracting_control_suite"
    hf_config_name: str = "cheetah_run"
    hf_split: str = "train"
    hf_streaming: bool = True
    hf_buffer_size: int = 10000
    use_masked_obs: bool = False
    clip_actions: bool = False


@dataclass
class Config:
    project: str = "laom"
    group: str = "lapo_dinov3_hf"
    name: str = "lapo_dinov3"
    seed: int = 0

    lapo: LAPOConfig = field(default_factory=LAPOConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def extract_dinov3_features(images, dinov3_encoder, config):
    """
    Extract DINOv3 features from images using patch pooling.
    
    Args:
        images: Tensor of shape (B, H, W, C) in range [0, 255], uint8
                C can be 3 (single frame) or 9 (3 stacked frames)
        dinov3_encoder: DINOv3Encoder instance
        config: LAPOConfig
    
    Returns:
        features: Tensor of shape (B, feature_dim) where feature_dim = 384 * num_frames
    """
    # Convert from (B, H, W, C) to (B, C, H, W)
    images = images.permute(0, 3, 1, 2)
    
    batch_size, channels, height, width = images.shape
    
    # Check if we have stacked frames
    if channels == 9:  # 3 stacked RGB frames
        num_frames = 3
        # Split into individual frames: (B, 9, H, W) -> (B, 3, 3, H, W) -> (B*3, 3, H, W)
        images = images.reshape(batch_size, num_frames, 3, height, width)
        images = images.reshape(batch_size * num_frames, 3, height, width)
    elif channels == 3:  # Single RGB frame
        num_frames = 1
    else:
        raise ValueError(f"Unexpected number of channels: {channels}. Expected 3 or 9.")
    
    # Upscale to 224x224
    images_upscaled = upscale_image(images, target_size=config.upscale_size)
    
    # Preprocess for DINOv3 (ImageNet normalization)
    images_preprocessed = preprocess_for_dinov2(images_upscaled)
    
    # Extract features (with patch pooling)
    features = dinov3_encoder(images_preprocessed)  # (B*num_frames, 384)
    
    # Reshape back to batch: (B*num_frames, output_dim) -> (B, num_frames * output_dim)
    if num_frames > 1:
        features = features.reshape(batch_size, num_frames * dinov3_encoder.output_dim)
    
    return features


def train_lapo_dinov3(config: LAPOConfig):
    # Load DINOv3 encoder
    print(f"Loading DINOv3 encoder: {config.dinov3_model_name}")
    dinov3_encoder = DINOv3Encoder(
        model_name=config.dinov3_model_name,
        freeze=config.freeze_dinov3,
        pooling_mode=config.pooling_mode,
        topk=config.topk,
        device=DEVICE,
    )
    
    # Calculate actual feature_dim from encoder
    actual_feature_dim = dinov3_encoder.output_dim * config.frame_stack
    print(f"Actual feature dimension: {actual_feature_dim} (encoder: {dinov3_encoder.output_dim} × {config.frame_stack} frames)")
    
    # Load dataset
    if config.use_hf_dataset:
        print(f"Loading HuggingFace dataset: {config.hf_dataset_name}/{config.hf_config_name}")
        dataset = DCSLAPOHFDataset(
            dataset_name=config.hf_dataset_name,
            config_name=config.hf_config_name,
            split=config.hf_split,
            frame_stack=config.frame_stack,
            max_offset=config.future_obs_offset,
            streaming=config.hf_streaming,
            buffer_size=config.hf_buffer_size,
            use_masked_obs=config.use_masked_obs,
            clip_actions=config.clip_actions,
            device=DEVICE,
        )
        dataloader = DataLoader(dataset, batch_size=config.batch_size)
        steps_per_epoch = 10_000_000 // (10 * config.batch_size)
    else:
        raise NotImplementedError("HDF5 dataset not supported for DINOv3 mode yet")
    
    # Create LAPO model for DINOv3 features
    lapo = LAPODINOv3(
        feature_dim=actual_feature_dim,
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
    ).to(DEVICE)
    
    print(f"LAPO-DINOv3 model:")
    print(f"  Feature dim: {actual_feature_dim}")
    print(f"  Latent action dim: {config.latent_action_dim}")
    print(f"  Act head dim: {config.act_head_dim}")
    print(f"  Obs head dim: {config.obs_head_dim}")
    
    optim = torch.optim.Adam(
        params=get_optim_groups(lapo, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    
    # Linear probe for action prediction
    linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    probe_optim = torch.optim.Adam(linear_probe.parameters(), lr=config.learning_rate)
    
    # Scheduler
    total_updates = steps_per_epoch * config.num_epochs
    warmup_updates = steps_per_epoch * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)
    
    start_time = time.time()
    total_iterations = 0
    total_tokens = 0
    
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        epoch_steps = 0
        
        for i, batch in enumerate(dataloader):
            total_tokens += config.batch_size
            total_iterations += 1
            epoch_steps += 1
            
            # Unpack batch - includes mask
            obs, next_obs, future_obs, actions, offset, mask = [b.to(DEVICE) for b in batch]
            
            # Extract DINOv3 features
            with torch.no_grad():
                obs_features = extract_dinov3_features(obs, dinov3_encoder, config)
                next_obs_features = extract_dinov3_features(next_obs, dinov3_encoder, config)
                future_obs_features = extract_dinov3_features(future_obs, dinov3_encoder, config)
            
            # LAPO forward pass (in feature space)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_next_obs, latent_action = lapo(obs_features, future_obs_features)
                
                # MSE loss between predicted and actual next observation features
                # Normalize target for stable training
                next_obs_target = lapo.feature_norm(next_obs_features)
                loss = F.mse_loss(pred_next_obs, next_obs_target.detach())
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            
            # Update linear probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = linear_probe(latent_action.detach())
                probe_loss = F.mse_loss(pred_action, actions)
            
            probe_optim.zero_grad(set_to_none=True)
            probe_loss.backward()
            probe_optim.step()
            
            wandb.log(
                {
                    "lapo_dinov3/mse_loss": loss.item(),
                    "lapo_dinov3/action_probe_mse_loss": probe_loss.item(),
                    "lapo_dinov3/throughput": total_tokens / (time.time() - start_time),
                    "lapo_dinov3/learning_rate": scheduler.get_last_lr()[0],
                    "lapo_dinov3/grad_norm": get_grad_norm(lapo).item(),
                    "lapo_dinov3/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                    "lapo_dinov3/epoch": epoch,
                    "lapo_dinov3/total_steps": total_iterations,
                }
            )
            
            # For streaming datasets, limit steps per epoch
            if config.use_hf_dataset and epoch_steps >= steps_per_epoch:
                break
    
    return lapo, dinov3_encoder


@pyrallis.wrap()
def train(config: Config):
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    set_seed(config.seed)
    
    # Train LAPO with DINOv3
    lapo, dinov3_encoder = train_lapo_dinov3(config=config.lapo)
    
    run.finish()
    return lapo, dinov3_encoder


if __name__ == "__main__":
    train()
