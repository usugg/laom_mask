"""
LAOM Training with DINOv2 Features

This script trains LAOM using DINOv2 features instead of raw pixels.
Pipeline: 64x64 → [mask] → 224x224 → DINOv2 → 768-dim features → LAOM
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

from src.augmentations import Augmenter
from src.nn import LAOMDINOv2
from src.dinov2_encoder import DINOv2Encoder
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSLAOMHFDataset,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    soft_update,
    upscale_image,
    preprocess_for_dinov2,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LAOMConfig:
    num_epochs: int = 10
    batch_size: int = 256
    use_aug: bool = False  # Augmentation before DINOv2 (not recommended)
    future_obs_offset: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    grad_norm: Optional[float] = None
    
    # DINOv2 settings
    use_dinov2: bool = True
    dinov2_model_name: str = "facebook/dinov2-base"
    upscale_size: int = 224
    freeze_dinov2: bool = True
    
    # LAOM architecture (for DINOv2 features)
    feature_dim: int = 768  # DINOv2-base output
    latent_action_dim: int = 128
    act_head_dim: int = 1024
    act_head_dropout: float = 0.0
    obs_head_dim: int = 1024
    obs_head_dropout: float = 0.0
    target_tau: float = 0.001
    target_update_every: int = 1
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


@dataclass
class Config:
    project: str = "laom"
    group: str = "laom_dinov2_hf"
    name: str = "laom_dinov2"
    seed: int = 0

    lapo: LAOMConfig = field(default_factory=LAOMConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def extract_dinov2_features(images, dinov2_encoder, config):
    """
    Extract DINOv2 features from images.
    
    Args:
        images: Tensor of shape (B, H, W, C) in range [0, 255], uint8
                C can be 3 (single frame) or 9 (3 stacked frames)
        dinov2_encoder: DINOv2Encoder instance
        config: LAOMConfig
    
    Returns:
        features: Tensor of shape (B, feature_dim) or (B, feature_dim * num_frames) for stacked frames
    """
    # Convert from (B, H, W, C) to (B, C, H, W)
    images = images.permute(0, 3, 1, 2)
    
    batch_size, channels, height, width = images.shape
    
    # Check if we have stacked frames
    if channels == 9:  # 3 stacked RGB frames
        num_frames = 3
        # Split into individual frames: (B, 9, H, W) → (B, 3, 3, H, W) → (B*3, 3, H, W)
        images = images.reshape(batch_size, num_frames, 3, height, width)
        images = images.reshape(batch_size * num_frames, 3, height, width)
    elif channels == 3:  # Single RGB frame
        num_frames = 1
    else:
        raise ValueError(f"Unexpected number of channels: {channels}. Expected 3 or 9.")
    
    # Upscale to 224x224
    images_upscaled = upscale_image(images, target_size=config.upscale_size)
    
    # Preprocess for DINOv2 (ImageNet normalization)
    images_preprocessed = preprocess_for_dinov2(images_upscaled)
    
    # Extract features
    features = dinov2_encoder(images_preprocessed)  # (B*num_frames, feature_dim)
    
    # Reshape back to batch: (B*num_frames, feature_dim) → (B, num_frames * feature_dim)
    if num_frames > 1:
        features = features.reshape(batch_size, num_frames * dinov2_encoder.feature_dim)
    
    return features


def train_laom_dinov2(config: LAOMConfig):
    # Load DINOv2 encoder
    print(f"Loading DINOv2 encoder: {config.dinov2_model_name}")
    dinov2_encoder = DINOv2Encoder(
        model_name=config.dinov2_model_name,
        freeze=config.freeze_dinov2,
        device=DEVICE,
    )
    
    # Load dataset
    if config.use_hf_dataset:
        print(f"Loading HuggingFace dataset: {config.hf_dataset_name}/{config.hf_config_name}")
        dataset = DCSLAOMHFDataset(
            dataset_name=config.hf_dataset_name,
            config_name=config.hf_config_name,
            split=config.hf_split,
            frame_stack=config.frame_stack,
            max_offset=config.future_obs_offset,
            streaming=config.hf_streaming,
            buffer_size=config.hf_buffer_size,
            use_masked_obs=config.use_masked_obs,
            device=DEVICE,
        )
        dataloader = DataLoader(dataset, batch_size=config.batch_size)
        steps_per_epoch = 10_000_000 // (10 * config.batch_size)
    else:
        raise NotImplementedError("HDF5 dataset not supported for DINOv2 mode yet")
    
    # Create LAOM model for DINOv2 features
    laom = LAOMDINOv2(
        feature_dim=config.feature_dim,
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
    ).to(DEVICE)
    
    target_laom = deepcopy(laom)
    for p in target_laom.parameters():
        p.requires_grad_(False)
    
    print(f"LAOM model:")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  Latent action dim: {config.latent_action_dim}")
    print(f"  Act head dim: {config.act_head_dim}")
    print(f"  Obs head dim: {config.obs_head_dim}")
    
    optim = torch.optim.Adam(
        params=get_optim_groups(laom, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    
    # Probes
    state_probe = nn.Linear(config.feature_dim, dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)
    
    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)
    
    state_act_linear_probe = nn.Linear(config.feature_dim, dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)
    
    # Scheduler
    total_updates = steps_per_epoch * config.num_epochs
    warmup_updates = steps_per_epoch * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)
    
    start_time = time.time()
    total_iterations = 0
    total_tokens = 0
    
    for epoch in trange(config.num_epochs, desc="Epochs"):
        laom.train()
        epoch_steps = 0
        
        for i, batch in enumerate(dataloader):
            total_tokens += config.batch_size
            total_iterations += 1
            epoch_steps += 1
            
            obs, next_obs, future_obs, actions, states, offset = [b.to(DEVICE) for b in batch]
            
            # Extract DINOv2 features
            with torch.no_grad():
                obs_features = extract_dinov2_features(obs, dinov2_encoder, config)
                next_obs_features = extract_dinov2_features(next_obs, dinov2_encoder, config)
                future_obs_features = extract_dinov2_features(future_obs, dinov2_encoder, config)
            
            # LAOM forward pass (in feature space)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                latent_next_obs, latent_action, obs_hidden = laom(obs_features, future_obs_features)
                
                with torch.no_grad():
                    next_obs_target = target_laom.feature_norm(next_obs_features)
                
                loss = F.mse_loss(latent_next_obs, next_obs_target.detach())
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(laom.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            
            if i % config.target_update_every == 0:
                soft_update(target_laom, laom, tau=config.target_tau)
            
            # Update probes
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, states)
            
            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()
            
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, actions)
            
            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()
            
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, actions)
            
            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()
            
            wandb.log(
                {
                    "laom_dinov2/mse_loss": loss.item(),
                    "laom_dinov2/state_probe_mse_loss": state_probe_loss.item(),
                    "laom_dinov2/action_probe_mse_loss": act_probe_loss.item(),
                    "laom_dinov2/state_action_probe_mse_loss": state_act_probe_loss.item(),
                    "laom_dinov2/throughput": total_tokens / (time.time() - start_time),
                    "laom_dinov2/learning_rate": scheduler.get_last_lr()[0],
                    "laom_dinov2/grad_norm": get_grad_norm(laom).item(),
                    "laom_dinov2/target_obs_norm": torch.norm(next_obs_target, p=2, dim=-1).mean().item(),
                    "laom_dinov2/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                    "laom_dinov2/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                    "laom_dinov2/epoch": epoch,
                    "laom_dinov2/total_steps": total_iterations,
                }
            )
            
            # For streaming datasets, limit steps per epoch
            if config.use_hf_dataset and epoch_steps >= steps_per_epoch:
                break
    
    return laom, dinov2_encoder


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
    
    # Train LAOM with DINOv2
    laom, dinov2_encoder = train_laom_dinov2(config=config.lapo)
    
    run.finish()
    return laom, dinov2_encoder


if __name__ == "__main__":
    train()
