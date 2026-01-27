#!/usr/bin/env python
"""Test script for LAPO with DINOv3 integration"""
import torch
from src.dinov3_encoder import DINOv3Encoder
from src.nn import LAPODINOv3
from src.utils import upscale_image, preprocess_for_dinov2

print("="*60)
print("Testing LAPO with DINOv3 Integration")
print("="*60)

# Test 1: Image upscaling
print("\n1. Testing image upscaling...")
img_64 = torch.randn(2, 3, 64, 64)
img_224 = upscale_image(img_64, target_size=224)
print(f"   Input shape: {img_64.shape}")
print(f"   Upscaled shape: {img_224.shape}")
assert img_224.shape == (2, 3, 224, 224), "Upscaling failed!"
print("   ✓ Upscaling test passed!")

# Test 2: DINOv3 preprocessing
print("\n2. Testing DINOv3 preprocessing...")
img_uint8 = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
img_preprocessed = preprocess_for_dinov2(img_uint8)  # Same normalization works
print(f"   Input dtype: {img_uint8.dtype}, range: [0, 255]")
print(f"   Preprocessed dtype: {img_preprocessed.dtype}")
print(f"   Preprocessed range: [{img_preprocessed.min():.2f}, {img_preprocessed.max():.2f}]")
assert img_preprocessed.dtype == torch.float32, "Preprocessing dtype failed!"
print("   ✓ Preprocessing test passed!")

# Test 3: DINOv3 encoder
print("\n3. Testing DINOv3 encoder...")
print("   Loading DINOv3-ViT-S/16 model...")
dinov3 = DINOv3Encoder(
    model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
    freeze=True,
    use_patch_pooling=True
)
print(f"   Model loaded: {dinov3.model_name}")
print(f"   Feature dim: {dinov3.feature_dim}")
print(f"   Frozen: {dinov3.is_frozen}")
print(f"   Patch pooling: {dinov3.use_patch_pooling}")

# Move input to same device as model
img_preprocessed = img_preprocessed.to(dinov3.device)
features = dinov3(img_preprocessed)
print(f"   Input shape: {img_preprocessed.shape}")
print(f"   Features shape: {features.shape}")
assert features.shape == (2, 384), f"DINOv3 feature extraction failed! Got {features.shape}"
print("   ✓ DINOv3 encoder test passed!")

# Test 4: Patch features extraction
print("\n4. Testing patch features extraction...")
patch_features = dinov3.extract_patch_features(img_preprocessed)
print(f"   Patch features shape: {patch_features.shape}")
assert patch_features.shape == (2, 196, 384), f"Patch extraction failed! Got {patch_features.shape}"
print("   ✓ Patch features test passed!")

# Test 5: LAPODINOv3
print("\n5. Testing LAPODINOv3 architecture...")
lapo = LAPODINOv3(
    feature_dim=1152,  # 384 * 3 frames
    latent_act_dim=128,
    act_head_dim=1024,
    obs_head_dim=1024,
).to(dinov3.device)
print(f"   Feature dim: {lapo.feature_dim}")
print(f"   Latent action dim: {lapo.latent_act_dim}")

# Simulate stacked features (3 frames)
obs_features = torch.randn(2, 1152).to(dinov3.device)
future_obs_features = torch.randn(2, 1152).to(dinov3.device)

pred_next_obs, latent_action = lapo(obs_features, future_obs_features)
print(f"   Input features shape: {obs_features.shape}")
print(f"   Predicted next obs shape: {pred_next_obs.shape}")
print(f"   Latent action shape: {latent_action.shape}")

assert pred_next_obs.shape == (2, 1152), "LAPO output shape failed!"
assert latent_action.shape == (2, 128), "LAPO action shape failed!"
print("   ✓ LAPODINOv3 test passed!")

# Test 6: Full pipeline
print("\n6. Testing full pipeline (64x64 → DINOv3 → LAPO)...")
img_64 = torch.randint(0, 256, (2, 64, 64, 9), dtype=torch.uint8)  # 3 stacked frames
print(f"   Input image shape: {img_64.shape} (HuggingFace format, 3 frames stacked)")

# Convert to (B, C, H, W)
img_64_chw = img_64.permute(0, 3, 1, 2)
print(f"   Converted to: {img_64_chw.shape}")

# Split into individual frames
batch_size, channels, height, width = img_64_chw.shape
num_frames = 3
img_split = img_64_chw.reshape(batch_size, num_frames, 3, height, width)
img_split = img_split.reshape(batch_size * num_frames, 3, height, width)
print(f"   Split frames: {img_split.shape}")

# Upscale
img_224 = upscale_image(img_split, target_size=224)
print(f"   Upscaled to: {img_224.shape}")

# Preprocess
img_preprocessed = preprocess_for_dinov2(img_224).to(dinov3.device)
print(f"   Preprocessed: {img_preprocessed.shape}")

# Extract features
features = dinov3(img_preprocessed)
print(f"   DINOv3 features: {features.shape}")

# Reshape back to batch
features = features.reshape(batch_size, num_frames * dinov3.feature_dim)
print(f"   Stacked features: {features.shape}")

# LAPO forward
future_features = features.clone()  # Dummy future features
pred_next_obs, latent_action = lapo(features, future_features)
print(f"   LAPO latent action: {latent_action.shape}")

print("   ✓ Full pipeline test passed!")

print("\n" + "="*60)
print("All tests passed successfully!")
print("="*60)
print("\nYou can now train LAPO with DINOv3:")
print("  python train_lapo_dinov3.py --config_path configs/lapo_dinov3_hf.yaml")
