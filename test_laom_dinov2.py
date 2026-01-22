#!/usr/bin/env python
"""Test script for LAOM with DINOv2 integration"""
import torch
from src.dinov2_encoder import DINOv2Encoder
from src.nn import LAOMDINOv2
from src.utils import upscale_image, preprocess_for_dinov2

print("="*60)
print("Testing LAOM with DINOv2 Integration")
print("="*60)

# Test 1: Image upscaling
print("\n1. Testing image upscaling...")
img_64 = torch.randn(2, 3, 64, 64)
img_224 = upscale_image(img_64, target_size=224)
print(f"   Input shape: {img_64.shape}")
print(f"   Upscaled shape: {img_224.shape}")
assert img_224.shape == (2, 3, 224, 224), "Upscaling failed!"
print("   ✓ Upscaling test passed!")

# Test 2: DINOv2 preprocessing
print("\n2. Testing DINOv2 preprocessing...")
img_uint8 = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
img_preprocessed = preprocess_for_dinov2(img_uint8)
print(f"   Input dtype: {img_uint8.dtype}, range: [0, 255]")
print(f"   Preprocessed dtype: {img_preprocessed.dtype}")
print(f"   Preprocessed range: [{img_preprocessed.min():.2f}, {img_preprocessed.max():.2f}]")
assert img_preprocessed.dtype == torch.float32, "Preprocessing dtype failed!"
print("   ✓ Preprocessing test passed!")

# Test 3: DINOv2 encoder
print("\n3. Testing DINOv2 encoder...")
print("   Loading DINOv2-base model...")
dinov2 = DINOv2Encoder(model_name='facebook/dinov2-base', freeze=True)
print(f"   Model loaded: {dinov2.model_name}")
print(f"   Feature dim: {dinov2.feature_dim}")
print(f"   Frozen: {dinov2.is_frozen}")

# Move input to same device as model
img_preprocessed = img_preprocessed.to(dinov2.device)
features = dinov2(img_preprocessed)
print(f"   Input shape: {img_preprocessed.shape}")
print(f"   Features shape: {features.shape}")
assert features.shape == (2, 768), "DINOv2 feature extraction failed!"
print("   ✓ DINOv2 encoder test passed!")

# Test 4: LAOMDINOv2
print("\n4. Testing LAOMDINOv2 architecture...")
laom = LAOMDINOv2(
    feature_dim=768,
    latent_act_dim=128,
    act_head_dim=1024,
    obs_head_dim=1024,
).to(dinov2.device)
print(f"   Feature dim: {laom.feature_dim}")
print(f"   Latent action dim: {laom.latent_act_dim}")

obs_features = torch.randn(2, 768).to(dinov2.device)
next_obs_features = torch.randn(2, 768).to(dinov2.device)

latent_next_obs, latent_action, obs_hidden = laom(obs_features, next_obs_features)
print(f"   Input features shape: {obs_features.shape}")
print(f"   Latent next obs shape: {latent_next_obs.shape}")
print(f"   Latent action shape: {latent_action.shape}")
print(f"   Obs hidden shape: {obs_hidden.shape}")

assert latent_next_obs.shape == (2, 768), "LAOM output shape failed!"
assert latent_action.shape == (2, 128), "LAOM action shape failed!"
print("   ✓ LAOMDINOv2 test passed!")

# Test 5: Full pipeline
print("\n5. Testing full pipeline (64x64 → DINOv2 → LAOM)...")
img_64 = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)  # HuggingFace format
print(f"   Input image shape: {img_64.shape} (HuggingFace format)")

# Convert to (B, C, H, W)
img_64_chw = img_64.permute(0, 3, 1, 2)
print(f"   Converted to: {img_64_chw.shape}")

# Upscale
img_224 = upscale_image(img_64_chw, target_size=224)
print(f"   Upscaled to: {img_224.shape}")

# Preprocess
img_preprocessed = preprocess_for_dinov2(img_224).to(dinov2.device)
print(f"   Preprocessed: {img_preprocessed.shape}")

# Extract features
features = dinov2(img_preprocessed)
print(f"   DINOv2 features: {features.shape}")

# LAOM forward
next_features = dinov2(img_preprocessed)  # Dummy next features
latent_next_obs, latent_action, _ = laom(features, next_features)
print(f"   LAOM latent action: {latent_action.shape}")

print("   ✓ Full pipeline test passed!")

print("\n" + "="*60)
print("All tests passed successfully!")
print("="*60)
print("\nYou can now train LAOM with DINOv2:")
print("  python train_laom_dinov2.py --config_path configs/laom_dinov2_hf.yaml")
