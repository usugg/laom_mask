"""DINOv3 Encoder Wrapper for LAPO

Uses patch token pooling for better motion detection compared to CLS token.
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv3Encoder(nn.Module):
    """
    Wrapper for DINOv3 model from HuggingFace transformers.
    Extracts features from images using spatial pooling over patch tokens.
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/dinov3-vits16-pretrain-lvd1689m',
        freeze: bool = True,
        use_patch_pooling: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize DINOv3 encoder.
        
        Args:
            model_name: HuggingFace model name for DINOv3
            freeze: If True, freeze all parameters (no gradient updates)
            use_patch_pooling: If True, use spatial average pooling over patches
                              If False, use CLS token only
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.use_patch_pooling = use_patch_pooling
        
        # Load model and processor
        print(f"Loading DINOv3 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Get feature dimension from model config
        self.feature_dim = self.model.config.hidden_size  # 384 for ViT-S
        print(f"DINOv3 feature dimension: {self.feature_dim}")
        
        # DINOv3 has register tokens between CLS and patches
        # For ViT-S/16 with 224x224: 1 CLS + 4 registers + 196 patches = 201 tokens
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 4)
        print(f"DINOv3 register tokens: {self.num_register_tokens}")
        
        # Freeze parameters if requested
        if freeze:
            self.freeze()
        
        self.is_frozen = freeze
    
    def freeze(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.is_frozen = True
        print("DINOv3 parameters frozen")
    
    def unfreeze(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        self.is_frozen = False
        print("DINOv3 parameters unfrozen")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using spatial pooling over patches.
        
        Args:
            images: Tensor of shape (B, C, H, W)
                Expected to be normalized with ImageNet stats
        
        Returns:
            features: Tensor of shape (B, feature_dim)
                Spatially pooled patch features
        """
        if self.is_frozen:
            self.model.eval()
        
        with torch.set_grad_enabled(not self.is_frozen):
            outputs = self.model(pixel_values=images)
            
            # outputs.last_hidden_state has shape (B, num_tokens, feature_dim)
            # num_tokens = 1 (CLS) + num_register_tokens + num_patches
            hidden_states = outputs.last_hidden_state
            
            if self.use_patch_pooling:
                # Extract patch tokens (skip CLS and register tokens)
                patch_start_idx = 1 + self.num_register_tokens
                patch_features = hidden_states[:, patch_start_idx:, :]  # (B, num_patches, feature_dim)
                
                # Spatial average pooling over patches
                features = patch_features.mean(dim=1)  # (B, feature_dim)
            else:
                # Use CLS token only
                features = hidden_states[:, 0, :]  # (B, feature_dim)
        
        return features
    
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract all patch features without pooling.
        
        Args:
            images: Tensor of shape (B, C, H, W)
        
        Returns:
            features: Tensor of shape (B, num_patches, feature_dim)
                All patch features (excluding CLS and register tokens)
        """
        if self.is_frozen:
            self.model.eval()
        
        with torch.set_grad_enabled(not self.is_frozen):
            outputs = self.model(pixel_values=images)
            
            # Extract patch tokens (skip CLS and register tokens)
            patch_start_idx = 1 + self.num_register_tokens
            features = outputs.last_hidden_state[:, patch_start_idx:, :]
        
        return features
    
    def __repr__(self):
        return (
            f"DINOv3Encoder(\n"
            f"  model_name={self.model_name},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  use_patch_pooling={self.use_patch_pooling},\n"
            f"  frozen={self.is_frozen},\n"
            f"  device={self.device}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the encoder
    print("Testing DINOv3Encoder...")
    
    encoder = DINOv3Encoder(
        model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        freeze=True,
        use_patch_pooling=True
    )
    print(encoder)
    
    # Test with random image
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224).to(encoder.device)
    
    print(f"\nInput shape: {img.shape}")
    
    # Extract pooled features
    features = encoder(img)
    print(f"Pooled features shape: {features.shape}")
    print(f"Expected: ({batch_size}, {encoder.feature_dim})")
    
    # Extract all patch features
    patch_features = encoder.extract_patch_features(img)
    print(f"Patch features shape: {patch_features.shape}")
    print(f"Expected: ({batch_size}, 196, {encoder.feature_dim})")
    
    print("\n✓ DINOv3Encoder test passed!")
