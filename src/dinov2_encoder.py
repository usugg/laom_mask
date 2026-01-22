"""DINOv2 Encoder Wrapper for LAOM"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv2Encoder(nn.Module):
    """
    Wrapper for DINOv2 model from HuggingFace transformers.
    Extracts features from images for use in LAOM.
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/dinov2-base',
        freeze: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize DINOv2 encoder.
        
        Args:
            model_name: HuggingFace model name
                - 'facebook/dinov2-small': 384-dim features
                - 'facebook/dinov2-base': 768-dim features (default)
                - 'facebook/dinov2-large': 1024-dim features
                - 'facebook/dinov2-giant': 1536-dim features
            freeze: If True, freeze all parameters (no gradient updates)
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        
        # Load model and processor
        print(f"Loading DINOv2 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Get feature dimension from model config
        self.feature_dim = self.model.config.hidden_size
        print(f"DINOv2 feature dimension: {self.feature_dim}")
        
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
        print("DINOv2 parameters frozen")
    
    def unfreeze(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        self.is_frozen = False
        print("DINOv2 parameters unfrozen")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: Tensor of shape (B, C, H, W)
                Expected to be normalized with ImageNet stats
                (use preprocess_for_dinov2 from utils.py)
        
        Returns:
            features: Tensor of shape (B, feature_dim)
                CLS token features from DINOv2
        """
        # Set to eval mode if frozen
        if self.is_frozen:
            self.model.eval()
        
        # Forward pass
        with torch.set_grad_enabled(not self.is_frozen):
            outputs = self.model(pixel_values=images)
            
            # Extract CLS token (first token)
            # outputs.last_hidden_state has shape (B, num_patches+1, feature_dim)
            # CLS token is at index 0
            features = outputs.last_hidden_state[:, 0, :]  # (B, feature_dim)
        
        return features
    
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract all patch features (not just CLS token).
        
        Args:
            images: Tensor of shape (B, C, H, W)
        
        Returns:
            features: Tensor of shape (B, num_patches, feature_dim)
                All patch features (excluding CLS token)
        """
        if self.is_frozen:
            self.model.eval()
        
        with torch.set_grad_enabled(not self.is_frozen):
            outputs = self.model(pixel_values=images)
            
            # Extract all patch features (excluding CLS token at index 0)
            features = outputs.last_hidden_state[:, 1:, :]  # (B, num_patches, feature_dim)
        
        return features
    
    def __repr__(self):
        return (
            f"DINOv2Encoder(\n"
            f"  model_name={self.model_name},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  frozen={self.is_frozen},\n"
            f"  device={self.device}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the encoder
    print("Testing DINOv2Encoder...")
    
    encoder = DINOv2Encoder(model_name='facebook/dinov2-base', freeze=True)
    print(encoder)
    
    # Test with random image
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224).to(encoder.device)
    
    print(f"\nInput shape: {img.shape}")
    
    # Extract CLS features
    features = encoder(img)
    print(f"CLS features shape: {features.shape}")
    print(f"Expected: ({batch_size}, {encoder.feature_dim})")
    
    # Extract patch features
    patch_features = encoder.extract_patch_features(img)
    print(f"Patch features shape: {patch_features.shape}")
    
    print("\n✓ DINOv2Encoder test passed!")
