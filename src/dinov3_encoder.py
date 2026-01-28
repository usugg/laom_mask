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
        pooling_mode: str = 'avg',  # 'avg', 'adaptive_4x4', 'flatten', 'cls', 'topk_variance'
        topk: int = 16,  # Number of patches to keep for topk_variance mode
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize DINOv3 encoder.
        
        Args:
            model_name: HuggingFace model name for DINOv3
            freeze: If True, freeze all parameters (no gradient updates)
            pooling_mode: Feature extraction mode:
                - 'avg': Average pooling over all patches -> 384-dim
                - 'adaptive_4x4': 4x4 adaptive pooling -> 6144-dim
                - 'flatten': Flatten all patches -> 75264-dim
                - 'cls': CLS token only -> 384-dim
                - 'topk_variance': Keep K patches with highest variance -> K*384-dim
            topk: Number of patches to keep (only for topk_variance mode)
            device: Device to load model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.pooling_mode = pooling_mode
        self.topk = topk
        
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
        self.num_patches = 196  # For 224x224 with patch_size=16: 14*14=196
        print(f"DINOv3 register tokens: {self.num_register_tokens}")
        print(f"DINOv3 pooling mode: {pooling_mode}")
        
        # Initialize positional embeddings
        self.pos_emb = self._get_positional_embeddings()
        
        # Calculate output dimension based on pooling mode
        if pooling_mode == 'avg' or pooling_mode == 'cls':
            self.output_dim = self.feature_dim  # 384
        elif pooling_mode == 'adaptive_4x4':
            self.output_dim = self.feature_dim * 16  # 384 * 16 = 6144
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        elif pooling_mode == 'flatten':
            self.output_dim = self.feature_dim * self.num_patches  # 384 * 196 = 75264
        elif pooling_mode == 'topk_variance':
            # Feature dim + 2 for coords (x,y)
            self.output_dim = (self.feature_dim + 2) * topk  # (384+2) * K
            print(f"DINOv3 topk: {topk} patches (with pos embeddings)")
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")
        print(f"DINOv3 output dimension: {self.output_dim}")
        
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
    
    def _get_positional_embeddings(self):
        """
        Create 2D positional embeddings (normalized coordinates).
        Returns: (196, 2) tensor with (x, y) coordinates in range [-1, 1].
        """
        # 14x14 grid
        grid_size = 14
        coords = torch.linspace(-1, 1, grid_size)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        
        # Stack to (14, 14, 2) and flatten to (196, 2)
        pos_emb = torch.stack([x, y], dim=-1).flatten(0, 1)
        return pos_emb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images based on pooling_mode.
        
        Args:
            images: Tensor of shape (B, C, H, W)
                Expected to be normalized with ImageNet stats
        
        Returns:
            features: Tensor of shape (B, output_dim)
        """
        if self.is_frozen:
            self.model.eval()
        
        with torch.set_grad_enabled(not self.is_frozen):
            outputs = self.model(pixel_values=images)
            hidden_states = outputs.last_hidden_state
            
            # Extract patch tokens (skip CLS and register tokens)
            patch_start_idx = 1 + self.num_register_tokens
            patch_features = hidden_states[:, patch_start_idx:, :]  # (B, 196, 384)
            
            if self.pooling_mode == 'avg':
                # Average pooling over all patches
                features = patch_features.mean(dim=1)  # (B, 384)
            elif self.pooling_mode == 'cls':
                # CLS token only
                features = hidden_states[:, 0, :]  # (B, 384)
            elif self.pooling_mode == 'adaptive_4x4':
                # Reshape to spatial grid and apply 4x4 adaptive pooling
                B = patch_features.shape[0]
                # (B, 196, 384) -> (B, 14, 14, 384) -> (B, 384, 14, 14)
                spatial = patch_features.reshape(B, 14, 14, self.feature_dim).permute(0, 3, 1, 2)
                pooled = self.adaptive_pool(spatial)  # (B, 384, 4, 4)
                features = pooled.flatten(1)  # (B, 6144)
            elif self.pooling_mode == 'flatten':
                # Flatten all patches
                features = patch_features.flatten(1)  # (B, 75264)
            elif self.pooling_mode == 'topk_variance':
                # Add positional embeddings BEFORE selection
                B = patch_features.shape[0]
                # (196, 2) -> (B, 196, 2)
                pos_emb = self.pos_emb.to(self.device).unsqueeze(0).expand(B, -1, -1)
                
                # Concatenate features + coords: (B, 196, 384+2)
                features_with_pos = torch.cat([patch_features, pos_emb], dim=-1)
                
                # Calculate variance only on features (not coords)
                patch_var = patch_features.var(dim=-1)  # (B, 196)
                
                # Select K patches with highest variance
                topk_indices = patch_var.topk(self.topk, dim=1).indices  # (B, K)
                
                # Gather selected patches (with coords)
                # (B, K, 384+2)
                topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim + 2)
                selected = features_with_pos.gather(1, topk_indices_expanded)
                
                features = selected.flatten(1)  # (B, K*(384+2))
            else:
                raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")
        
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
            f"  output_dim={self.output_dim},\n"
            f"  pooling_mode={self.pooling_mode},\n"
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
        pooling_mode='flatten'  # Test with all patches
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
