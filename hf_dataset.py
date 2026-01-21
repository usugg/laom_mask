"""
Test script to verify HuggingFace dataset integration
"""
import torch
from src.utils import DCSLAPOHFDataset

def hf_dataset():
    print("Testing HuggingFace Dataset Integration...")
    print("=" * 60)
    
    # Test configuration
    config = {
        "dataset_name": "EpicPinkPenguin/visual_distracting_control_suite",
        "config_name": "cheetah_run_distractor_hard",
        "split": "train",
        "frame_stack": 3,
        "max_offset": 1,
        "streaming": True,
        "buffer_size": 100,  # Small buffer for testing
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataset
    print("Creating dataset...")
    dataset = DCSLAPOHFDataset(**config)
    
    print(f"Dataset metadata:")
    print(f"  Image size: {dataset.img_hw}x{dataset.img_hw}")
    print(f"  Action dimension: {dataset.act_dim}")
    print(f"  Frame stack: {dataset.frame_stack}")
    print(f"  Max offset: {dataset.max_offset}")
    print()
    
    # Test iteration
    print("Testing data loading (first 5 batches)...")
    iterator = iter(dataset)
    
    for i in range(5):
        try:
            obs, next_obs, future_obs, action, offset = next(iterator)
            
            print(f"\nBatch {i+1}:")
            print(f"  obs shape: {obs.shape}")
            print(f"  next_obs shape: {next_obs.shape}")
            print(f"  future_obs shape: {future_obs.shape}")
            print(f"  action shape: {action.shape}")
            print(f"  offset: {offset}")
            print(f"  obs dtype: {obs.dtype}")
            print(f"  action dtype: {action.dtype}")
            print(f"  obs range: [{obs.min():.1f}, {obs.max():.1f}]")
            
            # Verify shapes
            expected_obs_shape = (dataset.img_hw, dataset.img_hw, 3 * dataset.frame_stack)
            assert obs.shape == expected_obs_shape, f"Expected {expected_obs_shape}, got {obs.shape}"
            assert next_obs.shape == expected_obs_shape, f"Expected {expected_obs_shape}, got {next_obs.shape}"
            assert future_obs.shape == expected_obs_shape, f"Expected {expected_obs_shape}, got {future_obs.shape}"
            assert action.shape == (dataset.act_dim,), f"Expected ({dataset.act_dim},), got {action.shape}"
            
        except Exception as e:
            print(f"\nError in batch {i+1}: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("HuggingFace dataset integration is working correctly.")
    
if __name__ == "__main__":
    df_dataset()
