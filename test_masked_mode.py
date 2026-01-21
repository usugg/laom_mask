"""
Test script to verify masked observation mode in HuggingFace dataset
"""
import torch
from src.utils import DCSLAPOHFDataset

def test_masked_mode():
    print("Testing Masked Observation Mode...")
    print("=" * 60)
    
    # Test configuration with masked mode enabled
    config = {
        "dataset_name": "EpicPinkPenguin/visual_distracting_control_suite",
        "config_name": "cheetah_run",
        "split": "train",
        "frame_stack": 3,
        "max_offset": 1,
        "streaming": True,
        "buffer_size": 100,  # Small buffer for testing
        "use_masked_obs": True,  # Enable masked mode
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataset with masked mode
    print("Creating dataset with masked mode enabled...")
    dataset = DCSLAPOHFDataset(**config)
    
    print(f"Dataset metadata:")
    print(f"  Image size: {dataset.img_hw}x{dataset.img_hw}")
    print(f"  Action dimension: {dataset.act_dim}")
    print(f"  Frame stack: {dataset.frame_stack}")
    print(f"  Max offset: {dataset.max_offset}")
    print(f"  Masked mode: {dataset.use_masked_obs}")
    print()
    
    # Test iteration
    print("Testing masked data loading (first 3 batches)...")
    iterator = iter(dataset)
    
    for i in range(3):
        try:
            obs, next_obs, future_obs, action, offset = next(iterator)
            
            print(f"\nBatch {i+1}:")
            print(f"  obs shape: {obs.shape}")
            print(f"  next_obs shape: {next_obs.shape}")
            print(f"  future_obs shape: {future_obs.shape}")
            print(f"  action shape: {action.shape}")
            print(f"  offset: {offset}")
            print(f"  obs dtype: {obs.dtype}")
            print(f"  obs range: [{obs.min():.1f}, {obs.max():.1f}]")
            
            # Check if masking is working (should have many zeros from masked background)
            zero_pixels = (obs == 0).sum().item()
            total_pixels = obs.numel()
            zero_percentage = (zero_pixels / total_pixels) * 100
            print(f"  Zero pixels (masked background): {zero_percentage:.1f}%")
            
            # Verify shapes
            expected_obs_shape = (dataset.img_hw, dataset.img_hw, 3 * dataset.frame_stack)
            assert obs.shape == expected_obs_shape, f"Expected {expected_obs_shape}, got {obs.shape}"
            assert next_obs.shape == expected_obs_shape, f"Expected {expected_obs_shape}, got {next_obs.shape}"
            assert future_obs.shape == expected_obs_shape, f"Expected {expected_obs_shape}, got {future_obs.shape}"
            assert action.shape == (dataset.act_dim,), f"Expected ({dataset.act_dim},), got {action.shape}"
            
        except Exception as e:
            print(f"\nError in batch {i+1}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("Masked observation mode is working correctly.")
    
    # Now test without masked mode for comparison
    print("\n" + "=" * 60)
    print("Testing WITHOUT masked mode for comparison...")
    config["use_masked_obs"] = False
    dataset_no_mask = DCSLAPOHFDataset(**config)
    iterator_no_mask = iter(dataset_no_mask)
    
    obs_no_mask, _, _, _, _ = next(iterator_no_mask)
    zero_pixels_no_mask = (obs_no_mask == 0).sum().item()
    total_pixels = obs_no_mask.numel()
    zero_percentage_no_mask = (zero_pixels_no_mask / total_pixels) * 100
    
    print(f"\nComparison:")
    print(f"  Zero pixels WITHOUT mask: {zero_percentage_no_mask:.1f}%")
    print(f"  Zero pixels WITH mask: {zero_percentage:.1f}%")
    print(f"  Difference: {zero_percentage - zero_percentage_no_mask:.1f}%")
    
    if zero_percentage > zero_percentage_no_mask:
        print("\n✓ Masking is working! More zeros in masked observations.")
    else:
        print("\n⚠ Warning: Masking may not be working as expected.")
    
if __name__ == "__main__":
    test_masked_mode()
