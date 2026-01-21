#!/usr/bin/env python
"""Test script for DCSLAOMHFDataset"""
from src.utils import DCSLAOMHFDataset
import torch

print('Testing DCSLAOMHFDataset...')
dataset = DCSLAOMHFDataset(
    dataset_name='EpicPinkPenguin/visual_distracting_control_suite',
    config_name='cheetah_run',
    split='train',
    frame_stack=3,
    max_offset=10,
    streaming=True,
    buffer_size=1000,
    use_masked_obs=False,
    device='cpu',
)

print(f'Dataset metadata:')
print(f'  Image size: {dataset.img_hw}x{dataset.img_hw}')
print(f'  Action dim: {dataset.act_dim}')
print(f'  State dim: {dataset.state_dim}')
print(f'  Frame stack: {dataset.frame_stack}')
print(f'  Max offset: {dataset.max_offset}')

# Get one sample
print('\nGetting first sample...')
for sample in dataset:
    obs, next_obs, future_obs, action, state, offset = sample
    print(f'Sample shapes:')
    print(f'  Observation: {obs.shape}')
    print(f'  Next observation: {next_obs.shape}')
    print(f'  Future observation: {future_obs.shape}')
    print(f'  Action: {action.shape}')
    print(f'  State: {state.shape}')
    print(f'  Offset: {offset}')
    print('\n✓ DCSLAOMHFDataset test passed!')
    break

print('\n' + '='*60)
print('Testing masked observation mode...')
print('='*60)

dataset_masked = DCSLAOMHFDataset(
    dataset_name='EpicPinkPenguin/visual_distracting_control_suite',
    config_name='cheetah_run',
    split='train',
    frame_stack=3,
    max_offset=10,
    streaming=True,
    buffer_size=1000,
    use_masked_obs=True,
    device='cpu',
)

for sample in dataset_masked:
    obs, next_obs, future_obs, action, state, offset = sample
    print(f'Masked observation shape: {obs.shape}')
    print(f'Observation dtype: {obs.dtype}')
    print('\n✓ Masked observation mode test passed!')
    break

print('\n' + '='*60)
print('All tests passed successfully!')
print('='*60)
