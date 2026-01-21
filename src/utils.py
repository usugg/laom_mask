import os
import random

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from shimmy import DmControlCompatibilityV0
from torch.utils.data import Dataset, IterableDataset

from .dcs import suite


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def get_optim_groups(model, weight_decay):
    return [
        # do not decay biases and single-column parameters (rmsnorm), those are usually scales
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {"params": (p for p in model.parameters() if p.dim() >= 2), "weight_decay": weight_decay},
    ]


def get_grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm


def soft_update(target, source, tau=1e-3):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class DCSInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu"):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return len(self.actions) * (self.traj_len - 1)

    def __getitem__(self, idx):
        traj_idx, transition_idx = divmod(idx, self.traj_len - 1)

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        action = self.actions[traj_idx][transition_idx]

        return obs, next_obs, action


class DCSLAPOInMemoryDataset(Dataset):
    """LAPO dataset loading from HDF5 files"""
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]
        assert 1 <= max_offset < self.traj_len
        self.max_offset = max_offset

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return len(self.actions) * (self.traj_len - self.max_offset)

    def __getitem__(self, idx):
        traj_idx, transition_idx = divmod(idx, self.traj_len - self.max_offset)
        action = self.actions[traj_idx][transition_idx]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        offset = random.randint(1, self.max_offset)
        future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

        return obs, next_obs, future_obs, action, (offset - 1)


class DCSLAPOHFDataset(IterableDataset):
    """LAPO dataset loading from HuggingFace Hub with streaming support"""
    def __init__(
        self,
        dataset_name="EpicPinkPenguin/visual_distracting_control_suite",
        config_name="cheetah_run_distractor_hard",
        split="train",
        frame_stack=3,
        max_offset=1,
        streaming=True,
        buffer_size=10000,
        use_masked_obs=False,
        device="cpu",
    ):
        self.dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=streaming,
        )
        
        self.frame_stack = frame_stack
        self.max_offset = max_offset
        self.buffer_size = buffer_size
        self.use_masked_obs = use_masked_obs
        self.device = device
        
        # Get metadata from first sample
        first_sample = next(iter(self.dataset))
        # Convert PIL image to numpy array to get shape
        first_obs = np.array(first_sample["observation"])
        self.img_hw = first_obs.shape[0]  # Assuming square images
        self.act_dim = len(first_sample["action"])
        
        # Buffer for frame stacking
        self.obs_buffer = []
        self.action_buffer = []
        
    def _process_observation(self, obs_array):
        """Convert observation to tensor and move to device"""
        # Handle PIL images from HuggingFace
        if hasattr(obs_array, 'mode'):  # PIL Image
            obs_array = np.array(obs_array)
        return torch.tensor(obs_array, dtype=torch.uint8, device=self.device)
    
    def _get_stacked_obs(self, buffer, idx):
        """Stack frames from buffer"""
        start_idx = max(0, idx - self.frame_stack + 1)
        frames = buffer[start_idx:idx + 1]
        
        # Pad if at the beginning
        if len(frames) < self.frame_stack:
            pad_frame = frames[0]
            frames = [pad_frame] * (self.frame_stack - len(frames)) + frames
        
        # Stack and reshape: (frame_stack, H, W, C) -> (H, W, frame_stack*C)
        stacked = torch.stack(frames)  # (frame_stack, H, W, C)
        stacked = stacked.permute((1, 2, 0, 3))  # (H, W, frame_stack, C)
        stacked = stacked.reshape(*stacked.shape[:2], -1)  # (H, W, frame_stack*C)
        
        return stacked
    
    def __iter__(self):
        """Iterate over the dataset with frame stacking and future observation sampling"""
        self.obs_buffer = []
        self.action_buffer = []
        
        for sample in self.dataset:
            obs = self._process_observation(sample["observation"])
            action = torch.tensor(sample["action"], dtype=torch.float32, device=self.device)
            
            # Apply mask if enabled
            if self.use_masked_obs:
                # Load mask from HuggingFace sample
                mask = sample["mask"]
                # Convert mask to tensor (handle PIL Image or numpy array)
                if hasattr(mask, 'mode'):  # PIL Image
                    mask = np.array(mask)
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
                
                # Expand mask from (H, W) to (H, W, 3) for RGB channels
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1).repeat(1, 1, 3)
                
                # Normalize mask to 0-1 range if needed
                if mask.max() > 1.0:
                    mask = mask / 255.0
                
                # Apply mask to observation (element-wise multiplication)
                obs = obs.float() * mask
                obs = obs.to(torch.uint8)
            
            self.obs_buffer.append(obs)
            self.action_buffer.append(action)
            
            # Keep buffer size manageable
            if len(self.obs_buffer) > self.buffer_size:
                self.obs_buffer.pop(0)
                self.action_buffer.pop(0)
            
            # Need at least max_offset + 1 frames to create a sample
            if len(self.obs_buffer) >= self.max_offset + 1:
                # Current observation index (relative to buffer)
                current_idx = len(self.obs_buffer) - self.max_offset - 1
                
                # Get stacked observations
                obs_stacked = self._get_stacked_obs(self.obs_buffer, current_idx)
                next_obs_stacked = self._get_stacked_obs(self.obs_buffer, current_idx + 1)
                
                # Random offset for future observation
                offset = random.randint(1, self.max_offset)
                future_obs_stacked = self._get_stacked_obs(self.obs_buffer, current_idx + offset)
                
                # Get corresponding action
                action_current = self.action_buffer[current_idx]
                
                yield obs_stacked, next_obs_stacked, future_obs_stacked, action_current, (offset - 1)


class DCSLAOMHFDataset(IterableDataset):
    """LAOM dataset loading from HuggingFace Hub with streaming support"""
    def __init__(
        self,
        dataset_name="EpicPinkPenguin/visual_distracting_control_suite",
        config_name="cheetah_run_distractor_hard",
        split="train",
        frame_stack=3,
        max_offset=1,
        streaming=True,
        buffer_size=10000,
        use_masked_obs=False,
        device="cpu",
    ):
        self.dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=streaming,
        )
        
        self.frame_stack = frame_stack
        self.max_offset = max_offset
        self.buffer_size = buffer_size
        self.use_masked_obs = use_masked_obs
        self.device = device
        
        # Get metadata from first sample
        first_sample = next(iter(self.dataset))
        # Convert PIL image to numpy array to get shape
        first_obs = np.array(first_sample["observation"])
        self.img_hw = first_obs.shape[0]  # Assuming square images
        self.act_dim = len(first_sample["action"])
        self.state_dim = len(first_sample["state"])
        
        # Buffer for frame stacking
        self.obs_buffer = []
        self.action_buffer = []
        self.state_buffer = []
        
    def _process_observation(self, obs_array):
        """Convert observation to tensor and move to device"""
        # Handle PIL images from HuggingFace
        if hasattr(obs_array, 'mode'):  # PIL Image
            obs_array = np.array(obs_array)
        return torch.tensor(obs_array, dtype=torch.uint8, device=self.device)
    
    def _get_stacked_obs(self, buffer, idx):
        """Stack frames from buffer"""
        start_idx = max(0, idx - self.frame_stack + 1)
        frames = buffer[start_idx:idx + 1]
        
        # Pad if at the beginning
        if len(frames) < self.frame_stack:
            pad_frame = frames[0]
            frames = [pad_frame] * (self.frame_stack - len(frames)) + frames
        
        # Stack and reshape: (frame_stack, H, W, C) -> (H, W, frame_stack*C)
        stacked = torch.stack(frames)  # (frame_stack, H, W, C)
        stacked = stacked.permute((1, 2, 0, 3))  # (H, W, frame_stack, C)
        stacked = stacked.reshape(*stacked.shape[:2], -1)  # (H, W, frame_stack*C)
        
        return stacked
    
    def __iter__(self):
        """Iterate over the dataset with frame stacking and future observation sampling"""
        self.obs_buffer = []
        self.action_buffer = []
        self.state_buffer = []
        
        for sample in self.dataset:
            obs = self._process_observation(sample["observation"])
            action = torch.tensor(sample["action"], dtype=torch.float32, device=self.device)
            state = torch.tensor(sample["state"], dtype=torch.float32, device=self.device)
            
            # Apply mask if enabled
            if self.use_masked_obs:
                # Load mask from HuggingFace sample
                mask = sample["mask"]
                # Convert mask to tensor (handle PIL Image or numpy array)
                if hasattr(mask, 'mode'):  # PIL Image
                    mask = np.array(mask)
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
                
                # Expand mask from (H, W) to (H, W, 3) for RGB channels
                if mask.ndim == 2:
                    mask = mask.unsqueeze(-1).repeat(1, 1, 3)
                
                # Normalize mask to 0-1 range if needed
                if mask.max() > 1.0:
                    mask = mask / 255.0
                
                # Apply mask to observation (element-wise multiplication)
                obs = obs.float() * mask
                obs = obs.to(torch.uint8)
            
            self.obs_buffer.append(obs)
            self.action_buffer.append(action)
            self.state_buffer.append(state)
            
            # Keep buffer size manageable
            if len(self.obs_buffer) > self.buffer_size:
                self.obs_buffer.pop(0)
                self.action_buffer.pop(0)
                self.state_buffer.pop(0)
            
            # Need at least max_offset + 1 frames to create a sample
            if len(self.obs_buffer) >= self.max_offset + 1:
                # Current observation index (relative to buffer)
                current_idx = len(self.obs_buffer) - self.max_offset - 1
                
                # Get stacked observations
                obs_stacked = self._get_stacked_obs(self.obs_buffer, current_idx)
                next_obs_stacked = self._get_stacked_obs(self.obs_buffer, current_idx + 1)
                
                # Random offset for future observation
                offset = random.randint(1, self.max_offset)
                future_obs_stacked = self._get_stacked_obs(self.obs_buffer, current_idx + offset)
                
                # Get corresponding action and state
                action_current = self.action_buffer[current_idx]
                state_current = self.state_buffer[current_idx]
                
                yield obs_stacked, next_obs_stacked, future_obs_stacked, action_current, state_current, (offset - 1)


class DCSLAOMInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]
        assert 1 <= max_offset < self.traj_len
        self.max_offset = max_offset

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return len(self.actions) * (self.traj_len - self.max_offset)

    def __getitem__(self, idx):
        traj_idx, transition_idx = divmod(idx, self.traj_len - self.max_offset)
        action = self.actions[traj_idx][transition_idx]
        state = self.states[traj_idx][transition_idx]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        offset = random.randint(1, self.max_offset)
        future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

        return obs, next_obs, future_obs, action, state, (offset - 1)


class DCSLAOMTrueActionsDataset(IterableDataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]
        assert 1 <= max_offset < self.traj_len
        self.max_offset = max_offset

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs

    def __iter__(self):
        while True:
            traj_idx = random.randint(0, len(self.actions) - 1)
            transition_idx = random.randint(0, self.actions[traj_idx].shape[0] - self.max_offset)

            obs = self.__get_padded_obs(traj_idx, transition_idx)
            next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
            offset = random.randint(1, self.max_offset)
            future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

            action = self.actions[traj_idx][transition_idx]
            state = self.states[traj_idx][transition_idx]

            yield obs, next_obs, future_obs, action, state, (offset - 1)


def normalize_img(img):
    return ((img / 255.0) - 0.5) * 2.0


def unnormalize_img(img):
    return ((img / 2.0) + 0.5) * 255.0


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class SelectPixelsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class FlattenStackedFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = old_shape[1:-1] + (old_shape[0] * old_shape[-1],)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        obs = obs.transpose((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs


def create_env_from_df(
    hdf5_path,
    backgrounds_path,
    backgrounds_split,
    frame_stack=1,
    pixels_only=True,
    flatten_frames=True,
    difficulty=None,
):
    with h5py.File(hdf5_path, "r") as df:
        dm_env = suite.load(
            domain_name=df.attrs["domain_name"],
            task_name=df.attrs["task_name"],
            difficulty=df.attrs["difficulty"] if difficulty is None else difficulty,
            dynamic=df.attrs["dynamic"],
            background_dataset_path=backgrounds_path,
            background_dataset_videos=backgrounds_split,
            pixels_only=pixels_only,
            render_kwargs=dict(height=df.attrs["img_hw"], width=df.attrs["img_hw"]),
        )
        env = DmControlCompatibilityV0(dm_env)
        env = gym.wrappers.ClipAction(env)

        if pixels_only:
            env = SelectPixelsObsWrapper(env)

        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            if flatten_frames:
                env = FlattenStackedFrames(env)

    return env
