import numpy as np
import torch

class UniformExperienceReplay(object):
    def __init__(
            self,
            capacity,
            observation_shape,
            stack_size=4,
            device = 'cpu'):
        self._size = 0
        self._device = device
        self._capacity = capacity
        self._ptr = 0
        self._stack_size = stack_size
        self._obs_buf = np.zeros((self._capacity, *observation_shape))
        self._next_obs_buf = np.zeros((self._capacity, *observation_shape))
        self._act_buf = np.zeros(self._capacity)
        self._rew_buf = np.zeros(self._capacity)
        self._done_buf = np.zeros(self._capacity)

    def add(self, obs, act, rew, next_obs, done):
        self._obs_buf[self._ptr] = obs
        self._act_buf[self._ptr] = act
        self._rew_buf[self._ptr] = rew
        self._next_obs_buf[self._ptr] = next_obs
        self._done_buf[self._ptr] = done
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def sample(self, batch_size=1):
        assert self._size > 0, "Trying to sample from empty replay buffer"
        obs_samples = []
        act_samples = []
        rew_samples = []
        next_obs_samples = []
        done_samples = []
        capacity = self._capacity
        stack_size = self._stack_size
        for _ in range(batch_size):
            idx = np.random.randint(self._size)
            num_valid_frames = self._count_valid_frames(idx)
            indices = [(idx - i) % self._capacity for i in range(num_valid_frames)]
            frames = self._obs_buf[indices]
            next_frame = self._next_obs_buf[idx]
            obs = self._zero_pad(frames)
            next_obs = np.concatenate(([next_frame], obs[:-1]), axis=0)
            obs_samples.append(obs)
            next_obs_samples.append(next_obs)
            act_samples.append(self._act_buf[idx])
            rew_samples.append(self._rew_buf[idx])
            done_samples.append(self._done_buf[idx])
        return (
            torch.tensor(obs_samples).to(self._device),
            torch.tensor(act_samples).to(self._device),
            torch.tensor(rew_samples).to(self._device),
            torch.tensor(next_obs_samples).to(self._device),
            torch.tensor(done_samples)
        )

    def full(self):
        return self._size == self._capacity

    def _zero_pad(self, frames):
        num_zero_pad = self._stack_size - len(frames)
        zero_padding = np.zeros((num_zero_pad, *self._observation_shape))
        return np.concatenate((frames, zero_padding), axis=0)

    def _count_valid_frames(self, idx):
        if (not self.full()) and (idx < self._stack_size - 1):
            # TODO: This is actually incorrectL
            # This code fails if there is a reset within the first `stack_size` entries
            # However this will basically never happen, for two reasons:
            #   1. It's usually a good idea to fill the replay buffer before sampling
            #   2. Normally the stack size is small enough to prevent such immediate resets
            return max(1, self._stack_size - idx)
        capacity = self._capacity
        stack_size = self._stack_size
        step_types = [self._done_buf[(idx - i) % capacity] for i in range(stack_size)]
        valid_frames = next((i for i in range(1, stack_size) if step_types[-i] == 1), None)
        if valid_frames is None:
            return stack_size
        return valid_frames
