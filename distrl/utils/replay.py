import numpy as np
import torch

class UniformExperienceReplay(object):
    def __init__(
            self,
            capacity,
            observation_shape,
            device = 'cpu'):
        self._size = 0
        self._device = device
        self._capacity = capacity
        self._ptr = 0
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
        if self._size == 0:
            raise ValueError("Trying to sample from empty replay buffer")
        indices = np.random.randint(self._size, size=batch_size)
        obs = torch.tensor(self._obs_buf[indices]).to(self._device)
        act = torch.tensor(self._act_buf[indices]).to(self._device)
        rew = torch.tensor(self._rew_buf[indices]).to(self._device)
        next_obs = torch.tensor(self._next_obs_buf[indices]).to(self._device)
        done = torch.tensor(self._done_buf[indices]).to(self._device)
        return obs, act, rew, next_obs, done
