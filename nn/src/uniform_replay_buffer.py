from typing import Tuple, List

import numpy as np
import torch


class UniformReplayBuffer:

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List = []
        self.generator: np.random.Generator = np.random.default_rng(seed)
        self._buffer = None

    def store(self, episodes: List[List]):
        for episode in episodes:
            for (s, pi, z) in episode:
                if isinstance(s, np.ndarray):
                    s = torch.from_numpy(s)
                if isinstance(pi, np.ndarray):
                    pi = torch.from_numpy(pi)
                if isinstance(z, np.ndarray):
                    z = torch.from_numpy(z)
                self._memory.append((s, pi, z))
        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]

    def sample(self, size: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self._buffer is None or self._buffer[0].shape[0] != size:
            self._buffer = [torch.zeros((size, 15, 9, 9)), torch.zeros((size, 82)),  torch.zeros((size, 1))]
        memories = self.generator.integers(0, len(self._memory), size, dtype=np.int)
        for i, index in enumerate(memories):
            sample = self._memory[index]
            self._buffer[0][i] = sample[0]
            self._buffer[1][i] = sample[1]
            self._buffer[2][i] = sample[2]
        return self._buffer[0], self._buffer[1], self._buffer[2]
