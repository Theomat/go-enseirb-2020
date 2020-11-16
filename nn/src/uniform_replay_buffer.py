from typing import List

import numpy as np


class UniformReplayBuffer:

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List = []
        self.generator: np.random.Generator = np.random.default_rng(seed)

    def store(self, episodes: List[List]):
        self._memory += episodes
        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]

    def sample(self, size: int) -> List[List]:
        memories = self.generator.integers(0, len(self._memory), size, dtype=np.int)
        output = []
        for index in memories:
            output.append(self._memory[index])
        return output
