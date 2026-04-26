"""
Prioritized Experience Replay buffer.
Optional enhancement: store transitions and re-sample high-TD-error experiences.
"""

import numpy as np
from typing import Tuple


class SumTree:
    """Binary sum-tree for O(log n) priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self._write   = 0
        self._n       = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self._n


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for off-policy extensions.

    Usage:
        buf = PrioritizedReplayBuffer(capacity=10_000)
        buf.add(transition, priority=td_error.abs())
        batch, indices, weights = buf.sample(batch_size=64)
        buf.update_priorities(indices, new_td_errors)
    """

    def __init__(
        self,
        capacity:  int   = 10_000,
        alpha:     float = 0.6,
        beta_start:float = 0.4,
        beta_end:  float = 1.0,
        beta_steps:int   = 100_000,
        eps:       float = 1e-6,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.eps        = eps
        self._beta      = beta_start
        self._beta_inc  = (beta_end - beta_start) / beta_steps
        self._tree      = SumTree(capacity)
        self._max_p     = 1.0

    @property
    def beta(self) -> float:
        return self._beta

    def add(self, transition, priority: float = None) -> None:
        p = (abs(priority) + self.eps) ** self.alpha if priority else self._max_p
        self._max_p = max(self._max_p, p)
        self._tree.add(p, transition)

    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        batch, indices, priorities = [], [], []
        total  = self._tree.total
        segment = total / batch_size

        for i in range(batch_size):
            s   = np.random.uniform(i * segment, (i + 1) * segment)
            idx, p, data = self._tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        probs   = np.array(priorities) / total
        weights = (len(self._tree) * probs) ** (-self._beta)
        weights /= weights.max()

        self._beta = min(1.0, self._beta + self._beta_inc)
        return batch, np.array(indices), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + self.eps) ** self.alpha
            self._max_p = max(self._max_p, p)
            self._tree.update(idx, p)

    def __len__(self) -> int:
        return len(self._tree)
