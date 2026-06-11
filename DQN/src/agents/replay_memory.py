from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

# Transition and replay memory implementation for DQN training.
@dataclass(frozen=True)
class Transition:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool


class _SumTree:
    """Binary tree that stores cumulative priorities for O(log N) sampling/updates."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)

    def total(self) -> float:
        return self.tree[1]
    # Update the priority of a data index and propagate the change up the tree.
    def update(self, data_idx: int, priority: float) -> None:
        if data_idx < 0 or data_idx >= self.capacity:
            return

        tree_idx = data_idx + self.capacity
        change = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)

        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] += change
            tree_idx //= 2
    # Get the data index and priority for a given prefix-sum value in [0, total).
    def get(self, value: float) -> tuple[int, float]:
        """Return (data_idx, priority) for a prefix-sum query value in [0, total]."""
        idx = 1
        total = self.total()
        if total <= 0.0:
            return 0, 0.0

        value = max(0.0, min(float(value), total - 1e-12))

        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity
        return data_idx, self.tree[idx]


# Simple replay memory with fixed capacity and random sampling.
class ReplayMemory:
    def __init__(
        self,
        capacity: int,
        *,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 250_000,
        priority_epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = max(0.0, float(alpha))
        self.beta_start = min(max(0.0, float(beta_start)), 1.0)
        self.beta_frames = max(1, int(beta_frames))
        self.priority_epsilon = max(1e-8, float(priority_epsilon))

        self._buffer: List[Transition] = []
        self._position = 0
        self._sample_calls = 0
        self._max_priority = 1.0
        self._sum_tree = _SumTree(capacity) if prioritized else None
    # Add a new transition to the replay memory, overwriting the oldest if at capacity.
    def push(self, transition: Transition) -> None:
        insert_idx = self._position
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
            insert_idx = len(self._buffer) - 1
        else:
            self._buffer[insert_idx] = transition

        if self.prioritized and self._sum_tree is not None:
            scaled = (self._max_priority + self.priority_epsilon) ** self.alpha
            self._sum_tree.update(insert_idx, scaled)

        self._position = (self._position + 1) % self.capacity
    # Compute the current beta value for importance-sampling weights based on the number of sample calls.
    def _current_beta(self) -> float:
        fraction = min(1.0, self._sample_calls / float(self.beta_frames))
        return self.beta_start + fraction * (1.0 - self.beta_start)
    # Sample a batch of transitions, returning the transitions, their indices, and importance-sampling weights.
    def sample(self, batch_size: int) -> tuple[List[Transition], List[int], List[float]]:
        if batch_size > len(self._buffer):
            raise ValueError("Not enough samples in replay memory")

        if not self.prioritized:
            indices = random.sample(range(len(self._buffer)), batch_size)
            transitions = [self._buffer[idx] for idx in indices]
            return transitions, indices, [1.0] * batch_size

        self._sample_calls += 1
        assert self._sum_tree is not None
        total_priority = self._sum_tree.total()
        if total_priority <= 0.0:
            indices = random.sample(range(len(self._buffer)), batch_size)
            transitions = [self._buffer[idx] for idx in indices]
            return transitions, indices, [1.0] * batch_size

        indices: List[int] = []
        priorities: List[float] = []
        segment = total_priority / float(batch_size)

        for batch_idx in range(batch_size):
            low = segment * batch_idx
            high = segment * (batch_idx + 1)
            sample_value = random.uniform(low, high)
            data_idx, priority = self._sum_tree.get(sample_value)
            indices.append(data_idx)
            priorities.append(max(priority, 1e-12))

        sample_probs = [priority / total_priority for priority in priorities]
        beta = self._current_beta()
        weights = [(len(self._buffer) * p) ** (-beta) for p in sample_probs]
        max_weight = max(weights) if weights else 1.0
        if max_weight <= 0.0:
            max_weight = 1.0
        weights = [w / max_weight for w in weights]

        transitions = [self._buffer[idx] for idx in indices]
        return transitions, indices, weights
    # Update the priorities of the given indices based on their TD errors.
    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        if not self.prioritized:
            return

        assert self._sum_tree is not None
        for idx, td_error in zip(indices, td_errors):
            if idx < 0 or idx >= len(self._buffer):
                continue
            priority = abs(float(td_error)) + self.priority_epsilon
            if priority > self._max_priority:
                self._max_priority = priority
            scaled = priority ** self.alpha
            self._sum_tree.update(int(idx), scaled)

    def __len__(self) -> int:
        return len(self._buffer)
