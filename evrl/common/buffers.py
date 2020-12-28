"""

Buffers to be used by RL algorithms in order to sample
transitions experienced in the past in order to learn
from them.

"""

from typing import List, Tuple, Iterable
import torch
import random
import numpy as np


class ReplayMemory:
    """

    A simple buffer that stores transitions of arbitrary values.
    Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training

    """
    def __init__(self, capacity: int = 1000000) -> None:
        """

        Initialize the buffer and set the pointer to position 0.

        Args:
            capacity: the maximum number of transitions to hold.

        """

        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition: List[torch.Tensor]) -> None:
        """

        Insert a list of tensor of shape (1, ...) into the buffer.

        Args:
            transition: a transition of arbitrary elements to be
            stored in the buffer.

        Returns:
            None.

        """

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def insert_batch(self, transitions: List[torch.Tensor]) -> None:
        """

        Inserts a batch of transitions into the buffer.

        Args:
            transitions: a list of tensors of size (batch_size, ...).

        Returns:
            None.

        """

        sliced_tensors = [x.split(1) for x in transitions]
        for transition in zip(*sliced_tensors):
            self.insert(list(transition))

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        """

        Sample a batch of transitions with equal probability
        to be chosen.

        Args:
            batch_size: number of transitions to be sampled.

        Returns:
            A list of tensors of size (batch_size, ...).

        """

        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size * 10

    def __len__(self) -> int:
        return len(self.memory)


class PrioritizedReplayMemory:
    """

    A transition buffer that orders its entries based on their
    priority and generates samples based on it.

    """

    def __init__(self, capacity: int = 1000000,
                 alpha: float = 0.6, beta: float = 0.4) -> None:
        """

        Initialize the buffer.

        Args:
            capacity: the maximum number of transitions to hold.

            alpha: a parameter in [0, 1] that controls how much
            we favor high priority samples.

            beta: a parameter in [0, 1] that controls the emphasis
            given to importance sampling.
        """
        self.tree = _SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

    def push(self, error: torch.FloatTensor, transition: Iterable[torch.Tensor]) -> None:
        """

        Add a transition to the buffer sorted by a priority value
        calculated from the error provided.

        Args:
            error: a 0-dimensional float tensor representing the learning
            potential of this transition.

            transition: an iterable of tensors representing the environment's
            transition at time t.

        Returns:
            None.

        """
        priority = self._compute_priority(error)
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, Iterable[torch.Tensor]]:
        """

        Retrieve a batch of transitions based on their priority.

        Args:
            batch_size: the number of transitions to sample.

        Returns:
            A tuple with the indices, importance sampling factors and transitions.

        """
        transitions = []
        indices = []
        priorities = []

        # Partition the span of the tree into as many segments
        # as samples we want to draw.
        segment = self.tree.total() / batch_size

        self.beta = np.min([1., self.beta + 0.001])

        for i in range(batch_size):

            # Sample a value from each segment.
            a = segment * i
            b = segment * (i + 1)
            segment_sample = random.uniform(a, b)

            # Retrieve the transition, its index and priority from the tree.
            index, priority, transition = self.tree.get(segment_sample)

            # Add the retrieved values to temporary buffers.
            priorities.append(torch.tensor(priority).view(1, 1))
            transitions.append(transition)
            indices.append(torch.tensor(index).view(1, 1))

        # Convert the batch values into tensors of shape (batch_shape, ...).
        transitions = [torch.cat(items) for items in zip(*transitions)]
        indices = torch.cat(indices)
        priorities = torch.cat(priorities)

        # Compute important sampling from priorities.
        sampling_probabilities = priorities / self.tree.total()
        is_weights = (sampling_probabilities * self.tree.n_entries).pow(-self.beta)
        # is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return indices, is_weights, transitions

    def update(self, index: int, error: torch.FloatTensor) -> None:
        """

        Update the priority of a transition in the buffer based on its
        learning potential (error).

        Args:
            index: the index of the transition whose priority will be updated.

            error: a 0-dimensional float tensor representing the learning
            potential of this transition.

        Returns:
            None.

        """
        priority = self._compute_priority(error)
        self.tree.update(index, priority)

    def can_sample(self, batch_size: int) -> bool:
        return self.tree.n_entries >= batch_size * 10

    def _compute_priority(self, error: torch.FloatTensor) -> float:
        return (torch.abs(error) + 0.01) ** self.alpha


class _SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    """

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        # store priority and sample
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        # update priority
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        # get priority and sample
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]

    def _propagate(self, idx, change):
        # update to the root node
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # find sample on leaf node
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
