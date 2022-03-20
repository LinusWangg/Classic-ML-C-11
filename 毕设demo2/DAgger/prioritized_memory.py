import random
import numpy as np
from sum_tree import SumTree


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_anneal_step=0.001, epsilon=0.00000001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.a = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_anneal_step
        self.e = epsilon

    def _get_priority(self, error):
        # Direct proportional prioritization
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            data = "ghost"

            while data == "ghost":
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return np.array(batch), idxs, is_weight

    def step(self):
        self.beta = np.min([1. - self.e, self.beta + self.beta_increment_per_sampling])

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
