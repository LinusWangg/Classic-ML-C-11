import numpy as np

class SumTree(object):

    def __init__(self, capacity):
        self.pointer = 0
        self.capacity = capacity
        self.n_entry = 0
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.pointer + self.capacity - 1
        self.data[self.pointer] = data
        self.update(tree_idx, p)

        self.pointer += 1
        self.pointer %= self.capacity
        self.n_entry = min(self.capacity, self.n_entry+1)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx-1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left = 2*parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left]:
                    parent_idx = left
                else:
                    v -= self.tree[left]
                    parent_idx = right
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]