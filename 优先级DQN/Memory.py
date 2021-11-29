import numpy as np
from SumTree import SumTree

class Memory(object):
    epsilon = 0.00000001  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        ps = np.empty((n, 1))
        pri_seg = self.tree.total_p() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()    # for later calculate ISweight
        sum = 0
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i+1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            sum += (p + self.epsilon) ** self.alpha
            ps[i] = p
            #prob = p / self.tree.total_p()
            #ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        ps /= sum
        for i in range(n):
            ISWeights[i, 0] = pow(self.tree.n_entry*ps[i], -self.beta)
        wmax = np.max(ISWeights)
        ISWeights /= wmax
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_error):
        abs_error += self.epsilon
        clipped_errors = np.minimum(abs_error.detach().numpy(), self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)