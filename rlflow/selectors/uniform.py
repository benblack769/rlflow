import numpy as np
import warnings
from .base import BaseScheme

class UniformSampleScheme(BaseScheme):
    def __init__(self, max_size, seed=None):
        self.sample_idxs = np.zeros(max_size, dtype=np.int32)
        self.data_idxs = np.zeros(max_size, dtype=np.int32)
        self.num_idxs = 0
        self.max_size = max_size
        self.np_random = np.random.RandomState(seed)
    def add(self, id):
        assert self.num_idxs < self.max_size, "added element makes buffer greater than max size, make sure to remove element first"
        self.data_idxs[self.num_idxs] = id
        self.sample_idxs[id] = self.num_idxs
        self.num_idxs += 1
    def sample(self, batch_size):
        if self.num_idxs < batch_size:
            return None, None
        else:
            idxs = self.np_random.randint(0,self.num_idxs,size=batch_size)
            return self.data_idxs[idxs], np.ones(batch_size)
    def remove(self, id):
        idx = self.sample_idxs[id]
        new_idx = self.num_idxs-1
        if idx != new_idx:
            new_id = self.data_idxs[new_idx]
            self.data_idxs[idx] = new_id
            self.sample_idxs[new_id] = idx
        self.num_idxs = new_idx

def test():
    scheme = UniformSampleScheme(4)
    print(scheme.add(1))
    print(scheme.add(2))
    print(scheme.add(3))
    print(scheme.add(0))
    print(scheme.remove(2))
    print(scheme.add(1))
    print([int(scheme.sample(1)[0]) for i in range(10)])
if __name__ == "__main__":
    test()
