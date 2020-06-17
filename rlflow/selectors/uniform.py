import numpy as np
import warnings
class UniformSampleScheme:
    def __init__(self, max_size, seed=None):
        self.sample_idxs = np.zeros(max_size, dtype=np.int32)
        self.data_idxs = np.zeros(max_size, dtype=np.int32)
        self.num_idxs = 0
        self.max_size = max_size
        self.np_random = np.random.RandomState(seed)
    def add(self, id):
        if self.num_idxs >= self.max_size:
            warnings.warn("added element makes buffer greater than max size, make sure to remove element first")
        self.data_idxs[self.num_idxs] = id
        self.sample_idxs[id] = self.num_idxs
        self.num_idxs += 1
    def sample(self, batch_size):
        if len(self.data_idxs) < batch_size:
            return None
        else:
            idxs = self.np_random.randint(0,self.num_idxs,size=batch_size)
            return self.data_idxs[idxs]
    def remove(self, ids):
        for id in ids:
            idx = self.sample_idxs[id]
            if idx != self.num_idxs-1:
                new_id = self.data_idxs[self.num_idxs-1]
                self.data_idxs[idx] = new_id
                self.sample_idxs[new_id] = idx
            self.num_idxs -= 1
