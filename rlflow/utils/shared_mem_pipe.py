import multiprocessing as mp
from .shared_array import SharedArray
import numpy as np

class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

def expand_example(example, batch_size):
    return [Struct(shape=(batch_size,)+item.shape,dtype=item.dtype) for item in example]

class SharedMemPipe:
    def __init__(self, data_example):
        self.shared_data = []
        self.copied_data = []
        for arr in data_example:
            assert np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool), "dtype of transition must be a number or bool, something wrong in adder or environment"
            data_entry = SharedArray(arr.shape,dtype=arr.dtype)
            self.shared_data.append(data_entry)
            self.copied_data.append(np.empty(arr.shape,dtype=arr.dtype))

        self.is_full = mp.Event()
        self.is_empty = mp.Event()
        self.is_empty.set()

    def can_store(self):
        return self.is_empty.is_set()

    def store(self, data_store):
        self.is_empty.wait()
        for source, dest in zip(data_store, self.shared_data):
            np.copyto(dest.np_arr, source)
        self.is_full.set()
        self.is_empty.clear()

    def get_wait(self):
        self.is_full.wait()

        for dest,src in zip(self.copied_data, self.shared_data):
            np.copyto(dest, src.np_arr)

        self.is_full.clear()
        self.is_empty.set()
        return self.copied_data

    def get(self):
        if not self.is_full.is_set():
            return None

        return self.get_wait()
