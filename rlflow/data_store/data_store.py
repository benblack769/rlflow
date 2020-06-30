import multiprocessing as mp
import queue
from rlflow.utils.shared_array import SharedArray
import numpy as np

class DataStore:
    def __init__(self, transition_example, max_size):
        '''
        Args:

        sample_scheme: description of the actions, observations, etc.
        removal_scheme: the online (optimized) policy.
        max_size: the online critic.
        '''
        self.max_size = max_size
        self.add_idx = 0

        self.data = []
        for arr in transition_example:
            assert np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool), "dtype of transition must be a number or bool, something wrong in adder or environment"
            data_entry = SharedArray((self.max_size,)+arr.shape,dtype=arr.dtype)
            self.data.append(data_entry)


    def add_item(self, id, transition):
        for data,trans in zip(self.data,transition):
            data.np_arr[id] = trans


class DataManager:
    def __init__(self, removal_scheme, sample_scheme, max_entries, empty_entries, new_entries, batch_samples, batch_size):
        self.removal_scheme = removal_scheme
        self.sample_scheme = sample_scheme
        self.empty_entries = empty_entries
        self.batch_samples = batch_samples
        self.new_entries = new_entries
        self.batch_size = batch_size
        self.max_entries = max_entries
        self.init_add_idx = 0

    def update(self):
        sample_result = self.sample_scheme.sample(self.batch_size)
        if sample_result is not None:
            self.batch_samples.put(sample_result,timeout=5.0)

        while not self.empty_entries.full():
            if self.init_add_idx < self.max_entries:
                remove_id = self.init_add_idx
                self.init_add_idx += 1
            else:
                remove_vals = self.removal_scheme.sample(1)
                if remove_vals is None:
                    break
                remove_id = remove_vals[0]
                self.removal_scheme.remove(remove_id)
                self.sample_scheme.remove(remove_id)
            # should never except because of the size calculation
            self.empty_entries.put_nowait(remove_id)

        try:
            while True:
                add_id = self.new_entries.get_nowait()
                self.sample_scheme.add(add_id)
                self.removal_scheme.add(add_id)
        except queue.Empty:
            pass



class DataSaver:
    def __init__(self, data_store, empty_entries, new_entries):
        self.data_store = data_store
        self.empty_entries = empty_entries
        self.new_entries = new_entries

    def save_data(self, transition):
        if self.new_entries.full():
            return
        try:
            id = self.empty_entries.get_nowait()
        except queue.Empty:
            return

        self.data_store.add_item(id, transition)

        try:
            self.new_entries.put_nowait(id)
        except queue.Full:
            return

class BatchStore:
    def __init__(self, batch_size, transition_example):
        self.batch_size = batch_size
        self.shared_data = []
        self.copied_data = []
        for arr in transition_example:
            assert np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool), "dtype of transition must be a number or bool, something wrong in adder or environment"
            data_entry = SharedArray((batch_size,)+arr.shape,dtype=arr.dtype)
            self.shared_data.append(data_entry)
            self.copied_data.append(np.empty((batch_size,)+arr.shape,dtype=arr.dtype))

        self.is_full = mp.Event()
        self.is_empty = mp.Event()
        self.is_empty.set()

    def store_batch(self, data_store, batch_idxs):
        self.is_empty.wait()
        for source, dest in zip(data_store.data, self.shared_data):
            np.take(source.np_arr,batch_idxs,axis=0,out=dest.np_arr)

        self.is_full.set()
        self.is_empty.clear()

    def get_batch(self):
        if not self.is_full.is_set():
            return None

        for dest,src in zip(self.copied_data, self.shared_data):
            dest[:] = src.np_arr

        self.is_full.clear()
        self.is_empty.set()
        return self.copied_data
