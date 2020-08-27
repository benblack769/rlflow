import multiprocessing as mp
import queue
from rlflow.utils.shared_mem_pipe import SharedMemPipe, expand_example
import numpy as np

class DataManager:
    def __init__(self, new_entries_pipes, transition_example, removal_scheme, sample_scheme, max_entries):
        self.removal_scheme = removal_scheme
        self.sample_scheme = sample_scheme
        self.max_entries = max_entries
        self.transition_example = transition_example
        self.new_entries_pipes = new_entries_pipes
        self.init_add_idx = 0

        self.data = []
        for arr in transition_example:
            assert np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool), "dtype of transition must be a number or bool, something wrong in adder or environment"
            data_entry = np.empty((self.max_entries,)+arr.shape,dtype=arr.dtype)
            self.data.append(data_entry)

    def receive_new_entries(self):
        for new_entry_pipes in self.new_entries_pipes:
            add_data = new_entry_pipes.get()

            if add_data is not None:
                self.add_data(add_data)

    def add_data(self, add_data):
        if self.init_add_idx < self.max_entries:
            new_id = self.init_add_idx
            self.init_add_idx += 1
        else:
            remove_vals = self.removal_scheme.sample(1)
            assert remove_vals is not None, "tried to remove item and could not, something is wrong with removal scheme or replay buffer size is too small"
            new_id = remove_vals[0]
            self.removal_scheme.remove(new_id)
            self.sample_scheme.remove(new_id)

        self.sample_scheme.add(new_id)
        self.removal_scheme.add(new_id)
        self._add_item(new_id, add_data)

    def sample_data(self, batch_size):
        sample_idxs, sample_weights = self.sample_scheme.sample(batch_size)
        if sample_idxs is None:
            return None, None, None
        else:
            return sample_idxs, sample_weights, self._get_data(sample_idxs)

    def _add_item(self, id, transition):
        for data,trans in zip(self.data,transition):
            data[id] = trans

    def _get_data(self, idxs):
        idxs = np.asarray(idxs,dtype=np.int64)
        result = []
        for source in self.data:
            result.append(source[idxs])
        return result
