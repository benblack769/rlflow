import numpy as np

def priority_pipe_example(batch_size):
    return (
        np.empty(batch_size,dtype=np.int64),
        np.empty(batch_size,dtype=np.float32)
    )

class PriorityUpdater:
    def __init__(self, alpha):
        self.alpha = alpha
        self.data_pipe = None

    def set_data_pipe(self, data_pipe):
        assert self.data_pipe is None, "cannot set data pipe twice"
        self.data_pipe = data_pipe

    def update_td_error(self, idxs, new_td_error):
        assert self.data_pipe is not None, "need to set data pipe before using priority updater"
        density = new_td_error**self.alpha
        self.data_pipe.store((idxs, new_td_error))

    def fetch_densities(self):
        assert self.data_pipe is not None, "need to set data pipe before using priority updater"
        return self.data_pipe.get()

class NoUpdater:
    def __init__(self):
        pass

    def set_data_pipe(self, data_pipe):
        pass

    def update_td_error(self, idxs, new_td_error):
        pass

    def fetch_densities(self):
        pass
