
class DataStore:
    def __init__(self):
        pass
    def set_table(self,
            sample_scheme,
            removal_scheme,
            max_size,
            min_size
            ):
        '''
        Args:

        sample_scheme: description of the actions, observations, etc.
        removal_scheme: the online (optimized) policy.
        max_size: the online critic.
        min_size: optional network to transform the observations before
        '''

class DataSaver:
    def __init__(self, data_store, saver_idx):
        self.saver_idx = saver_idx

    def save_data(self,transition):
        pass

class BatchGenerator:
    def __init__(self, data_store, batch_size):
        self.batch_size = batch_size

    def gen_batch
