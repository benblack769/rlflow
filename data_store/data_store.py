
class DataStore:
    def __init__(self):
        pass
    def create_table(self,
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
