
class Learner:
    def train_on(self, transition_batch):
        '''
        train the learner on the batch of transitions.

        See Adder for transition definitions
        '''
        raise NotImplementedError()

    def get_params(self):
        '''
        Needed for asynchronous actors.
        Returns the parameters that the actor network needs
        '''
        raise NotImplementedError()
