
class StatelessPolicy:
    def step(self, observations, infos):
        '''
        calculate action based off observations
        '''
        pass


class RecurrentPolicy:
    def step(self, observations, states, infos):
        '''
        Args:
        observations: a batch of single-step observations from the environment
        states: batch of states from previous step
        infos: environment info
        Returns:
        actions: the actions to pass to the environment
        states: the new states
        '''
    def new_state(self):
        '''
        creates new state when an environment resets.
        Most likely just returns zeros
        '''
