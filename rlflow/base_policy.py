
class Policy:
    def set_params(self, params):
        '''
        Sets the policy parameters to the argument.
        Used to transfer data from the learner to the actor
        Needed for asynchronous actors.
        '''
        raise NotImplementedError()

    def get_params(self):
        '''
        Gets the policy parameters to the argument.
        Used to transfer data from the learner to the actor
        Needed for asynchronous actors.
        '''
        raise NotImplementedError()


class StatelessPolicy(Policy):
    def calc_action(self, observations):
        '''
        calculate action based off observations
        '''
        raise NotImplementedError()


class RecurrentPolicy(Policy):
    def calc_action(self, observations, states):
        '''
        Args:
        observations: a batch of single-step observations from the environment
        states: batch of states from previous step
        infos: environment info
        Returns:
        actions: the actions to pass to the environment
        states: the new states
        '''
        raise NotImplementedError()

    def new_state(self):
        '''
        creates new state when an environment resets.
        Most likely just returns zeros
        '''
        raise NotImplementedError()
