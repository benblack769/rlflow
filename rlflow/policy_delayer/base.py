
class BasePolicyDelayer:
    def learn_step(self, learner_policy):
        '''
        the learner should call this every time
        the learner steps. The policy delayer
        chooses whether or not to store this policy
        so that it can update the actor policy when actor_step
        is called
        '''

    def actor_step(self, actor_policy):
        '''
        The actor calls this every time the actor steps.
        The policy delayer chooses whether to update the actor's
        policy or not
        '''
