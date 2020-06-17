
class NoUpdatePolicyDelayer:
    def set_policies(self, learner_policy, actor_policies):
        self.learner_policy = learner_policy
        self.actor_policies = actor_policies

    def learn_step(self):
        pass
