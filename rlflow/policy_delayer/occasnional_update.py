
class OccasionalUpdate:
    def __init__(self, steps_to_update):
        self.steps_to_update = steps_to_update
        self.train_steps = 0

    def set_policies(self, learner_policy, actor_policies):
        self.learner_policy = learner_policy
        self.actor_policies = actor_policies
        self.sync()

    def sync(self):
        policy_weights = self.learner_policy.get_weights()
        for act_policy in self.actor_policies:
            policy_weights.set_weights(policy_weights)

    def learn_step(self):
        if self.train_steps % self.steps_to_update == 0:
            self.sync()
