
class StatelessActor:
    def __init__(self, policy_mapping):
        self.policy_mapping = policy_mapping

    def step(self, agent, observations, dones, infos):
        # dones can be ignored here, taken care of in Adder
        return self.policy_mapping[agent].calc_action(observations, infos)


class RecurrentActor:
    def __init__(self, policy_mapping, batch_size):
        self.policy_mapping = policy_mapping
        self.states = np.stack([policy.new_state() for _ in range(batch_size)])
        self.batch_size = batch_size

    def step(self, observations, dones, infos):
        assert len(observations) == len(dones) == self.batch_size
        for i in range(self.batch_size):
            if dones[i]:
                self.states[i] = self.policy.new_state()
        actions, self.states = self.policy.calc_action(observations, self.states, infos)
        return actions
