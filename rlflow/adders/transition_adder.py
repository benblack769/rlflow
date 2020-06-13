
class TransitionAdder:
    def __init__(self, on_generate):
        self.last_observation = None
        self.on_generate = on_generate

    def add(self, obs, rew, done, info):
        if self.last_observation is None:
            self.last_observation = obs
        else:
            if done:
                transition = (None, rew, self.last_observation)
            else:
                transition = (obs, rew, self.last_observation)
            self.on_generate(transition)
            self.last_observation = obs
