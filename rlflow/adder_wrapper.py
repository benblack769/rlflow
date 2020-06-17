import gym

class AdderWrapper(gym.Wrapper):
    def __init__(self, env, adder):
        self.env = env
        self.adder = adder

    def step(self):
        pass
