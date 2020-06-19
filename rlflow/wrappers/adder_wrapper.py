import gym

class AdderWrapper(gym.Wrapper):
    def __init__(self, env, adder):
        self.env = env
        self.adder = adder

    def step(self, action):
        obs, rew, done, info = super().step(action)
        self.adder.add(obs, action, rew, done, info)
        return obs, rew, done, info
