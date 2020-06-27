import gym

class AdderWrapper(gym.Wrapper):
    def __init__(self, env, adder):
        self.env = env
        self.adder = adder
        super().__init__(env)

    def reset(self):
        obs = super().reset()
        self.adder.add(obs, None, 0.0, None, None)
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(action)
        self.adder.add(obs, action, rew, done, info)
        return obs, rew, done, info
