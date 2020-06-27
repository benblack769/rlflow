import gym

class MarkovAdderWrapper:
    def __init__(self, env, adder):
        self.env = env
        self.adder = adder
        self.agents = env.agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces

    def reset(self):
        obs = self.env.reset()
        assert len(obs) == len(self.env.agents)
        for agent, o in zip(self.env.agents, obs):
            self.adder.add(agent, o, None, 0.0, None, None)
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        assert len(obs) == len(self.env.agents)
        for i in range(len(self.agents)):
            self.adder.add(self.env.agents[i], obs[i], actions[i], rewards[i], dones[i], infos[i])
        return obs, rewards, dones, infos
