import numpy as np

class MarkovEnv(object):
    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

class aec_to_markov(MarkovEnv):
    def __init__(self, AECenv):
        self.AECenv = AECenv
        self.agents = AECenv.agents[:]
        self.observation_spaces = [AECenv.observation_spaces[agent] for agent in self.agents]
        self.action_spaces = [AECenv.action_spaces[agent] for agent in self.agents]

    def agent_index(self, agent):
        return self.agents.index(agent)

    def reset(self):
        self.AECenv.reset(observe=False)
        observations = [self.AECenv.observe(agent) for agent in self.agents]
        self.prev_dones = [False]*len(self.agents)
        return observations

    def render(self, mode='human'):
        self.AECenv.render(mode=mode)

    def close(self):
        self.AECenv.close()

    def step(self, actions):
        for i, (agent_inorder, was_done) in enumerate(zip(self.agents, self.prev_dones)):
            if not was_done:
                env_agent = self.AECenv.agent_selection
                assert agent_inorder == env_agent, "Markov Game is wrapping an environment which has an unusual agent order, this is not allowed"
                self.AECenv.step(actions[i], observe=False)

        observations = [self.AECenv.observe(agent) for agent in self.agents]

        dones = [self.AECenv.dones[agent] for agent in self.agents]
        rewards = [self.AECenv.rewards[agent] for agent in self.agents]
        infos = [self.AECenv.infos[agent] for agent in self.agents]
        self.prev_dones = dones
        return observations, rewards, dones, infos
