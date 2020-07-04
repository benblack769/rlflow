import numpy as np
import gym
import warnings

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


def alt_observation_space(obs_space):
    if isinstance(obs_space, gym.spaces.Box):
        return gym.spaces.Box(np.minimum(obs_space.low,0.), np.maximum(obs_space.high,1.), dtype=obs_space.dtype)
    else:
        raise RuntimeError("only box observation spaces supported by aec_to_markov wrapper")


def pass_observation(obs_space):
    return np.zeros_like(obs_space.low)

def done_observation(obs_space):
    return np.ones_like(obs_space.low)

class aec_to_markov(MarkovEnv):
    def __init__(self, AECenv, turn_based_game=False):
        self.AECenv = AECenv
        self.turn_based_game = turn_based_game
        self.agents = AECenv.agents[:]
        self.observation_spaces = [alt_observation_space(AECenv.observation_spaces[agent]) for agent in self.agents]
        self.action_spaces = [AECenv.action_spaces[agent] for agent in self.agents]

        self.pass_obs = [pass_observation(obs_space) for obs_space in self.observation_spaces]
        self.done_obs = [done_observation(obs_space) for obs_space in self.observation_spaces]

    def agent_index(self, agent):
        return self.agents.index(agent)

    def reset(self):
        self.AECenv.reset(observe=False)
        observations = [self.AECenv.observe(agent) for agent in self.agents]
        return observations

    def render(self, mode='human'):
        self.AECenv.render(mode=mode)

    def close(self):
        self.AECenv.close()

    def step(self, actions):
        observations = [None]*len(self.agents)
        if self.turn_based_game:
            current_agent = self.AECenv.agent_selection
            next_obs = self.AECenv.step(actions[self.agent.index(current_agent)])
            for i, agent in enumerate(self.agents):
                if self.AECenv.dones[agent]:
                    observations[i] = self.done_obs[i]
                elif agent != self.AECenv.agent_selection:
                    observations[i] = self.pass_obs[i]
                else:
                    observations[i] = next_obs
        else:
            for i, agent_inorder in enumerate(self.agents):
                should_pass = agent_inorder != self.AECenv.agent_selection
                if not self.AECenv.dones[agent_inorder] and not should_pass:
                    self.AECenv.step(actions[i], observe=False)

                if self.AECenv.dones[agent_inorder]:
                    observations[i] = self.done_obs[i]
                elif should_pass:
                    observations[i] = self.pass_obs[i]
                    warnings.warn("Markov Game is wrapping an environment which has an unusual agent order, this is really weird allowed")
                else:
                    observations[i] = self.AECenv.observe(agent_inorder)

        dones = [self.AECenv.dones[agent] for agent in self.agents]
        rewards = [self.AECenv.rewards[agent] for agent in self.agents]
        infos = [self.AECenv.infos[agent] for agent in self.agents]
        return observations, rewards, dones, infos
