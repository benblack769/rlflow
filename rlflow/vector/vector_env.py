import copy
import multiprocessing as mp
from gym.vector.utils import shared_memory
from pettingzoo.utils.agent_selector import agent_selector
import numpy as np
import ctypes
import gym

class VectorAECWrapper:
    def __init__(self, env_constructors):
        assert len(env_constructors) >= 1
        assert callable(env_constructors[0]), "env_constructor must be a callable object (i.e function) that create an environment"

        self.envs = [env_constructor() for env_constructor in env_constructors]
        self.num_envs = len(env_constructors)
        self.env = self.envs[0]
        self.num_agents = self.env.num_agents
        self.agents = self.env.agents
        self.observation_spaces = copy.copy(self.env.observation_spaces)
        self.action_spaces = copy.copy(self.env.action_spaces)
        self._agent_selector = agent_selector(self.agents)

    def _find_active_agent(self):
        cur_selection = self.agent_selection
        while not any(cur_selection == env.agent_selection for env in self.envs):
            cur_selection = self._agent_selector.next()
        return cur_selection

    def _collect_dicts(self):
        self.rewards = {agent: np.array([env.rewards[agent] for env in self.envs],dtype=np.float32) for agent in self.agents}
        self.dones = {agent: np.array([env.dones[agent] for env in self.envs],dtype=np.uint8) for agent in self.agents}
        env_dones = np.array([all(env.dones.values()) for env in self.envs],dtype=np.uint8)
        self.infos = {agent: [env.infos[agent] for env in self.envs] for agent in self.agents}
        return env_dones

    def reset(self, observe=True):
        '''
        returns: list of observations
        '''
        observations = []
        for env in self.envs:
            observations.append(env.reset(observe))

        self.agent_selection = self._agent_selector.reset()
        self.agent_selection = self._find_active_agent()

        env_dones = self._collect_dicts()
        passes = np.array([env.agent_selection != self.agent_selection for env in self.envs],dtype=np.uint8)

        return (np.stack(observations) if observe else None),passes

    def observe(self, agent):
        observations = []
        for env in self.envs:
            observations.append(env.observe(agent))
        return np.stack(observations)

    def last(self):
        last_agent = self.agent_selection
        return self.rewards[last_agent], self.dones[last_agent], self.infos[last_agent]

    def step(self, actions, observe=True):
        assert len(actions) == len(self.envs)
        old_agent = self.agent_selection

        observations = []
        for act,env in zip(actions,self.envs):
            observations.append(env.step(act,observe) if env.agent_selection == old_agent else None)

        self.agent_selection = self._agent_selector.next()
        self.agent_selection = self._find_active_agent()
        new_agent = self.agent_selection

        env_dones = self._collect_dicts()

        # self.rewards = {agent: [env.rewards[agent] for env in self.envs] for agent in self.agents}
        # self.dones = {agent: [env.dones[agent] for env in self.envs] for agent in self.agents}
        # self.infos = {agent: [env.infos[agent] for env in self.envs] for agent in self.agents}
        # self._agent_selections = [env.agent_selection for env in self.envs]
        # self.agent_selection = self._agent_selections[0]
        #env_dones = [all(env.dones.values()) for env in self.envs]
        for i,(env,done) in enumerate(zip(self.envs,env_dones)):
            if done:
                observations[i] = env.reset(observe)
            if observations[i] is None or self.agent_selection != env.agent_selection:
                observations[i] = env.observe(self.agent_selection)

        passes = np.array([env.agent_selection != self.agent_selection for env in self.envs],dtype=np.uint8)

        return (np.stack(observations) if observe else None),passes,env_dones
