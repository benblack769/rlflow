import numpy as np

class MarkovVectorEnv:
    def __init__(self, par_env, black_death=False):
        '''
        parameters:
            - par_env: the pettingzoo Parallel environment that will be converted to a gym vector environment
            - black_death: whether to give zero valued observations and 0 rewards when an agent is done, allowing for environments with multiple numbers of agents

        The resulting object will be a valid vector environment that has a num_envs
        parameter equal to the max number of agents, will return an array of observations,
        rewards, dones, etc, and will reset environment automatically when it finishes
        '''
        self.par_env = par_env
        self.observation_space = list(par_env.observation_spaces.values())[0]
        self.action_space = list(par_env.action_spaces.values())[0]
        assert all(self.observation_space == obs_space for obs_space in par_env.observation_spaces.values()), "observation spaces not consistent. Perhaps you should wrap with `supersuit.aec_wrappers.pad_observations`?"
        assert all(self.action_space == obs_space for obs_space in par_env.action_spaces.values()), "action spaces not consistent. Perhaps you should wrap with `supersuit.aec_wrappers.pad_actions`?"
        self.num_envs = len(par_env.possible_agents)
        self.black_death = black_death
        self.obs_buffer = np.empty((self.num_envs,)+self.observation_space.shape, dtype=self.observation_space.dtype)

    def seed(self, seed=None):
        self.par_env.seed(seed)

    def concat_obs(self, obs_dict):
        if self.black_death:
            self.obs_buffer[:] = 0
        for i, agent in enumerate(self.par_env.possible_agents):
            self.obs_buffer[i] = obs_dict[agent]
        return self.obs_buffer

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def reset(self):
        return self.concat_obs(self.par_env.reset())

    def step(self, actions):
        agent_set = set(self.par_env.agents)
        act_dict = {agent: actions[i] for i,agent in enumerate(self.par_env.possible_agents) if agent in agent_set}
        observations, rewards, dones, infos = self.par_env.step(act_dict)
        if all(dones):
            observations = self.reset()
        else:
            observations = self.concat_obs(observations)
        assert self.par_env.agents == self.par_env.possible_agents, "MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True"
        rews = np.array([rewards.get(agent,0) for agent in self.par_env.possible_agents], dtype=np.float32)
        dns = np.array([dones.get(agent,False) for agent in self.par_env.possible_agents], dtype=np.uint8)
        return observations, rews, dns, infos
