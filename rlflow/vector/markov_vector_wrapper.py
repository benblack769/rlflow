import numpy as np

class MarkovVectorEnv:
    def __init__(self, markov_env):
        self.markov_env = markov_env
        self.observation_space = markov_env.observation_spaces[0]
        self.action_space = markov_env.action_spaces[0]
        assert all(self.observation_space == obs_space for obs_space in markov_env.observation_spaces), "observation spaces not consistent. Perhaps you should wrap with `supersuit.aec_wrappers.pad_observations`?"
        assert all(self.action_space == obs_space for obs_space in markov_env.action_spaces), "action spaces not consistent. Perhaps you should wrap with `supersuit.aec_wrappers.pad_actions`?"
        self.num_envs = len(markov_env.agents)
        self.obs_buffer = np.empty((self.num_envs,)+self.observation_space.shape, dtype=self.observation_space.dtype)

    def concat_obs(self, obs_list):
        for i in range(self.num_envs):
            self.obs_buffer[i] = obs_list[i]
        return self.obs_buffer

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def reset(self):
        return self.concat_obs(self.markov_env.reset())

    def step(self, actions):
        observations, rewards, dones, infos = self.markov_env.step(actions)
        observations = self.concat_obs(observations)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        if all(dones):
            observations = self.reset()
        return observations, rewards, dones, infos
