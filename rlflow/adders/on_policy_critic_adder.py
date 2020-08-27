import numpy as np
import gym
from rlflow.utils.space_wrapper import SpaceWrapper

class OnPolicyAdder:
    def __init__(self, num_steps, observation_space, action_space):
        self.on_generate = None
        self.observation_space = SpaceWrapper(observation_space)
        self.action_space = SpaceWrapper(action_space)
        self.data_list = []
        self.num_steps = num_steps

    def get_example_output(self):
        return (
            self.observation_space,
            self.action_space,
            np.array(0,dtype=np.float32), # values
            np.array(0,dtype=np.float32), # advantages
            np.array(0,dtype=np.float32), # returns
            self.action_space,   # log_probs
        )

    def set_generate_callback(self, on_generate):
        assert self.on_generate is None, "set_generate_callback should only be called once"
        self.on_generate = on_generate

    def add(self, obs, action, rew, done, info):
        assert self.on_generate is not None, "need to call set_generate_callback before add"
        if isinstance(self.action_space, gym.spaces.Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data_list.append(np.array(obs), action, rew, done, info)

        if len(self.data_list) > self.num_steps:

            self.on_generate(transition)
            self.data_list = []
