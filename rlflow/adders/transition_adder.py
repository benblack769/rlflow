import numpy as np
from rlflow.utils.space_wrapper import SpaceWrapper

class TransitionAdder:
    def __init__(self, observation_space, action_space):
        self.last_observation = None
        self.on_generate = None
        self.observation_space = SpaceWrapper(observation_space)
        self.action_space = SpaceWrapper(action_space)

    def get_example_output(self):
        return (
            self.observation_space,
            self.action_space,
            np.array(0,dtype=np.float32),
            np.array(0,dtype=np.uint8),
            self.observation_space
        )

    def set_generate_callback(self, on_generate):
        assert self.on_generate is None, "set_generate_callback should only be called once"
        self.on_generate = on_generate

    def add(self, obs, action, rew, done, info):
        assert self.on_generate is not None, "need to call set_generate_callback before add"
        obs = np.copy(obs)
        if self.last_observation is None:
            self.last_observation = obs
        else:
            transition = (obs, action, rew, done, self.last_observation)
            self.on_generate(transition)
            self.last_observation = None if done else obs
