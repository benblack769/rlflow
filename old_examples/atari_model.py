
import sys
import gym
import random
import numpy as np

from pettingzoo.atari import boxing_v0, combat_tank_v0, joust_v1, surround_v0, space_invaders_v0, warlords_v1, tennis_v1
from pettingzoo.atari import entombed_competitive_v1, ice_hockey_v0
from supersuit import clip_reward_v0, sticky_actions_v0, resize_v0
from supersuit import frame_skip_v0, frame_stack_v1, agent_indicator_v0, flatten_v0
from torch import nn
import torch

#from cyclic_reward_wrapper import cyclic_reward_wrapper

# tf1, tf, tfv = try_import_tf()

def ortho(tens):
    torch.nn.init.normal_(tens.weight,std=0.01)
    return tens

class AtariModel(nn.Module):
    def __init__(self, name="atari_model"):
        super().__init__()
        # super(AtariModel, self).__init__(obs_space, action_space, num_outputs, model_config,
        #                  name)
        # inputs = torch.tensor(obs_space)
        # inputs2 = tf.keras.layers.Input(shape=(2,), name='agent_indicator')
        # Convolutions on the frames on the screen
        self.model = nn.Sequential(
            nn.Conv2d(
                4,
                32,
                [8, 8],
                stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                [4, 4],
                stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                [3, 3],
                stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136,512)),
            nn.ReLU(),
        )

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        obs_vec = self.model(obs)
        # logits = self.policy_layer(obs_vec)
        # value = self.value_layer(obs_vec)
        return obs_vec
