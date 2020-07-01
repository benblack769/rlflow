import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np
import gym
import math
import warnings
from torch.nn import functional as F


class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, obs_space, obs_shape, num_features, device):
        super().__init__()
        final_dim = num_features
        lim_dim = max(obs_shape[0],obs_shape[1])
        num_layers = int(math.log2(lim_dim) - math.log2(4))
        all_layer_sizes = [24,32,48,64,96]
        layer_sizes = [obs_shape[-1]]+all_layer_sizes[len(all_layer_sizes)-num_layers:]
        layers = []
        cur_shape = obs_shape
        for i in range(num_layers):
            layers.append(torch.nn.Conv2d(layer_sizes[i], layer_sizes[i+1], 3, padding=1))
            #layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Conv2d(layer_sizes[i+1], layer_sizes[i+1], 3, padding=1))
            #layers.append(torch.nn.ReLU())
            layers.append(torch.nn.MaxPool2d(kernel_size=2,stride=2))
            cur_shape = ((cur_shape[0])//2,(cur_shape[1])//2,layer_sizes[i+1])
        layers.append(torch.nn.Flatten())
        res_size = int(np.prod(cur_shape))

        layers.append(torch.nn.Linear(res_size, final_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(final_dim, final_dim))
        net = torch.nn.Sequential(*layers)

        self.net = net.to(device)
        self.obs_shape = obs_shape
        self.obs_space = obs_space
        self.obs_target_shape = obs_shape

    def forward(self, input):
        input = input.float()
        input = torch.transpose(input, 1, 3)
        features = self.net(input)
        return features


class MLPFeatureExtractor(torch.nn.Module):
    def __init__(self, obs_space, obs_shape, num_features, device):
        super().__init__()
        in_dim = obs_shape[0]
        final_dim = num_features
        net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, final_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(final_dim, final_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(final_dim, final_dim)
        )
        self.net = net.to(device)
        self.obs_space = obs_space
        self.obs_target_shape = obs_shape

    def forward(self, input):
        input = input.float()
        features = self.net(input)
        return features

def AdaptiveFeatureExtractor(obs_space, num_features, device):
    obs_shape = obs_space.shape
    if len(obs_shape) == 2:
        obs_shape = obs_shape+(1,)

    if len(obs_shape) == 3:
        if obs_shape[0] < 6 or obs_shape[1] < 6:
            warnings.warn("observation space is screwed up. Flateening observation")
            obs_shape = (np.prod(obs_shape),)

    if len(obs_shape) == 3:
        return CNNFeatureExtractor(obs_space, obs_shape, num_features, device)
    elif len(obs_shape) == 1:
        return MLPFeatureExtractor(obs_space, obs_shape, num_features, device)
    else:
        raise RuntimeError("Bad observation shape")
