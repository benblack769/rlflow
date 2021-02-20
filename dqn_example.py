import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import smooth_l1_loss, mse_loss
from all import nn
from all.approximation import QNetwork, FixedTarget
from all.agents import Agent, DQN, DQNTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all import nn
from all.core import StateArray


def fc_relu_q(env, hidden=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.observation_space.shape[0], hidden),
        nn.ReLU(),
        nn.Linear(hidden, env.action_space.n),
    )

default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 1e-3,
    # Training settings
    "minibatch_size": 64,
    "update_frequency": 1,
    "target_update_frequency": 100,
    # Replay buffer settings
    "replay_start_size": 1000,
    "replay_buffer_size": 10000,
    # Explicit exploration
    "initial_exploration": 1.,
    "final_exploration": 0.,
    "final_exploration_step": 10000,
    "test_exploration": 0.001,
    # Model construction
    "model_constructor": fc_relu_q
}


class DQNPolicy:
    def __init__(self, env, device="cpu"):
        self.out_dim = out_dim = env.action_space.n
        self.device = device
        self.hyperparameters = hyperparameters = default_hyperparameters
        self.model = hyperparameters['model_constructor'](env).to(device)
        writer = DummyWriter()
        self.epsilon = LinearScheduler(
            self.hyperparameters['initial_exploration'],
            self.hyperparameters['final_exploration'],
            self.hyperparameters['replay_start_size'],
            self.hyperparameters['final_exploration_step'] - self.hyperparameters['replay_start_size'],
            name="exploration",
            writer=writer
        )

    def calc_action(self, observations):
        with torch.no_grad():
            observations = torch.tensor(observations, device=self.device)
            greedy = torch.argmax(self.model(observations),axis=1)#.detach().cpu().numpy()
            rand_vals = torch.randint(self.out_dim,size=(len(observations),),device=self.device)
            epsilon = self.epsilon._get_value()
            pick_rand = torch.rand((len(observations),),device=self.device) < epsilon
            actions = torch.where(pick_rand, rand_vals, greedy)
        return actions.cpu().detach().numpy()

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.model.parameters()):
            dest.data = torch.tensor(source, device=self.device)

class DQNLearner:
    def __init__(self, policy, logger, out_dim, device="cpu"):
        self.hyperparameters = hyperparameters = default_hyperparameters
        self.policy = policy
        self.model = policy.model
        self.device = device
        self.logger = logger
        self.discount_factor = hyperparameters['discount_factor']
        self.out_dim = out_dim
        writer = DummyWriter()
        optimizer = Adam(self.model.parameters(), lr=self.hyperparameters['lr'])

        self.q = q = QNetwork(
            self.model,
            optimizer,
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters['replay_buffer_size'],
            device=self.device
        )

    def learn_step(self, idxs, transition_batch, weights):
        Otm1, old_action, env_rew, done, Ot = transition_batch
        batch_size = len(Otm1)
        actions = torch.tensor(old_action,device=self.device)
        rewards = torch.tensor(env_rew,device=self.device)
        dones = torch.tensor(done,device=self.device)
        states = StateArray({
            'observation': torch.tensor(Otm1,device=self.device),
            'reward': rewards,
            'done': dones,
            'mask': 1-dones,
        },shape=(batch_size,))
        next_states = StateArray({
            'observation': torch.tensor(Ot,device=self.device),
            'mask': 1-dones,
        },shape=(batch_size,))
        # forward pass
        values = self.q(states, actions)
        # compute targets
        targets = rewards + self.discount_factor * torch.max(self.q.target(next_states), dim=1)[0]
        # compute loss
        loss = mse_loss(values, targets)
        # backward pass
        self.q.reinforce(loss)

        # self.logger.record_mean("reward_mean", self.reward_normalizer.mean.detach().cpu().numpy())
        # self.logger.record_mean("reward_stdev", self.reward_normalizer.stdev.detach().cpu().numpy())
        self.logger.record_mean("critic_loss", loss.detach().cpu().numpy())
        self.logger.record_sum("learner_steps", batch_size)
