import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np
import gym
import math
import warnings
from torch.nn import functional as F
from rlflow.contrib.extractors.adaptive_extractor import AdaptiveFeatureExtractor

FeatureExtractor = AdaptiveFeatureExtractor

class Actor(torch.nn.Module):
    def __init__(self, num_features, action_space, device):
        super().__init__()
        act_size = int(np.prod(action_space.shape))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_features, act_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(act_size*2, act_size)
        ).to(device)

    def calc_action(self, features):
        actions = self.net(features)
        return actions

class Critic(torch.nn.Module):
    def __init__(self, num_features, action_space, device):
        super().__init__()
        act_feature_size = max(64, num_features//4)
        act_size = int(np.prod(action_space.shape))
        self.act_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(act_size, act_feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(act_feature_size, num_features),
        ).to(device)
        self.final_eval = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features//2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features//2, 1)
        ).to(device)

    def q_val(self, actions, features):
        act_features = self.act_extractor(actions)
        tot_features = act_features + features
        eval = self.final_eval(tot_features)
        return eval.view(-1)

class ClippedGuassianNoiseModel:
    def __init__(self, target_noise_clip=0.5, target_policy_noise=0.2):
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

    def add_noise(self, action):
        noise = action.clone().data.normal_(0, self.target_policy_noise)
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        return (action + noise).clamp(-1, 1)

class ActCriticPolicy(torch.nn.Module):
    def __init__(self, obs_space, act_space, device, noise_model, action_normalizer):
        super().__init__()
        num_features = 512
        self.feature_extractor = FeatureExtractor(obs_space, num_features, device)
        self.actor = Actor(num_features, act_space, device)
        self.critic1 = Critic(num_features, act_space, device)
        self.critic2 = Critic(num_features, act_space, device)
        self.noise_model = noise_model
        self.action_normalizer = action_normalizer
        self.device = device

    def calc_action(self, observations):
        with torch.no_grad():
            observations = torch.tensor(observations, device=self.device)
            features = self.feature_extractor(observations)
            action = self.actor.calc_action(features)
            action = self.noise_model.add_noise(action)
            rescaled_action = self.action_normalizer.unnormalize(action)
            return rescaled_action.detach().cpu().numpy()

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.parameters()):
            dest.data = torch.tensor(source, device=self.device)

def copy_params(dest_policy, src_policy, weight=1.0):
    for dest, src in zip(dest_policy.parameters(), src_policy.parameters()):
        dest.data = src.data * weight + (1-weight)*dest.data

class BoxActionNormalizer:
    def __init__(self, action_space, device):
        self.low = torch.tensor(action_space.low, device=device)
        self.high = torch.tensor(action_space.high, device=device)

    def normalize(self, action):
        return ((action - self.low) / (self.high - self.low) * 2) - 1

    def unnormalize(self, norm_actions):
        return (((norm_actions + 1)/2) * (self.high - self.low)) + self.low

class AdaptiveRewardNormalizer:
    def __init__(self, device, update_decay=0.99):
        self.mean = torch.tensor(0.0,device=device)
        self.stdev = torch.tensor(1.0,device=device)
        self.update_decay = update_decay

    def update_stats(self, rewards):
        return
        self.mean.data = self.mean * self.update_decay + (1 - self.update_decay) * torch.mean(rewards)
        cur_stdev = torch.sqrt(torch.mean((self.mean - rewards)**2))
        # no decayed update for stdev because we want to overestimate in case of dramatic
        # variation between time steps
        # and since stdev is calculated over decayed mean, it should be very high
        # in case of dramatic varation between time steps
        self.stdev.data = self.stdev * self.update_decay + (1 - self.update_decay) * cur_stdev

    def normalize(self, rewards):
        return (rewards - self.mean) / self.stdev

    def invert(self, rewards):
        return (rewards * self.stdev) + self.mean

class DDPGLearner:
    def __init__(self, policy_fn, reward_normalizer_fn, lr, gamma, target_update_val, logger, priority_updater, device):
        self.policy = policy_fn()
        self.delayed_policy = policy_fn()
        self.reward_normalizer = reward_normalizer_fn()
        self.priority_updater = priority_updater
        copy_params(self.delayed_policy, self.policy)
        self.gamma = gamma
        self.target_update_val = target_update_val
        self.logger = logger
        self.device = device
        self.num_steps = 0
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def learn_step(self, idxs, transition_batch, weights):
        Otm1, action, rew, done, Ot = transition_batch
        batch_size = len(Ot)
        Otm1 = torch.tensor(Otm1, device=self.device)
        action = torch.tensor(action, device=self.device)
        rew = torch.tensor(rew, device=self.device)
        done = torch.tensor(done, device=self.device)
        Ot = torch.tensor(Ot, device=self.device)
        weights = torch.tensor(weights, device=self.device)

        action = self.policy.action_normalizer.normalize(action)

        # compute target
        with torch.no_grad():
            target_features = self.delayed_policy.feature_extractor(Ot)
            target_action = self.delayed_policy.actor.calc_action(target_features)
            target_action = self.delayed_policy.noise_model.add_noise(target_action)
            target_q_norm = torch.min(
                self.delayed_policy.critic1.q_val(target_action, target_features),
                self.delayed_policy.critic2.q_val(target_action, target_features)
            )
            self.reward_normalizer.update_stats(target_q_norm)
            target_q = self.reward_normalizer.invert(target_q_norm)

        future_rew = ~done * target_q
        discounted_fut_rew = self.gamma * future_rew
        total_rew = rew + future_rew
        total_rew = total_rew.detach()
        total_rew = self.reward_normalizer.normalize(total_rew)
        total_rew = total_rew.detach()
        #rint(torch.mean(rew), torch.mean(target_q_norm), torch.mean(total_rew), self.reward_normalizer.mean, self.reward_normalizer.stdev)

        features = self.policy.feature_extractor(Otm1)

        pred_actions = self.policy.actor.calc_action(features.detach())
        critic_eval1 = self.policy.critic1.q_val(pred_actions, features.detach())
        actor_loss = -critic_eval1.mean()
        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        # set critic grad to zero to stop actor step updating critic
        for param in self.policy.critic1.parameters():
            param.grad[:] = 0.

        critic_eval1 = self.policy.critic1.q_val(action, features)
        critic_eval2 = self.policy.critic2.q_val(action, features)
        td_loss_sqr = (critic_eval1 - total_rew)**2 + (critic_eval2 - total_rew)**2
        abs_td_loss = torch.sqrt(td_loss_sqr)

        critic_loss = td_loss_sqr.mean()

        critic_loss.backward()
        if self.num_steps > 5:
            self.optimizer.step()

        copy_params(self.delayed_policy,self.policy,self.target_update_val)

        self.priority_updater.update_td_error(idxs, abs_td_loss.cpu().detach().numpy())

        self.logger.record_mean("actor_loss", actor_loss.detach().cpu().numpy())
        # self.logger.record_mean("reward_mean", self.reward_normalizer.mean.detach().cpu().numpy())
        # self.logger.record_mean("reward_stdev", self.reward_normalizer.stdev.detach().cpu().numpy())
        self.logger.record_mean("critic_loss", critic_loss.detach().cpu().numpy())
        self.logger.record_sum("learner_steps", batch_size)
        self.num_steps += 1
