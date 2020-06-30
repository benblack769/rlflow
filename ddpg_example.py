import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np
import gym
import math
import warnings
from torch.nn import functional as F

class FeatureExtractor(torch.nn.Module):
    def __init__(self, obs_space, num_features, device):
        super().__init__()
        obs_shape = obs_space.shape
        final_dim = num_features
        if len(obs_shape) == 2:
            obs_shape = obs_shape+(1,)

        if len(obs_shape) == 3:
            if obs_shape[0] < 6 or obs_shape[1] < 6:
                warnings.warn("observation space is screwed up. Flateening observation")
                obs_shape = (np.prod(obs_shape),)
            else:
                lim_dim = max(obs_shape[0],obs_shape[1])
                num_layers = int(math.log2(lim_dim) - math.log2(4))
                all_layer_sizes = [24,32,48,64,96]
                layer_sizes = [obs_shape[-1]]+all_layer_sizes[len(all_layer_sizes)-num_layers:]
                layers = []
                cur_shape = obs_shape
                for i in range(num_layers):
                    layers.append(torch.nn.Conv2d(layer_sizes[i], layer_sizes[i+1], 3, padding=1))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Conv2d(layer_sizes[i+1], layer_sizes[i+1], 3, padding=1))
                    #layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.MaxPool2d(kernel_size=2,stride=2))
                    cur_shape = ((cur_shape[0])//2,(cur_shape[1])//2,layer_sizes[i+1])
                layers.append(torch.nn.Flatten())
                res_size = int(np.prod(cur_shape))

                layers.append(torch.nn.Linear(res_size, final_dim))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(final_dim, final_dim))
                net = torch.nn.Sequential(*layers)

        if len(obs_shape) == 1:
            in_dim = obs_shape[0]
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
        input = input.float()#(torch.FloatTensor)
        if len(input.shape) == 4:
            input = torch.transpose(input, 1, 3)
        features = self.net(input)

        return features

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
    def __init__(self, obs_space, act_space, device, noise_model):
        super().__init__()
        num_features = 512
        self.feature_extractor = FeatureExtractor(obs_space, num_features, device)
        self.actor = Actor(num_features, act_space, device)
        self.critic1 = Critic(num_features, act_space, device)
        self.critic2 = Critic(num_features, act_space, device)
        self.noise_model = noise_model
        self.device = device

    def calc_action(self, observations):
        observations = torch.tensor(observations, device=self.device)
        features = self.feature_extractor(observations)
        action = self.noise_model.add_noise(self.actor.calc_action(features))
        return action.detach().cpu().numpy()

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
        return (((action + 1)/2) * (self.high - self.low)) + self.low

class DDPGLearner:
    def __init__(self, policy_fn, action_normalizer_fn, lr, gamma, target_update_val, logger, device):
        self.policy = policy_fn()
        self.delayed_policy = policy_fn()
        self.action_normalizer = action_normalizer_fn()
        copy_params(self.delayed_policy, self.policy)
        self.gamma = gamma
        self.target_update_val = target_update_val
        self.logger = logger
        self.device = device
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def learn_step(self, transition_batch):
        Otm1, action, rew, done, Ot = transition_batch
        batch_size = len(Ot)
        Otm1 = torch.tensor(Otm1, device=self.device)
        action = torch.tensor(action, device=self.device)
        rew = torch.tensor(rew, device=self.device)
        done = torch.tensor(done, device=self.device)
        Ot = torch.tensor(Ot, device=self.device)

        normalize_action = self.action_normalizer.normalize(action)

        # compute target
        with torch.no_grad():
            target_features = self.delayed_policy.feature_extractor(Ot)
            target_action = self.delayed_policy.actor.calc_action(target_features)
            target_action = self.delayed_policy.noise_model.add_noise(target_action)
            target_q = torch.min(
                self.delayed_policy.critic1.q_val(target_action, target_features),
                self.delayed_policy.critic2.q_val(target_action, target_features)
            )

        future_rew = ~done * target_q
        discounted_fut_rew = self.gamma * future_rew
        total_rew = rew + future_rew
        total_rew = total_rew.detach()

        features = self.policy.feature_extractor(Otm1)

        pred_actions = self.policy.actor.calc_action(features)
        critic_eval1 = self.policy.critic1.q_val(pred_actions, features.detach())
        actor_loss = -critic_eval1.mean()

        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        # set critic grad to zero to stop actor step updating critic
        for param in self.policy.critic1.parameters():
            param.grad[:] = 0.

        critic_eval1 = self.policy.critic1.q_val(action, features)
        critic_eval2 = self.policy.critic2.q_val(action, features)
        critic_loss = F.mse_loss(critic_eval1, total_rew) + F.mse_loss(critic_eval2, total_rew)

        critic_loss.backward()
        self.optimizer.step()

        copy_params(self.delayed_policy,self.policy,self.target_update_val)

        self.logger.record_mean("actor_loss", actor_loss.detach().cpu().numpy())
        self.logger.record_mean("critic_loss", critic_loss.detach().cpu().numpy())
        self.logger.record_sum("learner_steps", batch_size)
