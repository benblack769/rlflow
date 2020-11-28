import math
import numpy as np
import torch
import copy
from torch import nn
import gym
from rlflow.utils.space_wrapper import SpaceWrapper

class TargetTransitionAdder:
    def __init__(self, observation_space, action_space, targ_vec_shape):
        self.last_observation = None
        self.on_generate = None
        self.targ_vec_shape = targ_vec_shape
        self.observation_space = SpaceWrapper(observation_space)
        self.action_space = SpaceWrapper(action_space)

    def get_example_output(self):
        return (
            self.observation_space,
            np.empty(self.targ_vec_shape, dtype=np.float32),
            self.action_space,
            np.array(0,dtype=np.float32),
            np.array(0,dtype=np.uint8),
            self.observation_space
        )

    def set_generate_callback(self, on_generate):
        assert self.on_generate is None, "set_generate_callback should only be called once"
        self.on_generate = on_generate

    def add(self, obs, action, rew, done, info, targ_vec):
        assert self.on_generate is not None, "need to call set_generate_callback before add"
        obs = np.copy(obs)
        if self.last_observation is None:
            self.last_observation = obs
        else:
            transition = (obs, targ_vec, action, rew, done, self.last_observation)
            self.on_generate(transition)
            self.last_observation = np.zeros_like(obs) if done else obs

class ValueNet(nn.Module):
    def __init__(self, model_features, num_targets, num_actions):
        super().__init__()
        self.comb_layer1 = nn.Linear(model_features+num_targets, model_features)
        self.mean_layer = nn.Linear(model_features, num_targets)
        self.num_targets = num_targets
        self.num_actions = num_actions

    def q_value(self, features, targets):
        input = torch.cat([features, targets], axis=1)
        comb1 = torch.relu(self.comb_layer1(input))
        means = self.mean_layer(comb2).unsqueeze(2)
        return means

class QValueLayer(nn.Module):
    def __init__(self, model_features, num_targets, num_actions):
        super().__init__()
        self.comb_layer1 = nn.Linear(model_features+num_targets, model_features)
        self.comb_layer2 = nn.Linear(model_features+num_targets, model_features)
        self.action_layer = nn.Linear(model_features, num_targets*num_actions)
        self.mean_layer = nn.Linear(model_features, num_targets)
        self.num_targets = num_targets
        self.num_actions = num_actions
        # self.action_layer.weight.data *= 0.1
        # self.mean_layer.weight.data *= 0.1

    def q_value(self, features, targets):
        input = torch.cat([features, targets], axis=1)
        comb1 = torch.relu(self.comb_layer1(input))
        comb2 = torch.relu(self.comb_layer2(input))
        advantages = self.action_layer(comb1)
        means = self.mean_layer(comb2).unsqueeze(2)
        advantages = advantages.view(-1, self.num_targets, self.num_actions)
        advantages -= advantages.mean(axis=-1).view(-1,self.num_targets,1)
        return means + advantages

class PolicyLayer(nn.Module):
    def __init__(self, model_features, num_targets, num_actions):
        super().__init__()
        input_size = model_features #+ num_targets
        self.comb_layer1 = nn.Linear(input_size, model_features)
        self.action_layer = nn.Linear(model_features, num_actions)
        self.num_targets = num_targets
        self.num_actions = num_actions

    def forward(self, features):
        # input = torch.cat([features, targets], axis=1)
        input = features
        comb1 = torch.relu(self.comb_layer1(input))
        action_logits = self.action_layer(comb1)
        return action_logits

class QValueLayers(nn.Module):
    def __init__(self, model_features, num_targets, num_actions, num_duplicates = 2):
        super().__init__()
        self.values = nn.ModuleList([QValueLayer(model_features, num_targets, num_actions) for i in range(num_duplicates)])
        # self.value0 = self.values[0]
        # self.value1 = self.values[1]

    def q_value(self, features, targets):
        values = [value_fn.q_value(features, targets) for value_fn in self.values]
        return torch.min(torch.stack(values),dim=0).values

class TargetUpdaterActor:
    def __init__(self, policy, batch_size, num_targets, target_staggering=None):
        if target_staggering is None:
            target_staggering = math.log(100)/math.log(num_targets)
        self.policy = policy
        self.batch_size = batch_size
        self.num_targets = num_targets
        self.target_staggering = target_staggering
        self.episode_counts = np.zeros(batch_size, dtype=np.int64)
        self.target_schedule = np.cumprod(np.ones(num_targets, dtype=np.float32)*target_staggering).astype(np.int64)
        self.targets = self.new_target()
        print(self.target_schedule)

    def new_target(self):
        return np.random.normal(size=(self.batch_size, self.num_targets)).astype(np.float32)

    def update_target(self, target, count):
        return np.where(np.equal(np.expand_dims(count,1) % self.target_schedule,0), self.new_target(), self.targets)

    def update_targets(self, dones):
        self.episode_counts += 1
        for i, d in enumerate(dones):
            if d:
                self.episode_counts[i] = 0
        self.targets = self.update_target(self.targets, self.episode_counts)

    def step(self, observations, dones, infos):
        assert len(observations) == len(dones) == self.batch_size
        actions = self.policy.calc_action(observations)#, self.targets)
        self.update_targets(dones)
        return actions, self.targets

from all.approximation import QContinuous, PolyakTarget, VNetwork, FeatureNetwork, Approximation
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from all.policies import SoftmaxPolicy
from all.logging import DummyWriter
from all.core import StateArray
from all.nn import RLNetwork
from torch.nn.functional import mse_loss


class DiversityPolicy(nn.Module):
    def __init__(self, model_fn, model_features, num_actions, num_targets, obs_preproc, device):
        super().__init__()
        self.obs_preproc = obs_preproc
        self.policy_features = model_fn().to(device)
        self.model_features = model_features
        self.device = device
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.policy_layer = PolicyLayer(model_features, num_targets, num_actions).to(device)
        self.distribution = torch.distributions.categorical.Categorical
        self.steps = 0

    def forward(self, observation):
        observation = self.obs_preproc(torch.tensor(observation,device=self.device))
        features = self.policy_features(observation)
        # target = torch.tensor(target,device=self.device)
        action_dist = self.policy_layer(features)
        return action_dist

    def calc_action(self, observation):
        action_dist = self.forward(observation)
        dist = self.distribution(logits=action_dist)
        actions = dist.sample()
        return actions.detach().cpu().numpy()

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.parameters()):
            dest.data = torch.tensor(source, device=self.device)

class Predictor(nn.Module):
    def __init__(self, model_fn, model_features, num_targets):
        super().__init__()
        self.feature_extractor = model_fn()
        self.pred_layer = nn.Linear(model_features, num_targets)

    def forward(self, observation):
        return torch.relu(self.pred_layer(self.feature_extractor(observation)))

def first(x):
    return next(iter(x))

class DuelingQValueLayer(nn.Module):
    def __init__(self, model_features, num_targets, num_actions):
        super().__init__()
        input_size = model_features#+num_targets
        self.output_size = output_size = 1#num_targets
        self.comb_layer1 = nn.Linear(input_size, model_features)
        self.comb_layer2 = nn.Linear(input_size, model_features)
        self.action_layer = nn.Linear(model_features, output_size*num_actions)
        self.mean_layer = nn.Linear(model_features, output_size)
        self.num_targets = num_targets
        self.num_actions = num_actions
        # self.action_layer.weight.data *= 0.1
        # self.mean_layer.weight.data *= 0.1

    def forward(self, features, actions):
        # input = torch.cat([features, targets], axis=1)
        batch_size = len(features)
        input = features
        comb1 = torch.relu(self.comb_layer1(input))
        comb2 = torch.relu(self.comb_layer2(input))
        advantages = self.action_layer(comb1)
        # means = self.mean_layer(comb2).unsqueeze(2)
        advantages = advantages.view(-1, self.output_size, self.num_actions)
        # advantages -= advantages.mean(axis=-1).view(-1,self.output_size,1)
        qvals = advantages#means + advantages
        indicies = actions.view(batch_size,1,1).repeat(1, self.output_size, 1)
        taken_qvals = qvals.gather(2,indicies).squeeze()#[torch.arange((batch_size),dtype=torch.int64, device=self.device), old_action.view(1,-1).repeat(self.num_targets, 1)]
        return taken_qvals.unsqueeze(1)

class ValueLayer(nn.Module):
    def __init__(self, model_features, num_targets, num_actions):
        super().__init__()
        output_size = 1#num_targets
        input_size = model_features#+num_targets
        self.comb_layer1 = nn.Linear(input_size, model_features)
        self.mean_layer = nn.Linear(model_features, num_targets)
        self.num_targets = num_targets
        self.num_actions = num_actions

    def forward(self, features):
        # input = torch.cat([features, targets], axis=1)
        input = features
        comb1 = torch.relu(self.comb_layer1(input))
        means = self.mean_layer(comb1).unsqueeze(2)
        return means.squeeze().unsqueeze(1)

class QContinuous(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='q',
            **kwargs
    ):
        model = QContinuousModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class QContinuousModule(RLNetwork):
    def forward(self, states, actions):
        return self.model(states.observation, actions).squeeze(-1) * states.mask.float()

class DiversityLearner:
    def __init__(self,
            model_fn,
            model_features,
            logger,
            device,
            num_targets,
            max_learn_steps,
            num_actions,
            obs_preproc,
            discount_factor=0.99,
            entropy_target=-2,

            lr_value=1e-3,
            lr_pi=1e-4,
            # Training settings
            polyak_rate=0.005,
            # Replay Buffer settings
            replay_start_size=5000,
            replay_buffer_size=1e6,
            # Exploration settings
            temperature_initial=0.1,
            lr_temperature=1e-5,
            entropy_target_scaling=1.,
        ):
        self.writer = writer=DummyWriter()
        eps = 1e-5
        self.discount_factor = discount_factor
        self.entropy_target = entropy_target
        self.temperature = temperature_initial
        self.lr_temperature = lr_temperature
        self.logger = logger
        self.device = device
        self.num_targets = num_targets
        self.max_learn_steps = max_learn_steps
        self.num_actions = num_actions

        final_anneal_step = (max_learn_steps)
        self.policy = DiversityPolicy(model_fn, model_features, num_actions, num_targets, obs_preproc, device)
        self.policy = self.policy.to(device)
        self.obs_preproc = obs_preproc
        policy_optimizer = Adam(self.policy.parameters(), lr=lr_pi, eps=eps)
        self.policy_learner = SoftmaxPolicy(
            self.policy,
            policy_optimizer,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
            writer=writer
        )

        value_feature_model = model_fn().to(device)
        q_models = [DuelingQValueLayer(model_features, num_targets, num_actions).to(device) for i in range(2)]
        v_model = ValueLayer(model_features, num_targets, num_actions).to(device)
        feature_optimizer = Adam(value_feature_model.parameters(), lr=lr_value, eps=eps)
        q_optimizers = [Adam(q_models[i].parameters(), lr=lr_value, eps=eps) for i in range(2)]
        v_optimizer = Adam(v_model.parameters(), lr=lr_value, eps=eps)

        self.features = FeatureNetwork(
            value_feature_model,
            feature_optimizer,
            scheduler=CosineAnnealingLR(
                feature_optimizer,
                final_anneal_step,
            ),
            # clip_grad=clip_grad,
            writer=writer
        )

        self.qs = [QContinuous(
            q_models[i],
            q_optimizers[i],
            scheduler=CosineAnnealingLR(
                q_optimizers[i],
                final_anneal_step
            ),
            writer=writer,
            name=f'q_{i}'
        ) for i in range(2)]

        self.v = VNetwork(
            v_model,
            v_optimizer,
            scheduler=CosineAnnealingLR(
                v_optimizer,
                final_anneal_step
            ),
            target=PolyakTarget(polyak_rate),
            writer=writer,
            name='v',
        )

    def learn_step(self, idxs, transition_batch, weights):
        Otm1, targ_vec, old_action, env_rew, done, Ot = transition_batch
        batch_size = len(Ot)
        obsm1 = self.obs_preproc(torch.tensor(Otm1, device=self.device))
        targ_vec = torch.tensor(targ_vec, device=self.device)
        actions = torch.tensor(old_action, device=self.device)
        rewards = torch.tensor(env_rew, device=self.device)
        done = torch.tensor(done, device=self.device).float().to(self.device)
        next_obs = self.obs_preproc(torch.tensor(Ot, device=self.device))
        weights = torch.tensor(weights, device=self.device)
        # assert (not (Otm1 == Ot).all())
        # print(self.device)
        states = StateArray({
            'observation': obsm1,
            'reward': rewards,
            'done': done,
        }, shape=(batch_size,))
        # print(states['mask'])
        next_states = StateArray({
            'observation': obsm1,
            'reward': torch.zeros(batch_size,device=self.device),
            'done': torch.zeros(batch_size,device=self.device),
            'mask': torch.ones(batch_size,device=self.device),
        }, shape=(batch_size,))

        # prediction_reward = self.predictor(Ot) * targ_vec
        with torch.no_grad():
            distribution = self.policy_learner(states)
            _log_probs = distribution.log_prob(actions).detach().squeeze()
        value_feature1 = self.features(states)
        value_feature2 = self.features(next_states)
        _actions = distribution.sample()#torch.argmax(_log_probs, axis=-1)
        q_targets = rewards + self.discount_factor * self.v.target(value_feature2).detach()
        # print(value_feature1)
        v_targets = torch.min(
            self.qs[0].target(value_feature1, _actions),
            self.qs[1].target(value_feature1, _actions),
        ) - self.temperature * _log_probs
        # update Q and V-functions
        # print(q_targets.min(),torch.min(
        #     self.qs[0].target(value_feature1, _actions),
        #     self.qs[1].target(value_feature1, _actions),
        # ))
        for i in range(2):
            self.qs[i].reinforce(mse_loss(self.qs[i](value_feature1, actions), q_targets))
        # print(self.v(value_feature1).shape)
        # print(v_targets.shape)
        self.v.reinforce(mse_loss(self.v(value_feature1), v_targets))

        # update policy
        distribution = self.policy_learner(states)
        _actions2 = distribution.sample()
        _log_probs2 = distribution.log_prob(_actions2).squeeze()
        loss = (-self.qs[0](value_feature1, _actions2).detach() + self.temperature * _log_probs2).mean()
        self.policy_learner.reinforce(loss)
        self.features.reinforce()
        self.qs[0].zero_grad()

        # adjust temperature
        temperature_grad = (_log_probs + self.entropy_target).mean()
        self.temperature += self.lr_temperature * temperature_grad.detach().cpu().numpy()

        # additional debugging info
        # self.writer.add_loss('entropy', -_log_probs.mean())
        # self.writer.add_loss('v_mean', v_targets.mean())
        # self.writer.add_loss('r_mean', rewards.mean())
        # self.writer.add_loss('temperature_grad', temperature_grad)
        # self.writer.add_loss('temperature', self.temperature)

        # update target
        # for n in main_params:
        #     target_params[n].data = self.target_policy_delay*target_params[n] + (1-self.target_policy_delay)*main_params[n]
        #
        # final_q_loss = tot_q_loss.cpu().detach().numpy()
        # final_p_loss = tot_p_loss.cpu().detach().numpy()
        # # self.logger.record_mean("prediction_reward", prediction_reward.mean().item())
        # self.logger.record_mean("final_q_loss", final_q_loss)
        # self.logger.record_mean("final_p_loss", final_p_loss)
        # self.logger.record_sum("learner_steps", batch_size)

        # self.priority_updater.update_td_error(idxs, abs_td_loss.cpu().detach().numpy())

        # self.logger.record_mean("actor_loss", actor_loss.detach().cpu().numpy())
        # # self.logger.record_mean("reward_mean", self.reward_normalizer.mean.detach().cpu().numpy())
        # # self.logger.record_mean("reward_stdev", self.reward_normalizer.stdev.detach().cpu().numpy())
        # self.logger.record_mean("critic_loss", critic_loss.detach().cpu().numpy())
        # self.logger.record_sum("learner_steps", batch_size)
        # self.num_steps += 1
