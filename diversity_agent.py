import math
import numpy as np
import torch
from torch import nn
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
            self.last_observation = None if done else obs

class QValueLayer(nn.Module):
    def __init__(self, model_features, num_targets, num_actions):
        super().__init__()
        self.comb_layer = nn.Linear(model_features+num_targets, model_features)
        self.action_layer = nn.Linear(model_features, num_targets*num_actions)
        self.num_targets = num_targets
        self.num_actions = num_actions

    def q_value(self, features, targets):
        input = torch.cat([features, targets], axis=1)
        comb = self.comb_layer(input)
        comb = torch.relu(comb)
        logits = self.action_layer(comb)
        logits = logits.view(-1, self.num_targets, self.num_actions)
        return logits

class QValueLayers(nn.Module):
    def __init__(self, model_features, num_targets, num_actions, num_duplicates = 2):
        super().__init__()
        self.values = [QValueLayer(model_features, num_targets, num_actions) for i in range(num_duplicates)]

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
        self.target_schedule = np.cumprod(np.ones(num_targets, dtype=np.float32)*target_staggering)
        self.targets = self.new_target()
        print(self.target_schedule)

    def new_target(self):
        return np.random.normal(size=(self.batch_size, self.num_targets)).astype(np.float32)

    def update_target(self, target, count):
        return np.where(np.equal(np.expand_dims(count,1) % self.target_schedule,0), self.targets, self.new_target())

    def update_targets(self, dones):
        self.episode_counts += 1
        for i, d in enumerate(dones):
            if d:
                self.episode_counts[i] = 0
        self.targets = self.update_target(self.targets, self.episode_counts)

    def step(self, observations, dones, infos):
        assert len(observations) == len(dones) == self.batch_size
        actions = self.policy.calc_action(observations, self.targets)
        self.update_targets(dones)
        return actions, self.targets

class DiversityPolicy(nn.Module):
    def __init__(self, model_fn, model_features, num_actions, num_targets, device):
        super().__init__()
        self.policy_features = model_fn().to(device)
        self.model_features = model_features
        self.device = device
        self.num_targets = num_targets
        self.q_value_layers = QValueLayers(model_features, num_targets, num_actions).to(device)

    def calc_action(self, observation, target):
        observation = observation.copy(order='F')
        torch.tensor(np.zeros(32))
        observation = torch.tensor(observation,device=self.device)
        target = torch.tensor(target,device=self.device)
        features = self.policy_features(observation)
        q_values = self.q_value_layers.values[0].q_value(features, target)
        action = torch.argmax(torch.mean(q_values,axis=1),axis=-1)
        return action.detach().cpu().numpy()

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
        return torch.tanh(self.pred_layer(self.feature_extractor(observation)))

class DiversityLearner:
    def __init__(self, lr, gamma, model_fn, model_features, logger, device, num_targets, num_actions):
        self.policy = DiversityPolicy(model_fn, model_features, num_actions, num_targets, device)
        self.predictor = Predictor(model_fn, model_features, num_targets).to(device)
        self.gamma = gamma
        self.logger = logger
        self.device = device
        self.num_steps = 0
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.optimizer = torch.optim.RMSprop(list(self.policy.parameters()) + list(self.predictor.parameters()), lr=lr)
        self.distribution = torch.distributions.categorical.Categorical

    def learn_step(self, idxs, transition_batch, weights):
        Otm1, targ_vec, old_action, env_rew, done, Ot = transition_batch
        batch_size = len(Ot)
        Otm1 = torch.tensor(Otm1, device=self.device)
        targ_vec = torch.tensor(targ_vec, device=self.device)
        old_action = torch.tensor(old_action, device=self.device)
        env_rew = torch.tensor(env_rew, device=self.device)
        done = torch.tensor(done, device=self.device)
        Ot = torch.tensor(Ot, device=self.device)
        weights = torch.tensor(weights, device=self.device)

        prediction_reward = self.predictor(Ot) * targ_vec

        with torch.no_grad():
            future_features = self.policy.policy_features(Ot)
            future_q_values = self.policy.q_value_layers.q_value(future_features, targ_vec)
            future_prediction_rew = ~done.view(-1,1) * torch.max(future_q_values,dim=-1).values
            discounted_fut_rew = self.gamma * future_prediction_rew
            print(discounted_fut_rew.shape)

        total_rew = prediction_reward + discounted_fut_rew

        policy_features = self.policy.policy_features(Otm1)
        tot_q_loss = 0
        for q_layer in self.policy.q_value_layers.values:
            qvals = q_layer.q_value(policy_features, targ_vec)
            indicies = old_action.view(batch_size,1,1).repeat(1, self.num_targets, 1)
            taken_qvals = qvals.gather(2,indicies).squeeze()#[torch.arange((batch_size),dtype=torch.int64, device=self.device), old_action.view(1,-1).repeat(self.num_targets, 1)]
            # print(taken_qvals.shape)

            q_loss = torch.mean((taken_qvals - total_rew)**2)
            tot_q_loss = q_loss + tot_q_loss

        self.policy.zero_grad()
        self.predictor.zero_grad()
        tot_q_loss.backward()
        self.optimizer.step()
        final_loss = tot_q_loss.cpu().detach().numpy()
        self.logger.record_mean("loss", final_loss)
        self.logger.record_sum("learner_steps", batch_size)

        # self.priority_updater.update_td_error(idxs, abs_td_loss.cpu().detach().numpy())

        # self.logger.record_mean("actor_loss", actor_loss.detach().cpu().numpy())
        # # self.logger.record_mean("reward_mean", self.reward_normalizer.mean.detach().cpu().numpy())
        # # self.logger.record_mean("reward_stdev", self.reward_normalizer.stdev.detach().cpu().numpy())
        # self.logger.record_mean("critic_loss", critic_loss.detach().cpu().numpy())
        # self.logger.record_sum("learner_steps", batch_size)
        self.num_steps += 1
