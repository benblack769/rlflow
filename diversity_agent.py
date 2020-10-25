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
        self.comb_layer1 = nn.Linear(model_features+num_targets, model_features, bias=False)
        self.comb_layer2 = nn.Linear(model_features+num_targets, model_features, bias=False)
        self.action_layer = nn.Linear(model_features, num_targets*num_actions, bias=False)
        self.mean_layer = nn.Linear(model_features, num_targets, bias=False)
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.action_layer.weight.data *= 0.1
        self.mean_layer.weight.data *= 0.1

    def q_value(self, features, targets):
        input = torch.cat([features, targets], axis=1)
        comb1 = torch.relu(self.comb_layer1(input))
        comb2 = torch.relu(self.comb_layer2(input))
        advantages = self.action_layer(comb1)
        means = self.mean_layer(comb2).unsqueeze(2)
        advantages = advantages.view(-1, self.num_targets, self.num_actions)
        advantages -= advantages.mean(axis=-1).view(-1,self.num_targets,1)
        return means + advantages


class QValueLayers(nn.Module):
    def __init__(self, model_features, num_targets, num_actions, num_duplicates = 5):
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
        actions = self.policy.calc_action(observations, self.targets)
        self.update_targets(dones)
        return actions, self.targets

class DiversityPolicy(nn.Module):
    def __init__(self, model_fn, model_features, num_actions, num_targets, obs_preproc, device):
        super().__init__()
        self.obs_preproc = obs_preproc
        self.policy_features = model_fn().to(device)
        self.model_features = model_features
        self.device = device
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.q_value_layers = QValueLayers(model_features, num_targets, num_actions).to(device)
        self.steps = 0

    def calc_action(self, observation, target):
        batch_size = len(observation)
        observation = self.obs_preproc(torch.tensor(observation,device=self.device))
        target = torch.tensor(target,device=self.device)
        features = self.policy_features(observation)
        q_values = self.q_value_layers.values[0].q_value(features, target)
        greedy = torch.argmax(torch.mean(q_values,axis=1),axis=-1)
        rand_vals = torch.randint(self.num_actions,size=(batch_size,),device=self.device)
        epsilon = 1.0 if self.steps < 100000 else 1-(self.steps - 1000) / 1000 + 0.02
        pick_rand = torch.rand((batch_size,),device=self.device) < epsilon
        actions = torch.where(pick_rand, rand_vals, greedy)
        # print(actions, greedy)
        self.steps += 1
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
        self.pred_layer = nn.Linear(model_features, num_targets, bias=False)

    def forward(self, observation):
        return torch.relu(self.pred_layer(self.feature_extractor(observation)))

class DiversityLearner:
    def __init__(self, lr, gamma, model_fn, model_features, max_grad_norm, logger, device, num_targets, num_actions, obs_preproc, target_policy_delay=0.999):
        self.policy = DiversityPolicy(model_fn, model_features, num_actions, num_targets, obs_preproc, device)
        self.target_policy = DiversityPolicy(model_fn, model_features, num_actions, num_targets, obs_preproc, device)
        main_params = dict(self.policy.named_parameters())
        target_params = dict(self.target_policy.named_parameters())
        for n in main_params:
            target_params[n].data = main_params[n]

        self.predictor = Predictor(model_fn, model_features, num_targets).to(device)
        self.gamma = gamma
        self.logger = logger
        self.target_policy_delay = target_policy_delay
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.num_steps = 0
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.predictor.parameters()), lr=lr, eps=1.5e-4)
        self.distribution = torch.distributions.categorical.Categorical
        self.obs_preproc = obs_preproc

    def learn_step(self, idxs, transition_batch, weights):
        Otm1, targ_vec, old_action, env_rew, done, Ot = transition_batch
        batch_size = len(Ot)
        Otm1 = self.obs_preproc(torch.tensor(Otm1, device=self.device))
        targ_vec = torch.tensor(targ_vec, device=self.device)
        old_action = torch.tensor(old_action, device=self.device)
        env_rew = torch.tensor(env_rew, device=self.device)
        done = torch.tensor(done, device=self.device)
        Ot = self.obs_preproc(torch.tensor(Ot, device=self.device))
        weights = torch.tensor(weights, device=self.device)

        prediction_reward = self.predictor(Ot) * targ_vec

        with torch.no_grad():
            future_features = self.target_policy.policy_features(Ot)
            future_q_values = self.target_policy.q_value_layers.q_value(future_features, targ_vec)
            future_prediction_rew = ~done.view(-1,1) * torch.max(future_q_values,dim=-1).values
            discounted_fut_rew = self.gamma * future_prediction_rew
            # print(discounted_fut_rew.shape)

        total_rew = prediction_reward + discounted_fut_rew


        policy_features = self.policy.policy_features(Otm1)
        tot_q_loss = 0
        for q_layer in self.policy.q_value_layers.values:
            qvals = q_layer.q_value(policy_features, targ_vec)
            indicies = old_action.view(batch_size,1,1).repeat(1, self.num_targets, 1)
            taken_qvals = qvals.gather(2,indicies).squeeze()#[torch.arange((batch_size),dtype=torch.int64, device=self.device), old_action.view(1,-1).repeat(self.num_targets, 1)]
            # print(total_rew.max(0))
            q_loss = torch.mean((taken_qvals - total_rew.detach())**2) #+ 0.1*(taken_qvals**2).sum()
            # q_reg = 0.1*torch.mean(taken_qvals**2)
            tot_q_loss = q_loss + tot_q_loss # + q_reg

        main_params = dict(self.policy.named_parameters())
        target_params = dict(self.target_policy.named_parameters())
        # total_convexity_loss = 0
        # for n in main_params:
        #     total_convexity_loss += ((main_params[n] - target_params[n].detach())**2).mean()
        #
        # tot_q_loss += 0.1*total_convexity_loss

        tot_p_loss = -prediction_reward.mean()
        self.policy.zero_grad()
        self.predictor.zero_grad()
        tot_q_loss.backward()
        tot_p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # update target
        for n in main_params:
            target_params[n].data = self.target_policy_delay*target_params[n] + (1-self.target_policy_delay)*main_params[n]

        final_q_loss = tot_q_loss.cpu().detach().numpy()
        final_p_loss = tot_p_loss.cpu().detach().numpy()
        # self.logger.record_mean("prediction_reward", prediction_reward.mean().item())
        self.logger.record_mean("final_q_loss", final_q_loss)
        self.logger.record_mean("final_p_loss", final_p_loss)
        self.logger.record_sum("learner_steps", batch_size)

        # self.priority_updater.update_td_error(idxs, abs_td_loss.cpu().detach().numpy())

        # self.logger.record_mean("actor_loss", actor_loss.detach().cpu().numpy())
        # # self.logger.record_mean("reward_mean", self.reward_normalizer.mean.detach().cpu().numpy())
        # # self.logger.record_mean("reward_stdev", self.reward_normalizer.stdev.detach().cpu().numpy())
        # self.logger.record_mean("critic_loss", critic_loss.detach().cpu().numpy())
        # self.logger.record_sum("learner_steps", batch_size)
        self.num_steps += 1
