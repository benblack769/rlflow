import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.save_util import recursive_getattr

import torch as th
import torch.nn.functional as F
import torch as th

class SB3Wrapper(StatelessPolicy):
    def __init__(self, sb3_policy):
        self.policy = sb3_policy
        self.device = sb3_policy.device

    def calc_action(self, observations):
        observations = torch.tensor(observations, device=self.device)
        actions, state = self.policy.predict(observations)
        return actions#.detach().cpu().numpy()

    def get_tensors(self):
        attr = recursive_getattr(self, "policy")
        save_tensors = attr.state_dict()
        res_tensors = [tensor for name, tensor in save_tensors.items()]
        return res_tensors

    def get_params(self):
        return [tensor.cpu().detach().numpy() for tensor in self.get_tensors()]

    def set_params(self, params):
        for source,dest in zip(params,self.get_tensors()):
            dest.data = torch.tensor(source, device=self.device)


class A2CLearner:
    def __init__(self, policy, lr, gamma, logger, device):
        self.policy = policy
        self.gamma = gamma
        self.logger = logger
        self.device = device
        #self.optimizer = torch.optim.Adam(self.policy.get_tensors(), lr=lr)

    def learn_step(self, transition_batch):
        #model = self.policy.model
        Otm1, action, rew, done, Ot = transition_batch
        Otm1 = torch.tensor(Otm1, device=self.device)
        actions = torch.tensor(action, device=self.device)
        rew = torch.tensor(rew, device=self.device)
        done = torch.tensor(done, device=self.device)
        Ot = torch.tensor(Ot, device=self.device)

        # TODO: avoid second computation of everything because of the gradient
        values, log_prob, entropy = self.policy.policy.evaluate_actions(Otm1, actions)
        # print(values.shape)
        # print(log_prob.shape)
        future_values, _, _ = self.policy.policy.evaluate_actions(Ot, actions)
        values = values.flatten()
        future_values = future_values.detach().flatten()

        returns = future_values*(~done) + rew
        advantages = returns - values

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns, values)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -log_prob.mean()
        else:
            entropy_loss = -th.mean(entropy)

        ent_coef = 0.0
        vf_coef  = 0.5
        max_grad_norm = 0.5
        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

        # Optimization step
        self.policy.policy.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.policy.parameters(), max_grad_norm)
        self.policy.policy.optimizer.step()
