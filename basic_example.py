import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np
import random
import time
from torch import nn

class AtariPolicy(torch.nn.Module):
    def __init__(self, device):
        in_d = 4
        self.model = nn.Sequential(nn.Conv2d(in_d, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Linear(3136, 512),nn.ReLU(),
                                 nn.Linear(512, 18)).to(device)
        self.device = device

    def forward(self, observations):
        with torch.no_grad():
            observations = torch.tensor(observations, device=self.device)
            greedy = torch.argmax(self.model(observations),axis=1)#.detach().cpu().numpy()
            rand_vals = torch.randint(self.out_dim,size=(len(observations),),device=self.device)
            epsilon = 0.1
            pick_rand = torch.rand((len(observations),),device=self.device) < epsilon
            actions = torch.where(pick_rand, rand_vals, greedy)
        return actions

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.model.parameters()):
            dest.data = torch.tensor(source, device=self.device)

class Scale(torch.nn.Module):
    def __init__(self, sval):
        super().__init__()
        self.sval = sval

    def forward(self, observations):
        return observations * self.sval
from all.nn import Linear0

class FCPolicy(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, device):
        super().__init__()
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear0(hidden_dim, out_dim),
        )
        self.out_dim = out_dim
        self.model = model.to(device)
        self.device = device

    def forward(self, observations):
        with torch.no_grad():
            observations = torch.tensor(observations, device=self.device)
            greedy = torch.argmax(self.model(observations),axis=1)#.detach().cpu().numpy()
            rand_vals = torch.randint(self.out_dim,size=(len(observations),),device=self.device)
            epsilon = 0.1
            pick_rand = torch.rand((len(observations),),device=self.device) < epsilon
            actions = torch.where(pick_rand, rand_vals, greedy)
        return actions

    def calc_action(self, obs):
        return self.forward(obs).cpu().detach().numpy()

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.model.parameters()):
            dest.data = torch.tensor(source, device=self.device)

class DQNLearner:
    def __init__(self, policy_fn, lr, gamma, logger, device):
        self.policy = policy_fn()
        self.gamma = gamma
        self.logger = logger
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)

    def learn_step(self, idxs, transition_batch, weights):
        model = self.policy.model
        Otm1, action, rew, done, Ot = transition_batch
        batch_size = len(Ot)
        Otm1 = torch.tensor(Otm1, device=self.device)
        action = torch.tensor(action, device=self.device)
        rew = torch.tensor(rew, device=self.device)
        done = torch.tensor(done, device=self.device)
        Ot = torch.tensor(Ot, device=self.device)

        with torch.no_grad():
            future_rew = ~done * torch.max(model(Ot),axis=1).values
            discounted_fut_rew = self.gamma * future_rew

        total_rew = rew + future_rew

        model.zero_grad()
        qvals = model(Otm1)
        action = action.type(torch.long)
        taken_qvals = qvals[torch.arange(len(qvals),dtype=torch.int64, device=self.device),action]

        q_loss = torch.mean((taken_qvals - total_rew)**2)
        q_loss.backward()
        self.optimizer.step()
        final_loss = q_loss.cpu().detach().numpy()
        self.logger.record_mean("loss", final_loss)
        self.logger.record_sum("learner_steps", batch_size)
