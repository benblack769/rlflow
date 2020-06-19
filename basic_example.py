import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np

class FCPolicy(StatelessPolicy):
    def __init__(self, in_dim, out_dim, hidden_dim, device):
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim
        self.model = model.to(device)
        self.device = device

    def calc_action(self, observations):
        observations = torch.tensor(observations, device=self.device)
        greedy = torch.argmax(self.model(observations),axis=1).detach().cpu().numpy()
        random = np.random.randint(0,self.out_dim,size=len(observations))
        epsilon = 0.1
        pick_rand = np.random.random(size=len(observations)) < epsilon
        actions = np.where(pick_rand, random, greedy)
        return actions

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.model.parameters()):
            dest.data = torch.tensor(source, device=self.device)

class DQNLearner:
    def __init__(self, policy, lr, gamma, logger, device):
        self.policy = policy
        self.gamma = gamma
        self.logger = logger
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)

    def learn_step(self, transition_batch):
        model = self.policy.model
        Otm1, action, rew, done, Ot = transition_batch
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
        self.logger.record("loss", final_loss)
