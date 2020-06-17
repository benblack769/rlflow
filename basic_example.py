import torch
from rlflow.base_policy import StatelessPolicy

class FCPolicy(StatelessPolicy):
    def __init__(self, in_dim, out_dim, hidden_dim):
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        self.model = model

    def calc_action(self, observations):
        observations = torch.tensor(observations)
        return torch.argmax(self.model(observations),axis=1).detach().cpu().numpy()

    def get_params(self):
        return list(self.model.parameters())

    def set_params(self, params):
        for source,dest in zip(params, self.model.parameters()):
            dest.data = source.data

class DQNLearner:
    def __init__(self, policy, lr, gamma):
        self.policy = policy
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)


    def learn_step(self, transition_batch):
        model = self.policy.model
        Otm1, rew, action, Ot = transition_batch

        with torch.no_grad():
            future_rew = self.gamma * np.max(model(Ot),axis=1)

        total_rew = rew + future_rew

        model.zero_grad()
        qvals = model(Otm1)
        taken_qvals = qvals[torch.arange(len(qvals)),action]

        q_loss = (taken_qvals - total_rew)**2
        loss.backward()
        optimizer.step()
