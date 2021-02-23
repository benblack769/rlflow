from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
import torch
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update

def mean(items):
    return sum(items)*(1./len(items))

def convert_size(space):
    assert len(space.shape) == 1
    return space.shape[0]

def sac_policy(obs_space, act_space):
    obs_size = convert_size(obs_space)
    act_size = convert_size(act_space)
    return nn.Sequential(
        nn.Linear(obs_size, 400),
        nn.Tanh(),
        nn.Linear(400, 300),
        nn.Tanh(),
        nn.Linear(300, act_size*2),
    )

def sac_q_fn(obs_space, act_space):
    obs_size = convert_size(obs_space)
    act_size = convert_size(act_space)
    return nn.Sequential(
        nn.Linear(obs_size+act_size, 400),
        nn.Tanh(),
        nn.Linear(400, 300),
        nn.Tanh(),
        nn.Linear(300, 1),
    )

class SACCritic(nn.Module):
    def __init__(self, obs_space, act_space, n_critics=2):
        super().__init__()
        self.qs = nn.ModuleList([sac_q_fn(obs_space, act_space) for i in range(n_critics)])

    def forward(self, obs, actions):
        q_input = torch.cat([obs,actions],dim=1)
        return [q_net(q_input).flatten() for q_net in self.qs]

    def q1_forward(self, obs, actions):
        q_input = torch.cat([obs,actions],dim=1)
        return self.qs[0](q_input).flatten()


class SACPolicy(nn.Module):
    def __init__(self, obs_space, act_space, device):
        super().__init__()
        self.device = device
        self.policy = sac_policy(obs_space, act_space).to(device)
        self.critic = SACCritic(obs_space, act_space).to(device)
        self.critic_target = SACCritic(obs_space, act_space).to(device)
        self.act_size = convert_size(act_space)

    def calc_distribution(self, obs):
        data = self.policy(obs)
        mean = data[:,:self.act_size]
        stdev = data[:,self.act_size:].exp()
        return torch.distributions.normal.Normal(mean, stdev)

    def calc_action(self, obs):
        obs = torch.tensor(obs,device=self.device)
        dist = self.calc_distribution(obs)
        action = dist.sample()
        # print(action)
        return action.detach().cpu().numpy()

    def forward(self, obs):
        dist = self.calc_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).flatten()
        return action, log_prob

    def get_params(self):
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_params(self, params):
        for source,dest in zip(params, self.parameters()):
            dest.data = torch.tensor(source, device=self.device)

class AnnealedAdam:
    def __init__(self, params, T_max, lr):
        self.base_optim = torch.optim.Adam(params, lr=lr)
        self.annealer = torch.optim.lr_scheduler.CosineAnnealingLR(self.base_optim, T_max)

    def step(self):
        self.base_optim.step()
        self.annealer.step()

    def get_last_lr(self):
        return self.annealer.get_last_lr()

    def zero_grad(self):
        self.base_optim.zero_grad()


class SACLearner:
    def __init__(self, policy, T_max, device, logger,
            gamma=0.99,
            tau=0.005):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.logger = logger
        self.target_entropy = -float(policy.act_size)
        self.policy = policy
        init_value = 1.
        self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)

        self.policy_optimizer = AnnealedAdam(self.policy.policy.parameters(), T_max=T_max, lr=1e-4)
        self.critic_optimizer = AnnealedAdam(self.policy.critic.parameters(), T_max=T_max, lr=1e-3)
        self.ent_coef_optimizer = AnnealedAdam([self.log_ent_coef], T_max=T_max, lr=1e-3)


    def learn_step(self, idxs, transition_batch, weights):
        Otm1, old_action, env_rew, done, Ot = transition_batch
        batch_size = len(Ot)
        observations = (torch.tensor(Otm1, device=self.device))
        actions = torch.tensor(old_action, device=self.device)
        rewards = torch.tensor(env_rew, device=self.device)
        done = torch.tensor(done, device=self.device).float().to(self.device)
        next_observations = (torch.tensor(Ot, device=self.device))
        # weights = torch.tensor(weights, device=self.device)
        with torch.no_grad():
            actions_pred, log_probs_pred = self.policy(observations)
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_probs_pred + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        with th.no_grad():
            next_actions, next_log_prob = self.policy(next_observations)
            targets = self.policy.critic_target(next_observations, next_actions)
            targets = torch.stack(targets, dim=1)
            target_q, _ = torch.min(targets,dim=1)

            target_q = target_q - ent_coef * next_log_prob
            # td error + entropy term
            q_backup = rewards + (1 - done) * self.gamma * target_q

        current_q_estimates = self.policy.critic(observations, actions)
        critic_loss = mean([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        q_values_pi = th.stack(self.policy.critic.forward(observations, actions_pred), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_probs_pred - min_qf_pi).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

        logger = self.logger
        logger.record_mean("ent_coef_loss",ent_coef_loss.item())
        logger.record_mean("critic_loss",critic_loss.item())
        logger.record_mean("actor_loss",actor_loss.item())
        logger.record_mean("q_backup",q_backup.mean().item())
        logger.record("policy_lr",self.policy_optimizer.get_last_lr())
