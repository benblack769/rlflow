import torch
from rlflow.base_policy import StatelessPolicy
import numpy as np
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.save_util import recursive_getattr
from collections import deque
from stable_baselines3.common.type_aliases import RolloutBufferSamples, ReplayBufferSamples


import torch as th
import torch.nn.functional as F
import torch as th

class SB3Wrapper(StatelessPolicy):
    def __init__(self, sb3_policy):
        self.policy = sb3_policy
        self.device = sb3_policy.device

    def calc_action(self, observations):
        #observations = torch.tensor(observations, device=self.device)
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

class ReplayBuffWrap:
    def __init__(self, transition_batch):
        self.transition_batch = transition_batch

    def sample(self, batch_size, env=None):
        #assert batch_size == len(self.transition_batch[])
        return self.transition_batch

class RolloutBuffWrap:
    def __init__(self, transition_batch):
        self.transition_batch = transition_batch

    def get(self, batch_size, env=None):
        assert batch_size is None
        return self.transition_batch

class SB3LearnWrapper:
    def __init__(self, sb3_learner):
        self.sb3_learner = sb3_learner
        self.device = sb3_learner.device
        self.policy = SB3Wrapper(sb3_learner.policy)

        self.sb3_learner.ep_info_buffer = deque(maxlen=100)
        self.sb3_learner.ep_success_buffer = deque(maxlen=100)

        if self.sb3_learner.action_noise is not None:
            self.sb3_learner.action_noise.reset()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        return th.tensor(array, requires_grad=False).to(self.device)

    def learn_step(self, transition_batch):
        batch_size = len(transition_batch[0])
        cur_obs, action, rew, done, last_obs = transition_batch
        done = done.astype(np.float32)
        action = action.astype(np.float32)
        cur_obs = cur_obs.astype(np.float32)
        rew = rew.astype(np.float32)
        last_obs = last_obs.astype(np.float32)
        sb3_trans = (last_obs, action, cur_obs, done, rew)
        # print([s.dtype for s in sb3_trans])
        # print(rew)
        data = ReplayBufferSamples(*tuple(map(self.to_torch, sb3_trans)))
        buff_wrap = ReplayBuffWrap(data)
        self.sb3_learner.replay_buffer = buff_wrap
        gradient_steps = 1
        self.sb3_learner.train(gradient_steps, batch_size=batch_size, policy_delay=1)

class SB3OnlineLearnWrapper:
    def __init__(self, sb3_learner):
        self.sb3_learner = sb3_learner
        self.policy = SB3Wrapper(sb3_learner.policy)

        self.sb3_learner.ep_info_buffer = deque(maxlen=100)
        self.sb3_learner.ep_success_buffer = deque(maxlen=100)

        if self.sb3_learner.action_noise is not None:
            self.sb3_learner.action_noise.reset()


    def learn_step(self, transition_batch):
        data = ReplayBufferSamples(*tuple(map(self.to_torch, transition_batch)))
        buff_wrap = RolloutBuffWrap(data)
        self.sb3_learner.rollout_buffer = buff_wrap
        self.sb3_learner.train()#gradient_steps)#, batch_size=batch_size, policy_delay=1)
        #return


class TD3Learner:
    def __init__(self, policy, lr, gamma, logger, device):
        self.policy = policy
        self.gamma = gamma
        self.logger = logger
        self.device = device
        self.actor_target = self.policy.policy.actor_target
        self.critic = self.policy.policy.critic
        self.actor = self.policy.policy.actor

        self.target_policy_noise = 0.2
        self.target_noise_clip = 0.5
        self.policy_delay = 2
        self.gamma = 0.99
        self.tau = 0.005
        #self.optimizer = torch.optim.Adam(self.policy.get_tensors(), lr=lr)

    def learn_step(self, transition_batch):
        #model = self.policy.model
        Otm1, action, rew, done, Ot = transition_batch
        observations = torch.tensor(Otm1, device=self.device)
        actions = torch.tensor(action, device=self.device)
        rewards = torch.tensor(rew, device=self.device)
        dones = torch.tensor(done, device=self.device)
        next_observations = torch.tensor(Ot, device=self.device)

        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_observations, next_actions)
            target_q = th.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(observations, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if gradient_step % policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(observations,
                                                 self.actor(observations)).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the frozen target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
