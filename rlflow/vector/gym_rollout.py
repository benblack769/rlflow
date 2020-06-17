import numpy as np
import random

class BasePolicy:
    def rollout_step(self, obs, infos, state=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], Optional[float]) actions, states
        """
        raise NotImplementedError

class RandomPolicy:
    '''
    example of a simple batched policy.
    '''
    def __init__(self, act_space, batch_size, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.act_space = act_space

    def rollout_step(self, obs, infos, state=None, deterministic=False):
        assert len(obs) == self.batch_size
        if 'legal_moves' in infos[0]:
            actions = [random.choice(inf['legal_moves']) for inf in infos]
        else:
            actions = [self.act_space.sample() for _ in range(self.batch_size)]
        state = None
        return actions, state

class BatchPolicyWrapper:
    '''
    wraps a policy that acts on single observations into a policy
    that acts on batched observations.

    Should not use for wrapping neural networks for performance reasons,
    these should be altered for batched execution by hand.
    '''
    def __init__(self, base_policy, batch_size):
        self.policy = base_policy
        self.batch_size = batch_size

    def start_state(self):
        base_state = self.policy.start_state()
        if base_state is None:
            return None
        else:
            return np.array([base_state]*self.batch_size)

    def rollout_step(self, obs, infos, state=None, deterministic=False):
        if state is None:
            state = [None]*self.batch_size

        results = [self.policy.rollout_step(o,inf,s,deterministic) for o,s,inf in zip(obs,state,infos)]
        actions = [act for act,state in results]
        states = [state for act,state in results]
        return actions, states


class RolloutBuilder:
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.num_envs = self.vec_env.num_envs
        self.states = None
        self.prev_observes = None
        self.obs_buffer = None
        self.prev_infos = None

    def restart(self, policy):
        observations = self.vec_env.reset()
        self.prev_observes = observations
        self.prev_infos = [{} for _ in range(self.num_envs)]
        self.start_states = policy.start_state()
        self.states = self.start_states

    def rollout(self, policy, n_steps, deterministic=False):
        assert self.prev_observes is not None, "must call restart()  before rollout()"
        num_envs = self.vec_env.num_envs
        rews = np.empty((n_steps,self.num_envs),dtype=np.float64)
        dones = np.empty((n_steps,self.num_envs),dtype=np.bool)
        infos = []
        for x in range(n_steps):
            actions,states = policy.rollout_step(self.prev_observes, self.prev_infos, self.states)

            obs, rew, done, info = self.vec_env.step(actions)

            if self.obs_buffer is None or len(self.obs_buffer) != n_steps:
                # cache observation buffer between rollout so it doesn't have to always reallocate
                self.obs_buffer = np.empty((n_steps,self.num_envs)+obs.shape,dtype=obs.dtype)

            self.obs_buffer[x] = obs
            rews[x] = rew
            dones[x] = done
            infos.append(info)

            # clear states of done environments
            for i in range(num_envs):
                if done[i]:
                    states[i] = self.start_states[i]

            self.states = states
            self.prev_observes = obs
            self.prev_infos = info

        return self.obs_buffer, rews, dones, infos

def transpose_rollout(obss, rews, dones, infos):
    obss = np.asarray(obss)
    obss = obss.transpose((1,0)+tuple(range(2,len(obss.shape))))
    rews = np.asarray(rews,dtype=np.float64).T
    dones = np.asarray(dones,dtype=np.bool).T
    infos = [[infos[i][j] for i in range(len(infos))] for j in range(len(infos[0]))]
    return obss, rews, dones, infos

def split_rollouts_on_dones(batch_obs,batch_rews,batch_dones, batch_infos):
    assert len(batch_obs) == len(batch_rews) == len(batch_dones) == len(batch_infos)
    assert len(batch_obs) > 0
    n_steps = len(batch_obs[0])
    assert n_steps > 0

    result_ranges = []
    for i in range(len(batch_obs)):
        dones = np.asarray(batch_dones[i])
        sidx = 0
        fidx = np.argmax(dones)
        while dones[fidx]:
            if sidx != fidx:
                result_ranges.append((i,(sidx,fidx)))
            dones[fidx] = False
            fidx = np.argmax(dones)

        result_ranges.append((i,(sidx,n_steps)))

    observes = [batch_obs[i,s:f] for i, (s,f) in result_ranges]
    rewards = [batch_rews[i,s:f] for i, (s,f) in result_ranges]
    infos = [batch_infos[i][s:f] for i, (s,f) in result_ranges]

    return observes,rewards,infos
