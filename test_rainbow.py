import os
import torch
from Rainbow.agent import Agent
from rlflow.env_loops.multi_threaded_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.policy_delayer.no_update import NoUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme, UniformSampleScheme
from rlflow.utils.logger import make_logger
from rlflow.vector import MakeCPUAsyncConstructor
from rlflow.selectors.priority_updater import PriorityUpdater
from gym.vector import SyncVectorEnv, AsyncVectorEnv
from supersuit.gym_wrappers import normalize_obs, resize, dtype
import numpy as np
import argparse
import supersuit.aec_wrappers
from pettingzoo.butterfly import pistonball_v0
from rlflow.vector import ConcatVecEnv, aec_to_markov, MarkovVectorEnv, SingleVecEnv, SpaceWrap
from rlflow.utils.saver import Saver, load_latest
import cv2

# def env_fn():
#     env = down_scale(continuous_actions(gym.make("SpaceInvaders-v4")), 2, 3)
#     #env._max_episode_steps = 50
#     return env
def env_fn():
    #env = gym.make("CartPole-v0")#
    env = pistonball_v0.env()
    # print(env.action_spaces.values())
    # exit(0)
    env = supersuit.resize(env, 84,84)
    env = supersuit.observation_lambda(env,lambda obs: np.transpose(obs, axes=(2,0,1)))
    #env = supersuit.clip_rewards(-1,1)
    # env = supersuit.aec_wrappers.pad_observations(env)
    # env = supersuit.aec_wrappers.pad_action_space(env)
    #env = supersuit.aec_wrappers.continuous_actions(env)
    markov_env = aec_to_markov(env)
    venv = MarkovVectorEnv(markov_env)
    return venv

import gym
import numpy
import numpy as np

xsize = 84
ysize = 84

def downsize(obs):
    obs = cv2.resize(obs, (xsize, ysize), interpolation=cv2.INTER_AREA)
    obs = (obs.astype(np.float32) @ GRAYSCALE_WEIGHTS).astype(np.uint8)
    return obs.reshape(84,84,1)

class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(84,84,4),dtype=np.uint8)

    def step(self, action):
        obss = []
        tot_rew = 0
        tot_done = False
        tot_info = {}
        for i in range(4):
            if not tot_done:
                obs, rew, done, info = self.env.step(action)
                tot_rew += rew
                tot_done = done
                obs = downsize(obs)
            obss.append(obs)

        res_obs = np.concatenate(obss,axis=2)
        return res_obs, tot_rew, tot_done, tot_info

    def reset(self):
        return np.concatenate([downsize(self.env.reset())]*4,axis=2)

GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
def env_fn():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariWrapper(env)
    env = supersuit.clip_reward_v0(env,-1,1)
    env = supersuit.observation_lambda_v0(env,lambda obs: np.transpose(obs, axes=(2,0,1)))
    env = SingleVecEnv([lambda:env])
    return env
old_policy = None
env = env_fn()


# def backtrace_callback(learner):
#     global old_policy
#     if old_policy is None:
#         old_policy = learner.policy
#     else:
#         obss = env.reset()
#         done = False
#         while not done:
#
#             obss, rews, dones, infos = vec_env.step(actions)
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')

args = parser.parse_args()

def main():

    save_folder = "savedata/"
    def policy_fn_dev(device,is_learner=False):
        device = torch.device(device)
        policy = Agent(device, args, env,logger,priority_updater,is_learner=is_learner)
        load_latest(save_folder, policy)
        return policy
    data_store_size = 500000
    batch_size = 256
    args.batch_size = batch_size
    n_envs = 32
    n_cpus = 32
    priority_updater = PriorityUpdater()
    logger = make_logger("log")
    print("cpu create")

    print("cpu finish create")
    run_loop(
        logger,
        lambda: policy_fn_dev("cuda:0",is_learner=True),#DDPGLearner(policy_fn, reward_normalizer_fn, 0.001, 0.99, 0.1, logger, priority_updater, device),
        OccasionalUpdate(100, lambda: policy_fn_dev("cpu")),
        lambda: StatelessActor(policy_fn_dev("cuda:0")),
        env_fn,
        Saver(save_folder),
        #MakeCPUAsyncConstructor(n_cpus),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        UniformSampleScheme(data_store_size),#, alpha=0.5, beta_fn=lambda x:0.),
        data_store_size,
        batch_size,
        act_steps_until_learn=200000,
        num_env_ids=n_envs,
        num_cpus=n_cpus,
        priority_updater=priority_updater,
        log_frequency=5.,
        max_learn_steps=10000000,
    )
    print("loopterm")
main()
