import time

import gym
import numpy as np

from stable_baselines import ACKTR
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_vec_env

# By default, we use a DummyVecEnv as it is usually faster (cf doc)

env_id = "CartPole-v1"
num_cpu = 4  # Number of processes to use
vec_env = make_vec_env(env_id, n_envs=num_cpu)

model = ACKTR('MlpPolicy', vec_env, verbose=0)
