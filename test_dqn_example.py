# from gym.vector import SyncVectorEnv
import gym
import torch
import multiprocessing as mp
from atari_model import AtariModel
from atari_preprocessing import AtariWrapper

import numpy as np
import torch as th
import multiprocessing as mp
from torch.nn import functional as F
from dqn_example import DQNLearner, DQNPolicy

from rlflow.env_loops.single_threaded_env_loop import run_loop
# from rlflow.env_loops.multi_threaded_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme, UniformSampleScheme
from rlflow.utils.logger import make_logger
from rlflow.vector import MakeCPUAsyncConstructor
from rlflow.selectors.priority_updater import PriorityUpdater, NoUpdater
from gym.vector import SyncVectorEnv, AsyncVectorEnv
from supersuit.gym_wrappers import normalize_obs, resize, dtype
import numpy as np
import supersuit.aec_wrappers
from pettingzoo.sisl import waterworld_v0
from rlflow.vector import ConcatVecEnv, MarkovVectorEnv, SingleVecEnv, SpaceWrap
from rlflow.utils.saver import Saver, load_latest
import supersuit
from torch import nn

def env_fn():
    return gym.make("CartPole-v0")

def obs_preproc(obs):
    return obs

def main():
    env = env_fn()
    cpu_count = mp.cpu_count()
    # cpu_count = 0
    num_envs = 32
    num_cpus = 0
    num_targets = 1
    model_features = 512
    data_store_size = 10000
    batch_size = 512
    max_grad_norm = 0.1
    device="cuda"
    num_actors = 1
    max_learn_steps = 100000

    save_folder = "savedata/"
    def policy_fn_dev(device):
        policy = DQNPolicy(env, device)
        # load_latest(save_folder, policy)
        return policy

    priority_updater = NoUpdater()
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DQNLearner(policy_fn_dev(device), logger, env.action_space.n, device=device),
        OccasionalUpdate(100, lambda: policy_fn_dev("cpu")),
        lambda: StatelessActor(policy_fn_dev(device)),
        env_fn,
        Saver(save_folder),
        # MakeCPUAsyncConstructor(n_cpus),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        UniformSampleScheme(data_store_size),
        data_store_size,
        batch_size,
        num_cpus=num_cpus,
        num_env_ids=num_envs,
        priority_updater=priority_updater,
        log_frequency=5,
        max_learn_steps=max_learn_steps,
        # act_steps_until_learn=10000,
        # num_actors=num_actors,
    )
if __name__=="__main__":
    main()
