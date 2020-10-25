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
from diversity_agent import TargetTransitionAdder, DiversityLearner, TargetUpdaterActor, DiversityPolicy

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
from rlflow.vector import ConcatVecEnv, aec_to_markov, MarkovVectorEnv, SingleVecEnv, SpaceWrap
from rlflow.utils.saver import Saver, load_latest
import supersuit

def env_fn():
    env = AtariWrapper(gym.make("SpaceInvadersNoFrameskip-v4"),clip_reward=False)
    env = supersuit.frame_stack_v1(env, 4)
    env = supersuit.observation_lambda_v0(env,lambda obs: np.transpose(obs, axes=(2,0,1)))
    # env = supersuit.dtype_v0(env,np.float32)
    # env = supersuit.normalize_obs_v0(env)
    return env

def obs_preproc(obs):
    return obs.float()/255.

def main():
    env = env_fn()
    cpu_count = mp.cpu_count()
    # cpu_count = 0
    num_envs = 8
    num_cpus = 4
    num_targets = 1
    model_features = 512
    data_store_size = 100000
    batch_size = 512
    max_grad_norm = 0.1
    num_actions = env.action_space.n
    device="cuda"
    num_actors = 1

    # venv = MakeCPUAsyncConstructor(cpu_count)([env_fn]*num_envs, env.observation_space, env.action_space)
    # venv.reset()
    def model_fn():
        return AtariModel()

    save_folder = "savedata/"
    def policy_fn_dev(device):
        policy = DiversityPolicy(model_fn, model_features, num_actions, num_targets, obs_preproc, device)
        load_latest(save_folder, policy)
        return policy

    policy_fn = lambda: policy_fn_dev(device)
    priority_updater = NoUpdater()
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DiversityLearner(lr=0.0001, gamma=0.9, max_grad_norm=max_grad_norm, obs_preproc=obs_preproc, model_fn=model_fn, model_features=model_features, logger=logger, device=device, num_targets=num_targets, num_actions=num_actions),
        OccasionalUpdate(200, lambda: policy_fn_dev("cpu")),
        lambda: TargetUpdaterActor(policy_fn(), num_envs//num_actors, num_targets, target_staggering=1.314),
        env_fn,
        Saver(save_folder),
        # MakeCPUAsyncConstructor(n_cpus),
        lambda: TargetTransitionAdder(env.observation_space, env.action_space, num_targets),
        UniformSampleScheme(data_store_size),
        data_store_size,
        batch_size,
        num_cpus=num_cpus,
        num_env_ids=num_envs,
        priority_updater=priority_updater,
        log_frequency=5,
        max_learn_steps=10000000,
        act_steps_until_learn=100000,
        # num_actors=num_actors,
    )
if __name__=="__main__":
    main()
