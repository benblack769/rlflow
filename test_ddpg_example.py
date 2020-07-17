from ddpg_example import ActCriticPolicy, DDPGLearner, ClippedGuassianNoiseModel, BoxActionNormalizer, AdaptiveRewardNormalizer
#from rlflow.env_loops.single_threaded_env_loop import run_loop
from rlflow.env_loops.multi_threaded_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from rlflow.vector import MakeCPUAsyncConstructor
from rlflow.selectors.priority_updater import PriorityUpdater
from gym.vector import SyncVectorEnv, AsyncVectorEnv
from supersuit.gym_wrappers import normalize_obs, resize, dtype
import numpy as np
import supersuit.aec_wrappers
from pettingzoo.sisl import waterworld_v0
from rlflow.vector import ConcatVecEnv, aec_to_markov, MarkovVectorEnv, SingleVecEnv, SpaceWrap
from rlflow.utils.saver import Saver, load_latest

# def env_fn():
#     env = down_scale(continuous_actions(gym.make("SpaceInvaders-v4")), 2, 3)
#     #env._max_episode_steps = 50
#     return env
def env_fn():
    #env = gym.make("CartPole-v0")#
    env = waterworld_v0.env()
    # print(env.action_spaces.values())
    # exit(0)
    env = supersuit.aec_wrappers.pad_observations(env)
    env = supersuit.aec_wrappers.pad_action_space(env)
    #env = supersuit.aec_wrappers.continuous_actions(env)
    markov_env = aec_to_markov(env)
    venv = MarkovVectorEnv(markov_env)
    return venv

# def env_fn():
#     return continuous_actions(gym.make("CartPole-v0"))


def main():
    env = env_fn()

    device = "cuda"
    noise_model = ClippedGuassianNoiseModel()
    action_normalizer_fn = lambda device: BoxActionNormalizer(env.action_space, device)
    save_folder = "savedata/"
    def policy_fn_dev(device):
        policy = ActCriticPolicy(env.observation_space, env.action_space, device, noise_model, action_normalizer_fn(device))
        load_latest(save_folder, policy)
        return policy
    policy_fn = lambda: policy_fn_dev(device)
    reward_normalizer_fn = lambda: AdaptiveRewardNormalizer(device)
    data_store_size = 50000
    batch_size = 256
    n_envs = 16
    n_cpus = 0
    priority_updater = PriorityUpdater(alpha=0.5)
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DDPGLearner(policy_fn, reward_normalizer_fn, 0.001, 0.99, 0.1, logger, priority_updater, device),
        OccasionalUpdate(10, policy_fn_dev("cpu")),
        lambda: StatelessActor(policy_fn()),
        env_fn,
        Saver(save_folder),
        MakeCPUAsyncConstructor(n_cpus),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size,
        n_envs=n_envs,
        priority_updater=priority_updater,
        log_frequency=0.1,
    )
main()
