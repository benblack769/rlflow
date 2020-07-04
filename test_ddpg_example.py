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
from supersuit.gym_wrappers import normalize_obs, continuous_actions, down_scale, dtype
import numpy as np

# def env_fn():
#     env = down_scale(continuous_actions(gym.make("SpaceInvaders-v4")), 2, 3)
#     #env._max_episode_steps = 50
#     return env

def env_fn():
    return continuous_actions(gym.make("CartPole-v0"))


def main():
    env = env_fn()

    device = "cuda"
    noise_model = ClippedGuassianNoiseModel()
    action_normalizer = lambda: BoxActionNormalizer(env.action_space, device)
    policy_fn_dev = lambda device: ActCriticPolicy(env.observation_space, env.action_space, device, noise_model)
    policy_fn = lambda: policy_fn_dev(device)
    reward_normalizer_fn = lambda: AdaptiveRewardNormalizer(device)
    data_store_size = 50000
    batch_size = 256
    n_envs = 128
    n_cpus = 0
    priority_updater = PriorityUpdater(alpha=0.5)
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DDPGLearner(policy_fn, action_normalizer, reward_normalizer_fn, 0.001, 0.99, 0.1, logger, priority_updater, device),
        OccasionalUpdate(10, policy_fn_dev("cpu")),
        lambda: StatelessActor(policy_fn()),
        env_fn,
        MakeCPUAsyncConstructor(n_cpus),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size,
        n_envs=n_envs,
        priority_updater=priority_updater,
        log_frequency=5,
    )
main()
