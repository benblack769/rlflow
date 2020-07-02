from ddpg_example import ActCriticPolicy, DDPGLearner, ClippedGuassianNoiseModel, BoxActionNormalizer
#from rlflow.env_loops.single_threaded_env_loop import run_loop
from rlflow.env_loops.multi_threaded_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from rlflow.vector import MakeCPUAsyncConstructor
from gym.vector import SyncVectorEnv, AsyncVectorEnv
from supersuit.gym_wrappers import normalize_obs, continuous_actions, down_scale, dtype
import numpy as np

def env_fn():
    env = down_scale(continuous_actions(gym.make("SpaceInvaders-v4")), 2, 3)
    #env._max_episode_steps = 50
    return env

# def env_fn():
#     return continuous_actions(gym.make("CartPole-v0"))


def main():
    env = env_fn()

    device = "cuda"
    noise_model = ClippedGuassianNoiseModel()
    action_normalizer = lambda: BoxActionNormalizer(env.action_space, device)
    policy_fn_dev = lambda device: ActCriticPolicy(env.observation_space, env.action_space, device, noise_model)
    policy_fn = lambda: policy_fn_dev(device)
    data_store_size = 20000
    batch_size = 128
    n_envs = 64
    n_cpus = 8
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DDPGLearner(policy_fn, action_normalizer, 0.001, 0.99, 0.1, logger, device),
        OccasionalUpdate(10, policy_fn_dev("cpu")),
        lambda: StatelessActor(policy_fn()),
        env_fn,
        MakeCPUAsyncConstructor(n_cpus),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size,
        n_envs,
        log_frequency=15,
    )
main()
