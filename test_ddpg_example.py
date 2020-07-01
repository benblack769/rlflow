from ddpg_example import ActCriticPolicy, DDPGLearner, ClippedGuassianNoiseModel, BoxActionNormalizer
#from rlflow.env_loops.single_threaded_env_loop import run_loop
from rlflow.env_loops.multi_threaded_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.wrappers.adder_wrapper import AdderWrapper
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from gym.vector import SyncVectorEnv, AsyncVectorEnv
from supersuit.gym_wrappers import normalize_obs, continuous_actions, down_scale, dtype
import numpy as np

def env_fn():
    env = dtype(continuous_actions(gym.make("SpaceInvaders-v4")), np.float32)
    #env._max_episode_steps = 50
    return env

# def env_fn():
#     return continuous_actions(gym.make("CartPole-v0"))
# def MakeMultiprocEnv(num_envs, num_cpus):



def main():
    env = env_fn()

    device = "cuda"
    noise_model = ClippedGuassianNoiseModel()
    action_normalizer = lambda: BoxActionNormalizer(env.action_space, device)
    policy_fn_dev = lambda device: ActCriticPolicy(env.observation_space, env.action_space, device, noise_model)
    policy_fn = lambda:policy_fn_dev(device)
    data_store_size = 58000
    batch_size = 128
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DDPGLearner(policy_fn, action_normalizer, 0.001, 0.99, 0.1, logger, device),
        OccasionalUpdate(10, policy_fn_dev("cpu")),
        lambda: StatelessActor(policy_fn()),
        env_fn,
        SyncVectorEnv,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        AdderWrapper,
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size,
        log_frequency=15,
    )
main()
