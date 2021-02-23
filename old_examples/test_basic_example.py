from basic_example import FCPolicy, DQNLearner
# from rlflow.env_loops.multi_threaded_loop import run_loop
from rlflow.env_loops.single_threaded_env_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import UniformSampleScheme
from rlflow.utils.logger import make_logger
from rlflow.vector import SingleVecEnv
from rlflow.utils.saver import Saver, load_latest
from gym.vector import SyncVectorEnv

import supersuit
from pettingzoo.mpe import simple_spread_v2
from supersuit import pettingzoo_env_to_vec_env_v0

# def env_fn():
#     env = gym.make("LunarLander-v2")
#     env = SingleVecEnv([lambda: env])
#     return env
def env_fn():
    #env = gym.make("CartPole-v0")#
    env = simple_spread_v2.parallel_env()
    # print(env.action_spaces.values())
    # exit(0)
    env = supersuit.pad_observations_v0(env)
    env = supersuit.pad_action_space_v0(env)
    #env = supersuit.aec_wrappers.continuous_actions(env)
    venv = pettingzoo_env_to_vec_env_v0(env)
    return venv

def env_fn():
    return gym.make("CartPole-v0")

def main():
    env = env_fn()
    print(env.observation_space)
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    device = "cuda"
    policy_fn = lambda device: lambda: FCPolicy(obs_size, act_size, 512, device)
    data_store_size = 12800
    batch_size = 64
    n_envs = 8
    n_cpus = 0
    logger = make_logger("log")
    save_folder="basic_test_save"

    run_loop(
        logger,
        lambda: DQNLearner(policy_fn("cuda"), 0.001, 0.99, logger, device),
        OccasionalUpdate(10, policy_fn("cpu")),
        lambda: StatelessActor(policy_fn("cuda")()),
        env_fn,
        Saver(save_folder),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        UniformSampleScheme(data_store_size),
        data_store_size,
        batch_size,
        num_env_ids=n_envs,
        log_frequency=5,
        num_cpus=n_cpus,
        act_steps_until_learn=8000
    )
main()
