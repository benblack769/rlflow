from basic_example import FCPolicy, DQNLearner
from rlflow.env_loops.efficient_rollout_loop import run_loop
#from rlflow.env_loops.single_threaded_env_loop import run_loop
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
from pettingzoo.mpe import simple_spread_v0
from rlflow.vector import aec_to_markov, MarkovVectorEnv

# def env_fn():
#     env = gym.make("LunarLander-v2")
#     env = SingleVecEnv([lambda: env])
#     return env
def env_fn():
    #env = gym.make("CartPole-v0")#
    env = simple_spread_v0.env()
    # print(env.action_spaces.values())
    # exit(0)
    env = supersuit.aec_wrappers.pad_observations(env)
    env = supersuit.aec_wrappers.pad_action_space(env)
    #env = supersuit.aec_wrappers.continuous_actions(env)
    markov_env = aec_to_markov(env)
    venv = MarkovVectorEnv(markov_env)
    return venv

def main():
    env = env_fn()
    print(env.observation_space)
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    device = "cuda"
    policy_fn = lambda device: lambda: FCPolicy(obs_size, act_size, 512, device)
    data_store_size = 12800
    batch_size = 64
    n_envs = 128*4
    logger = make_logger("log")
    save_folder="basic_test_save"

    run_loop(
        logger,
        lambda: DQNLearner(policy_fn("cuda"), 0.001, 0.99, logger, device),
        OccasionalUpdate(10, policy_fn("cpu")),
        policy_fn("cuda"),
        env_fn,
        Saver(save_folder),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        UniformSampleScheme(data_store_size),
        data_store_size,
        batch_size,
        num_env_ids=n_envs,
        log_frequency=5,
        num_cpus=8,
        act_steps_until_learn=800000
    )
main()
