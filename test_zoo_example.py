from basic_example import FCPolicy, DQNLearner
from rlflow.env_loops.single_threaded_env_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders import TransitionAdder
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from rlflow.vector import ConcatVecEnv, aec_to_markov, MarkovVectorEnv
from pettingzoo.mpe import simple_world_comm_v0
from supersuit.aec_wrappers import pad_observations, pad_action_space
import copy

def env_fn():
    #env = gym.make("CartPole-v0")#
    env = simple_world_comm_v0.env()
    # print(env.action_spaces.values())
    # exit(0)
    env = pad_observations(env)
    env = pad_action_space(env)
    markov_env = aec_to_markov(env)
    venv = MarkovVectorEnv(markov_env)
    return venv

def main():
    env = env_fn()
    print(env.observation_space)
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    device = "cpu"
    policy = FCPolicy(obs_size, act_size, 64, device)
    data_store_size = 12800
    batch_size = 16
    logger = make_logger("log")
    run_loop(
        logger,
        DQNLearner(policy, 0.001, 0.99, logger, device),
        OccasionalUpdate(10, policy),
        StatelessActor(policy),
        env_fn,
        ConcatVecEnv,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size,
        n_envs=16,
        log_frequency=5
    )
main()
