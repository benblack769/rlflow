from basic_example import FCPolicy, DQNLearner
from rlflow.env_loops.single_threaded_env_loop import run_loop
import gym
from rlflow.policy_delayer.base import NoUpdatePolicyDelayer
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger

def env_fn():
    return gym.make("CartPole-v0")

def main():
    env = env_fn()
    print(env.observation_space)
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    policy = FCPolicy(obs_size, act_size, 64)
    data_store_size = 12800
    batch_size = 16
    logger = make_logger("log")
    run_loop(
        logger,
        DQNLearner(policy, 0.001, 0.99, logger),
        NoUpdatePolicyDelayer(),
        StatelessActor(policy),
        env_fn,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size
    )
main()
