from basic_example import FCPolicy, DQNLearner
from rlflow.env_loops.multi_threaded_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from gym.vector import SyncVectorEnv

def env_fn():
    return gym.make("CartPole-v0")

def main():
    env = env_fn()
    print(env.observation_space)
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    device = "cuda"
    policy_fn = lambda: FCPolicy(obs_size, act_size, 64, device)
    data_store_size = 128000
    batch_size = 64
    logger = make_logger("log")
    run_loop(
        logger,
        lambda: DQNLearner(policy_fn(), 0.001, 0.99, logger, device),
        OccasionalUpdate(10, FCPolicy(obs_size, act_size, 64, "cpu")),
        lambda: StatelessActor(policy_fn()),
        env_fn,
        SyncVectorEnv,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size,
        n_envs=32,
        log_frequency=5
    )
main()
