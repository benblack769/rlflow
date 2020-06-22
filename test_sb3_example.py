from sb3_example import SB3Wrapper, A2CLearner
from rlflow.env_loops.single_threaded_env_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.utils import get_schedule_fn


def env_fn():
    return gym.make("CartPole-v0")

def main():
    env = env_fn()
    print(env.observation_space)
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    device = "cpu"
    learn_rate = lambda x: 0.01
    policy = SB3Wrapper(MlpPolicy(env.observation_space, env.action_space, learn_rate, device="cpu"))
    data_store_size = 12800
    batch_size = 16
    logger = make_logger("log")
    run_loop(
        logger,
        A2CLearner(policy, 0.001, 0.99, logger, device),
        OccasionalUpdate(10, policy),
        StatelessActor(policy),
        env_fn,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size
    )
main()
