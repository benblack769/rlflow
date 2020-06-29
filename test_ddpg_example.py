from ddpg_example import ActCriticPolicy, DDPGLearner, ClippedGuassianNoiseModel, BoxActionNormalizer
from rlflow.env_loops.single_threaded_env_loop import run_loop
import gym
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.wrappers.adder_wrapper import AdderWrapper
from rlflow.selectors import DensitySampleScheme
from rlflow.utils.logger import make_logger
from gym.vector import SyncVectorEnv
from supersuit.gym_wrappers import normalize_obs

def env_fn():
    return (gym.make("BipedalWalker-v3"))

def main():
    env = env_fn()
    print(env.observation_space)
    device = "cuda"
    noise_model = ClippedGuassianNoiseModel()
    action_normalizer = lambda: BoxActionNormalizer(env.action_space, device)
    policy_fn = lambda: ActCriticPolicy(env.observation_space, env.action_space, device, noise_model)
    data_store_size = 12800
    batch_size = 16
    logger = make_logger("log")
    run_loop(
        logger,
        DDPGLearner(policy_fn, action_normalizer, 0.001, 0.99, 0.1, logger, device),
        OccasionalUpdate(10, policy_fn()),
        StatelessActor(policy_fn()),
        env_fn,
        SyncVectorEnv,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        AdderWrapper,
        DensitySampleScheme(data_store_size),
        data_store_size,
        batch_size
    )
main()
