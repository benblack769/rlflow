from sb3_example import SB3Wrapper, SB3LearnWrapper, SB3OnlineLearnWrapper
from rlflow.env_loops.single_threaded_env_loop import run_loop
import gym
from rlflow.policy_delayer.no_update import NoUpdate
from rlflow.policy_delayer.occasional_update import OccasionalUpdate
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors import UniformSampleScheme
from rlflow.utils.logger import make_logger
from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.utils import get_schedule_fn
import supersuit
from supersuit.gym_wrappers import continuous_actions
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecTransposeImage

from rlflow.wrappers.adder_wrapper import AdderWrapper
from gym.vector import SyncVectorEnv

def main():
    n_envs = 8
    env_id = "CartPole-v0"
    def env_fn():
        return continuous_actions(gym.make(env_id))
    env = env_fn()
    #print(env.observation_space)
    #obs_size, = env.observation_space.shape
    #act_size = env.action_space.n

    env = DummyVecEnv([env_fn])
    eval_env = DummyVecEnv([env_fn])

    n_timesteps = 1000
    save_path = "log"
    eval_freq = 50

    tensorboard_log = ""

    model = TD3(env=env, tensorboard_log=tensorboard_log, policy=MlpPolicy)
    learner = SB3LearnWrapper(model)
    device = "cpu"
    learn_rate = lambda x: 0.01
    policy = SB3Wrapper(model.policy)#MlpPolicy(env.observation_space, env.action_space, learn_rate, device="cpu"))
    data_store_size = 12800
    batch_size = 16
    logger = make_logger("log")
    run_loop(
        logger,
        learner,#A2CLearner(policy, 0.001, 0.99, logger, device),
        OccasionalUpdate(10, policy),
        StatelessActor(policy),
        env_fn,
        SyncVectorEnv,
        lambda: TransitionAdder(env.observation_space, env.action_space),
        AdderWrapper,
        UniformSampleScheme(data_store_size),
        data_store_size,
        batch_size
    )
main()
