from rlflow.vector.aec_markov_wrapper import aec_to_markov
#from rlflow.vector import DummyVecEnv
from rlflow.vector import ProcConcatVec
from rlflow.vector import ConcatVecEnv
from rlflow.vector import MarkovVectorEnv
from rlflow.vector import VecEnvWrapper,MakeCPUAsyncConstructor

from pettingzoo.sisl import multiwalker_v0
from supersuit.aec_wrappers import pad_observations, pad_action_space
from stable_baselines3 import PPO, A2C
import time
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.logger import ScopedConfigure, configure
from stable_baselines3.common import logger
import stable_baselines3

from rlflow.utils.logger import make_logger
#from stable_baselines.acktr import MlpPolicy
import gym

def main():
    def env_contr():
        return gym.make("CartPole-v0")#
        # env = multiwalker_v0.env()
        # env = pad_observations(env)
        # env = pad_action_space(env)
        # markov_env = aec_to_markov(env)
        # venv = MarkovVectorEnv(markov_env)
        # return venv
    n_envs = 6
    # def nest_env_const():
    #     cat = ConcatVecEnv([env_contr]*envs_per_proc)
    #     return cat
    example_env = env_contr()
    num_envs =n_envs*1#example_env.num_envs
    #cat = ProcConcatVec([nest_env_const]*n_procs,example_env.observation_space, example_env.action_space, num_envs)
    cat = MakeCPUAsyncConstructor(0)([env_contr]*n_envs,example_env.observation_space, example_env.action_space)#, num_envs)
    cat = VecEnvWrapper(cat)
    env = cat
    policy = "MlpPolicy"
    logger = make_logger("log")
    stable_baselines3.common.logger.Logger.CURRENT = logger
    a2c = PPO(policy,cat,n_steps=4,batch_size=6,n_epochs=3)
    print(type(a2c.env))
    #a2c.learn(1000000)

    total_timesteps, callback = a2c._setup_learn(
        10000, None, None, None, n_eval_episodes=5, reset_num_timesteps=None, tb_log_name="PPo"
    )

    #total_timesteps = 100
    iteration = 0
    log_interval = 1
    for i in range(total_timesteps):
        continue_training = a2c.collect_rollouts(env, callback, a2c.rollout_buffer, n_rollout_steps=a2c.n_steps)
        print(a2c.ep_info_buffer)
        if continue_training is False:
            break

        iteration += 1
        a2c._update_current_progress_remaining(a2c.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            fps = int(a2c.num_timesteps / (time.time() - a2c.start_time))
            logger.record("time/iterations", iteration, exclude="tensorboard")
            print(a2c.ep_info_buffer)
            if len(a2c.ep_info_buffer) > 0 and len(a2c.ep_info_buffer[0]) > 0:
                logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in a2c.ep_info_buffer]))
                logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in a2c.ep_info_buffer]))
            logger.record("time/fps", fps)
            logger.record("time/time_elapsed", int(time.time() - a2c.start_time), exclude="tensorboard")
            logger.record("time/total_timesteps", a2c.num_timesteps, exclude="tensorboard")
            logger.dump(step=a2c.num_timesteps)

        a2c.train()

    # obs = cat.reset()
    # print("obs_shape")
    # print(obs.shape)
    # for i in range(10000):
    #     obs,rew,done,info = cat.step([0]*cat.num_envs)
    # print(obs.shape)
    # print(rew.shape)

configure("log", format_strings='stdout,log,csv'.split(','))
#with ScopedConfigure("log", format_strings='stdout,log,csv'.split(',')):
main()
#res = ProcConcatVec([env_contr])
# env = aec_to_markov(env)
# env.reset()
# env.step([0]*len(env.agents))
