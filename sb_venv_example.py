from rlflow.vector.aec_markov_wrapper import aec_to_markov
from rlflow.vector.dummy_vec_env import DummyVecEnv
from rlflow.vector.multiproc_vec import ProcConcatVec
from rlflow.vector.concat_vec_env import ConcatVecEnv
from rlflow.vector.markov_vector_wrapper import MarkovVectorEnv
from rlflow.vector.sb_vector_wrapper import VecEnvWrapper

from pettingzoo.sisl import multiwalker_v0
from supersuit.aec_wrappers import pad_observations, pad_action_space
from stable_baselines3 import A2C
#from stable_baselines.acktr import MlpPolicy
import gym

def main():
    def env_contr():
        #env = gym.make("CartPole-v0")#
        env = multiwalker_v0.env()
        env = pad_observations(env)
        env = pad_action_space(env)
        markov_env = aec_to_markov(env)
        venv = MarkovVectorEnv(markov_env)
        return venv
    envs_per_proc = 4
    n_procs = 4
    def nest_env_const():
        cat = ConcatVecEnv([env_contr]*envs_per_proc)
        return cat
    example_env = env_contr()
    num_envs = n_procs*envs_per_proc*example_env.num_envs
    #cat = ProcConcatVec([nest_env_const]*n_procs,example_env.observation_space, example_env.action_space, num_envs)
    cat = ConcatVecEnv([nest_env_const]*n_procs)#,example_env.observation_space, example_env.action_space, num_envs)
    cat = VecEnvWrapper(cat)
    policy = "MlpPolicy"
    a2c = A2C(policy,cat)
    print(type(a2c.env))
    a2c.learn(1000000,log_interval=2,eval_freq=100000000000000)

    # obs = cat.reset()
    # print("obs_shape")
    # print(obs.shape)
    # for i in range(10000):
    #     obs,rew,done,info = cat.step([0]*cat.num_envs)
    # print(obs.shape)
    # print(rew.shape)
main()
#res = ProcConcatVec([env_contr])
# env = aec_to_markov(env)
# env.reset()
# env.step([0]*len(env.agents))
