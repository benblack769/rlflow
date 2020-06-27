from rlflow.vector.aec_markov_wrapper import aec_to_markov
from rlflow.vector.dummy_vec_env import DummyVecEnv
from rlflow.vector.multiproc_vec import ProcConcatVec
from rlflow.vector.concat_vec_env import ConcatVecEnv
from rlflow.vector.markov_vector_wrapper import MarkovVectorEnv
from rlflow.vector.sb_vector_wrapper import VecEnvWrapper

from pettingzoo.mpe import simple_world_comm
from supersuit.aec_wrappers import pad_observations, pad_action_space
import gym

def main():
    def env_contr():
        #env = gym.make("CartPole-v0")#
        env = simple_world_comm.env()
        env = pad_observations(env)
        env = pad_action_space(env)
        markov_env = aec_to_markov(env)
        venv = MarkovVectorEnv(markov_env)
        return venv
    def nest_env_const():
        cat = ConcatVecEnv([env_contr,env_contr])
        return cat
    example_env = env_contr()
    num_envs = 3*example_env.num_envs
    cat = ProcConcatVec([nest_env_const,env_contr],example_env.observation_space, example_env.action_space, num_envs)
    cat = VecEnvWrapper(cat)
    obs = cat.reset()
    print("obs_shape")
    print(obs.shape)
    for i in range(10000):
        obs,rew,done,info = cat.step([0]*cat.num_envs)
    print(obs.shape)
    print(rew.shape)
main()
#res = ProcConcatVec([env_contr])
# env = aec_to_markov(env)
# env.reset()
# env.step([0]*len(env.agents))
