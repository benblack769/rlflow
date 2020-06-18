from basic_example import FCPolicy, DQNLearner
from rlflow.env_loop import run_loop
import gym
from rlflow.policy_delayer.base import NoUpdatePolicyDelayer
from rlflow.actors.single_agent_actor import StatelessActor
from rlflow.adders.transition_adder import TransitionAdder
from rlflow.selectors.uniform import UniformSampleScheme

def main():
    env = gym.make("Acrobot-v1")
    obs_size, = env.observation_space.shape
    act_size = env.action_space.n
    policy = FCPolicy(obs_size, act_size, 64)
    data_store_size = 128
    batch_size = 16
    run_loop(
        DQNLearner(policy, 0.01, 0.99),
        NoUpdatePolicyDelayer(),
        StatelessActor(policy),
        lambda: gym.make("Acrobot-v1"),
        lambda: TransitionAdder(env.observation_space, env.action_space),
        UniformSampleScheme(data_store_size),
        data_store_size,
        batch_size
    )
main()
