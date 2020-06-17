from gym.vector import SyncVectorEnv
import numpy as np

def run_loop(
        learner,
        policy_delayer,
        actor,
        environment_fn,
        adder_fn,
        replay_sampler
        ):

    example_env = environment_fn()
    n_envs = 8
    vec_env = SyncVectorEnv([environment_fn]*n_envs, example_env.observation_space, example_env.action_space)
    obs = vec_env.reset()
    dones = np.zeros(n_envs,dtype=np.bool)
    infos = [{} for _ in range(n_envs)]
    adders = [adder_fn() for _ in range(n_envs)]

    policy_delayer.set_policies(learner.policy, [actor.policy])
    for x in range(1000000):
        policy_delayer.learn_step()


        actions = actor.step(obs, dones, infos)

        obs, rews, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            adders[i].add(obs[i],rews[i],dones[i],infos[i])

        learner.learn_step()
