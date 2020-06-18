from gym.vector import SyncVectorEnv
import numpy as np
from rlflow.data_store.data_store import DataStore, DataManager, DataSaver, BatchStore
from rlflow.selectors.fifo import FifoScheme
import multiprocessing as mp
import queue

# def get_transition_example(env, adder_fn):
#     info = {}
#     def gen_callback(trans):
#         info['trans'] = trans
#     adder = adder_fn()
#     adder.set_generate_callback(gen_callback)

def run_loop(
        learner,
        policy_delayer,
        actor,
        environment_fn,
        adder_fn,
        replay_sampler,
        data_store_size,
        batch_size
        ):

    example_env = environment_fn()
    example_adder = adder_fn()
    n_envs = 8
    vec_env = SyncVectorEnv([environment_fn]*n_envs, example_env.observation_space, example_env.action_space)
    obs = vec_env.reset()
    dones = np.zeros(n_envs,dtype=np.bool)
    infos = [{} for _ in range(n_envs)]
    adders = [adder_fn() for _ in range(n_envs)]

    transition_example = example_adder.get_transition_example()
    data_store = DataStore(transition_example, data_store_size)
    removal_scheme = FifoScheme()
    empty_entries = mp.Queue()
    new_entries = mp.Queue()
    batch_samples = mp.Queue()
    data_manager = DataManager(removal_scheme, replay_sampler, data_store_size, empty_entries, new_entries, batch_samples, batch_size)

    for adder in adders:
        saver = DataSaver(data_store, empty_entries, new_entries)
        adder.set_generate_callback(saver.save_data)

    batch_generator = BatchStore(batch_size, transition_example)

    policy_delayer.set_policies(learner.policy, [actor.policy])
    for x in range(1000000):
        policy_delayer.learn_step()

        data_manager.update()

        actions = actor.step(obs, dones, infos)

        obs, rews, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            adders[i].add(obs[i],actions[i],rews[i],dones[i],infos[i])

        try:
            batch_idxs = batch_samples.get_nowait()
        except queue.Empty:
            continue

        batch_generator.store_batch(data_store, batch_idxs)
        learn_batch = batch_generator.get_batch()
        learner.learn_step(learn_batch)
        batch_generator.batch_copied()
