from gym.vector import SyncVectorEnv
import numpy as np
from rlflow.data_store.data_store import DataStore, DataManager, DataSaver, BatchStore
from rlflow.selectors.fifo import FifoScheme
import multiprocessing as mp
import queue
from rlflow.adders.logger_adder import LoggerAdder


def run_loop(
        logger,
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

    transition_example = example_adder.get_example_output()
    data_store = DataStore(transition_example, data_store_size)
    removal_scheme = FifoScheme()
    empty_entries = mp.Queue(n_envs*2)
    new_entries = mp.Queue(n_envs*2)
    batch_samples = mp.Queue(2)
    data_manager = DataManager(removal_scheme, replay_sampler, data_store_size, empty_entries, new_entries, batch_samples, batch_size)

    for adder in adders:
        saver = DataSaver(data_store, empty_entries, new_entries)
        adder.set_generate_callback(saver.save_data)

    env_log_queue = mp.Queue()
    logger_adders = [LoggerAdder() for _ in range(n_envs)]
    for adder in logger_adders:
        adder.set_generate_callback(env_log_queue.put)

    batch_generator = BatchStore(batch_size, transition_example)

    policy_delayer.set_policies(learner.policy, [actor.policy])
    for train_step in range(1000000):
        policy_delayer.learn_step()

        data_manager.update()

        actions = actor.step(obs, dones, infos)

        obs, rews, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            adders[i].add(obs[i],actions[i],rews[i],dones[i],infos[i])
            logger_adders[i].add(obs[i],actions[i],rews[i],dones[i],infos[i])

        try:
            batch_idxs = batch_samples.get_nowait()
        except queue.Empty:
            continue
        batch_generator.store_batch(data_store, batch_idxs)
        learn_batch = batch_generator.get_batch()
        learner.learn_step(learn_batch)
        batch_generator.batch_copied()

        while not env_log_queue.empty():
            logger.record(*env_log_queue.get_nowait())

        if train_step % 1000 == 0:
            logger.dump()
