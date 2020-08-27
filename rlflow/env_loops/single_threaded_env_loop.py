import numpy as np
from rlflow.data_store.data_store import DataManager
from rlflow.selectors.fifo import FifoScheme
import multiprocessing as mp
import queue
from rlflow.adders.logger_adder import LoggerAdder
from rlflow.utils.shared_mem_pipe import SharedMemPipe, expand_example
from rlflow.selectors.priority_updater import priority_pipe_example, PriorityUpdater, NoUpdater
import time

def noop(x):
    return x

def run_loop(
        logger,
        learner_fn,
        policy_delayer,
        actor_fn,
        environment_fn,
        saver,
        vec_environment_fn,
        adder_fn,
        replay_sampler,
        data_store_size,
        batch_size,
        priority_updater=NoUpdater(),
        n_envs=1,
        log_frequency=100,
        max_learn_steps=2**100,
        log_callback=noop,
        ):


    example_env = environment_fn()

    vec_env = vec_environment_fn([environment_fn]*n_envs, example_env.observation_space, example_env.action_space)
    num_envs = vec_env.num_envs

    example_adder = adder_fn()
    dones = np.zeros(num_envs,dtype=np.bool)
    infos = [{} for _ in range(num_envs)]

    transition_example = example_adder.get_example_output()
    removal_scheme = FifoScheme()
    sample_scheme = replay_sampler
    new_entry_pipes = [SharedMemPipe(transition_example) for _ in range(num_envs)]

    priority_updater.set_data_pipe(SharedMemPipe(priority_pipe_example(batch_size)))

    data_manager = DataManager(new_entry_pipes, transition_example, removal_scheme, sample_scheme, data_store_size)

    adders = [adder_fn() for _ in range(num_envs)]
    log_adders = [LoggerAdder() for _ in range(num_envs)]
    for adder,entry_pipe in zip(adders, new_entry_pipes):
        adder.set_generate_callback(entry_pipe.store)

    for log_adder in log_adders:
        log_adder.set_generate_callback(lambda args: logger.record_type(*args))

    learner = learner_fn()
    actor = actor_fn()
    prev_time = time.time()/log_frequency

    obss = vec_env.reset()
    learn_steps = 0

    for train_step in range(1000000):
        policy_delayer.learn_step(learner.policy)
        policy_delayer.actor_step(actor.policy)

        for i in range(max(1,batch_size//num_envs)):
            actions = actor.step(obss, dones, infos)

            obss, rews, dones, infos = vec_env.step(actions)
            for i in range(len(obss)):
                obs,act,rew,done,info = obss[i], actions[i], rews[i], dones[i], infos[i]
                adders[i].add(obs,act,rew,done,info)
                log_adders[i].add(obs,act,rew,done,info)

            data_manager.receive_new_entries()

        learn_idxs, learn_weights, learn_batch = data_manager.sample_data(batch_size)
        if learn_batch is not None:
            learner.learn_step(learn_idxs, learn_weights, learn_batch)
            learn_steps += 1

            density_result = priority_updater.fetch_densities()
            if density_result is not None:
                ids, priorities = density_result
                data_manager.sample_scheme.update_priorities(ids, priorities)
                data_manager.removal_scheme.update_priorities(ids, priorities)

        if time.time()/log_frequency > prev_time:
            logger.dump()
            saver.checkpoint(learner.policy)
            log_callback(learner)
            prev_time += 1

        if learn_steps >= max_learn_steps:
            break
