from gym.vector import SyncVectorEnv
import numpy as np
from rlflow.data_store.data_store import DataManager
from rlflow.selectors.fifo import FifoScheme
import multiprocessing as mp
import queue
import traceback
import time
from rlflow.utils.shared_mem_pipe import SharedMemPipe, expand_example
from rlflow.adders.logger_adder import LoggerAdder
from rlflow.selectors.priority_updater import priority_pipe_example, PriorityUpdater, NoUpdater
from rlflow.vector import MakeCPUAsyncConstructor

def run_batch_generator(term_event, transition_example, removal_scheme, sample_scheme, max_entries, batch_store, new_entries_pipes, priority_updater, batch_size, logger):
    data_manager = DataManager(new_entries_pipes, transition_example, removal_scheme, sample_scheme, max_entries)

    while not term_event.is_set():
        # load data from actors
        data_manager.receive_new_entries()

        # load priority data from learner
        density_result = priority_updater.fetch_densities()
        if density_result is not None:
            ids, priorities = density_result
            data_manager.sample_scheme.update_priorities(ids, priorities)
            data_manager.removal_scheme.update_priorities(ids, priorities)

        # store batched samples for learner
        if batch_store.can_store():
            batch_idxs, batch_weights, batch_data = data_manager.sample_data(batch_size)
            if batch_data is not None:
                store_data = [batch_idxs, batch_weights]+list(batch_data)
                batch_store.store(store_data)

def run_actor_except(term_event, *args):
    try:
        run_actor_loop(term_event, *args)
    except Exception as e:
        traceback.print_exc()
        print(e)
        term_event.set()
        raise e

def run_worker_except(term_event, *args):
    try:
        run_batch_generator(term_event, *args)
    except Exception as e:
        term_event.set()
        traceback.print_exc()

def run_actor_loop(terminate_event, start_learn_event, actor_fn, adder_fn, log_adder_fn, new_entry_pipes, num_cpus, num_env_ids, policy_delayer, env_fn, logger_pipe, data_store_size, act_steps_until_learn):
    example_env = env_fn()

    vec_env = MakeCPUAsyncConstructor(num_cpus)([env_fn]*num_env_ids, example_env.observation_space, example_env.action_space)
    del example_env
    num_envs = vec_env.num_envs

    if act_steps_until_learn is None:
        act_steps_until_learn = data_store_size / num_envs

    actor = actor_fn()

    tot_time = 0
    start_time = time.time()
    adders = [adder_fn() for _ in range(num_envs)]
    log_adders = [log_adder_fn() for _ in range(num_envs)]

    assert len(adders) == len(new_entry_pipes)
    for adder,entry_pipe in zip(adders, new_entry_pipes):
        adder.set_generate_callback(entry_pipe.store)

    for log_adder in log_adders:
        log_adder.set_generate_callback(logger_pipe.put)

    dones = np.zeros(num_envs,dtype=np.uint8)
    infos = [{} for _ in range(num_envs)]
    obss = vec_env.reset()

    start_tot = time.time()
    tot_time = 0
    for act_step in range(1000000):
        if terminate_event.is_set():
            break
        policy_delayer.actor_step(actor.policy)

        if act_step * num_envs < act_steps_until_learn:
            actions = [vec_env.action_space.sample() for _ in range(num_envs)]
        else:
            start_learn_event.set()

        actions, actor_info = actor.step(obss, dones, infos)

        obss, rews, dones, infos = vec_env.step(actions)

        for i in range(len(obss)):
            obs,act,rew,done,info,act_info = obss[i], actions[i], rews[i], dones[i], infos[i], actor_info[i]
            adders[i].add(obs,act,rew,done,info,act_info)
            log_adders[i].add(obs,act,rew,done,info,act_info)


def noop(x):
    return x

def run_loop(
        logger,
        learner_fn,
        policy_delayer,
        actor_fn,
        environment_fn,
        saver,
        adder_fn,
        replay_sampler,
        data_store_size,
        batch_size,
        priority_updater=NoUpdater(),
        num_env_ids=4,
        log_frequency=100,
        act_steps_until_learn=None,
        max_learn_steps=2**100,
        log_callback=noop,
        num_cpus=0,
        num_actors=1,
        ):

    if act_steps_until_learn is None:
        act_steps_until_learn = data_store_size//2

    terminate_event = mp.Event()
    start_learn_event = mp.Event()

    example_adder = adder_fn()

    example_env = environment_fn()
    envs_per_env = getattr(example_env, "num_envs", 1)
    del example_env
    num_envs = num_env_ids*envs_per_env

    transition_example = example_adder.get_example_output()
    removal_scheme = FifoScheme()
    sample_scheme = replay_sampler

    env_log_queue = mp.Queue()

    priority_updater.set_data_pipe(SharedMemPipe(priority_pipe_example(batch_size)))

    batch_store = SharedMemPipe([np.empty(batch_size,dtype=np.int64), np.empty(batch_size,dtype=np.float32)]+expand_example(transition_example, batch_size))

    new_entry_pipes = [SharedMemPipe(transition_example) for _ in range(num_envs)]
    logger_adder_fn = LoggerAdder

    batch_proc = mp.Process(target=run_worker_except,args=(terminate_event, transition_example, removal_scheme, sample_scheme, data_store_size, batch_store, new_entry_pipes, priority_updater, batch_size, env_log_queue))
    procs = [batch_proc]
    assert num_envs % num_env_ids == 0
    envs_per_act = num_envs // num_actors
    for aidx in range(num_actors):
        sidx = aidx * envs_per_act
        eidx = (aidx+1) * envs_per_act
        actor_proc = mp.Process(target=run_actor_except,args=(terminate_event, start_learn_event, actor_fn, adder_fn, logger_adder_fn, new_entry_pipes[sidx:eidx], num_cpus//num_actors, envs_per_act, policy_delayer, environment_fn, env_log_queue, data_store_size, act_steps_until_learn//num_actors))
        procs.append(actor_proc)

    for proc in procs:
        proc.start()

    try:
        learner = learner_fn()
        prev_time = time.time()/log_frequency

        learn_steps = 0
        cur_learn_steps = 0
        for train_step in range(2**100):
            if terminate_event.is_set():
                break
            if start_learn_event.is_set():
                policy_delayer.learn_step(learner.policy)

                learn_batch = batch_store.get()
                if learn_batch is None:
                    continue

                ids = learn_batch[0]
                weights = learn_batch[1]
                transition_data = learn_batch[2:]
                learner.learn_step(ids, transition_data, weights)


                if learn_steps >= max_learn_steps:
                    break

                cur_learn_steps += 1
                learn_steps += 1

            while not env_log_queue.empty():
                logger.record_type(*env_log_queue.get_nowait())

            if time.time()/log_frequency > prev_time:
                logger.record_sum("learn_steps", cur_learn_steps*batch_size)
                cur_learn_steps = 0
                logger.dump()
                saver.checkpoint(learner.policy)
                prev_time += 1
                log_callback(learner)

    finally:
        terminate_event.set()
        for proc in procs:
            proc.join(0.2)
            proc.terminate()
