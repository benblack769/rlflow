import numpy as np
from rlflow.data_store.data_store import DataManager
from rlflow.selectors.fifo import FifoScheme
import multiprocessing as mp
import queue
from rlflow.adders.logger_adder import LoggerAdder
from rlflow.utils.shared_mem_pipe import SharedMemPipe, expand_example
from rlflow.selectors.priority_updater import priority_pipe_example, PriorityUpdater, NoUpdater
import time
from ..utils.shared_array import SharedArray
from ..utils.space_wrapper import SpaceWrapper
import multiprocessing as mp
import numpy as np
import torch
import traceback
import random
from rlflow.vector import SingleVecEnv


def noop(x):
    return x


start_time = time.time()
tot_time = 0

class BatchActor:
    def __init__(self, policy, num_envs, batch_size):
        assert batch_size < num_envs
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.policy = policy
        self.queued_obss = {}
        self.completed_results = {}
        self.batched_results = []

    def step_async(self, obs, idx):
        assert idx not in self.queued_obss
        self.queued_obss[idx] = obs
        if len(self.queued_obss) == self.batch_size:
            items = list(self.queued_obss.items())
            idxs = [idx for idx,obs in items]
            obss = [obs for idx,obs in items]
            obss = np.stack(obss)
            self.batched_results.append((idxs, self.policy(obss)))
            self.queued_obss = {}

    def step_wait(self, idx):
        if idx in self.completed_results:
            action = self.completed_results[idx]
            del self.completed_results[idx]
            return action
        else:
            assert self.batched_results, "querreied action that was not assigned observation"
            idxs, actions = self.batched_results.pop(0)
            actions = actions.cpu().detach().numpy()

            for idx, action in zip(idxs, actions):
                self.completed_results[idx] = action

            return self.step_wait(idx)

def async_env_loop(env_constr, env_ids, instr_pipe, shared_readys, shared_obs, shared_rews, shared_dones):
    try:
        envs = {id: env_constr() for id in env_ids}
        envs_per_id = list(envs.values())[0].num_envs

        while True:
            instr,id,actions = instr_pipe.recv()
            if instr == "reset":
                obs = envs[id].reset()
                env_start_idx = id*envs_per_id
                env_end_idx = (id+1)*envs_per_id
                shared_obs.np_arr[env_start_idx:env_end_idx] = obs
                shared_dones.np_arr[env_start_idx:env_end_idx] = False
                shared_rews.np_arr[env_start_idx:env_end_idx] = 0.
                shared_readys.np_arr[id] = True
            elif instr == "step":
                env_start_idx = id*envs_per_id
                env_end_idx = (id+1)*envs_per_id
                observations, rewards, dones, infos = envs[id].step(actions)
                shared_obs.np_arr[env_start_idx:env_end_idx] = observations
                shared_dones.np_arr[env_start_idx:env_end_idx] = dones
                shared_rews.np_arr[env_start_idx:env_end_idx] = rewards
                shared_readys.np_arr[id] = True
            elif instr == "terminate":
                return
            #pipe.send(comp_infos)
    except BaseException as e:
        tb = traceback.format_exc()
        instr_pipe.send((e,tb))

def alloc_envs_to_cpus(num_fns, max_num_cpus):
    envs_per_cpu = (num_fns+max_num_cpus-1)// max_num_cpus
    alloced_num_cpus = (num_fns+envs_per_cpu-1) // envs_per_cpu

    env_cpu_div = []
    num_envs_alloced = 0
    while num_envs_alloced < num_fns:
        start_idx = num_envs_alloced
        end_idx = min(num_fns, start_idx+envs_per_cpu)
        env_cpu_div.append(list(range(start_idx,end_idx)))
        num_envs_alloced = end_idx

    return env_cpu_div


class AsyncMultiEnv:
    def __init__(self, env_fn, num_ids, num_procs=None):
        env = env_fn()
        envs_per_id = env.num_envs
        # if getattr(env, "num_envs", None) is None:
        #     env_fn = lambda: SingleVecEnv(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        if num_procs is None:
            num_procs = mp.cpu_count()
        self.num_ids = num_ids
        self.envs_per_id = envs_per_id
        num_procs = min(num_ids, num_procs)

        self.num_envs = num_envs = envs_per_id * num_ids
        self.shared_obs = SharedArray((num_envs,)+self.observation_space.shape, dtype=self.observation_space.dtype)
        self.shared_rews = SharedArray((num_envs,), dtype=np.float32)
        self.shared_dones = SharedArray((num_envs,), dtype=np.uint8)
        self.shared_readys = SharedArray((num_ids,), dtype=np.uint8)
        self.collected = [False]*num_ids

        self.env_cpu_div = env_cpu_div = alloc_envs_to_cpus(num_ids, num_procs)

        pipes = []
        procs = []
        for i in range(num_procs):
            inpt,outpt = mp.Pipe()
            proc = mp.Process(target=async_env_loop, args=(env_fn, env_cpu_div[i], outpt, self.shared_readys, self.shared_obs, self.shared_rews, self.shared_dones))
            proc.start()
            pipes.append(inpt)
            procs.append(proc)
        idx_cpu = {id:cpu for cpu in range(num_procs) for id in env_cpu_div[cpu]}
        self.pipe_map = {id:pipes[idx_cpu[id]] for id in range(num_ids)}
        self.pipes = pipes
        self.procs = procs

    def reset_all_async(self):
        for id in range(self.num_ids):
            self.reset_async(id)

    def check_error(self, id):
        pipe = self.pipe_map[id]
        if pipe.poll():
            e, tb = pipe.recv()
            print(tb)
            raise e

    def reset_async_id(self, id):
        self.check_error(id)
        pipe = self.pipe_map[id]
        self.collected[id] = False
        self.shared_readys.np_arr[id] = False
        pipe.send(("reset",id,None))

    def ready(self, id):
        return bool(self.shared_readys.np_arr[id])

    def collect_wait_id(self, id):
        assert not self.collected[id]
        self.collected[id] = True
        while not self.shared_readys.np_arr[id]:
            pass
        #self.check_error(id)
        env_start_idx = id*self.envs_per_id
        env_end_idx = (id+1)*self.envs_per_id
        observations = self.shared_obs.np_arr[env_start_idx:env_end_idx]
        dones = self.shared_dones.np_arr[env_start_idx:env_end_idx]
        rewards = self.shared_rews.np_arr[env_start_idx:env_end_idx]
        return observations, rewards, dones

    def step_async_id(self, id, actions):
        self.check_error(id)
        self.collected[id] = False
        pipe = self.pipe_map[id]
        self.shared_readys.np_arr[id] = False
        pipe.send(("step",id,actions))

    def __del__(self):
        for pipe in self.pipes:
            try:
                pipe.send("terminate")
            except BrokenPipeError:
                pass
        for proc in self.procs:
            proc.join()

class AsyncEnv:
    def __init__(self, async_multi_env):
        self.async_multi_env = async_multi_env
        self.num_ids = async_multi_env.num_ids
        self.envs_per_id = async_multi_env.envs_per_id
        self.id_states = {id:("init",None) for id in range(self.num_ids)}
        self.num_envs = async_multi_env.num_envs

    def ready(self, idx):
        id = idx // self.envs_per_id
        return self.async_multi_env(id)

    def reset_all_async(self):
        for idx in range(self.num_envs):
            self.reset_async(idx)

    def reset_async(self, idx):
        id = idx // self.envs_per_id
        if self.id_states[id][0] != "reset":
            self.async_multi_env.reset_async_id(id)
            self.id_states[id] = ("reset", None)

    def step_async(self, idx, action):
        id = idx // self.envs_per_id
        offset = idx - id * self.envs_per_id
        #assert self.id_states[id][0] != "waiting"
        if self.id_states[id][0] == "stepping":
            actions = self.id_states[id][1]
            actions[offset] = action
            if len(actions) == self.envs_per_id:
                action_list = [actions[i] for i in range(self.envs_per_id)]
                self.async_multi_env.step_async_id(id, action_list)
                self.id_states[id] = ("waiting", None)
        else:
            self.id_states[id] = ("stepping", {})
            self.step_async(idx, action)

    def collect_wait_id(self, idx):
        id = idx // self.envs_per_id
        offset = idx - id * self.envs_per_id
        async_type = self.id_states[id][0]
        assert async_type == "waiting" or async_type == "reset"
        wait_data = self.id_states[id][1]
        if wait_data is None:
            obss, rews, dones = self.async_multi_env.collect_wait_id(id)
            new_wait_data = [[obss[i], rews[i], dones[i]] for i in range(self.envs_per_id)]
            self.id_states[id] = ("waiting", new_wait_data)
            return new_wait_data[offset]
        else:
            return wait_data[offset]



def run_loop(
        logger,
        learner_fn,
        policy_delayer,
        policy_fn,
        environment_fn,
        saver,
        adder_fn,
        replay_sampler,
        data_store_size,
        batch_size,
        priority_updater=NoUpdater(),
        act_steps_until_learn=None,
        log_frequency=100,
        max_learn_steps=2**100,
        num_env_ids=1,
        num_cpus=1,
        log_callback=noop,
        ):

    example_env = environment_fn()
    multi_env = AsyncEnv(AsyncMultiEnv(environment_fn, num_env_ids, num_cpus))
    #vec_env = vec_environment_fn([environment_fn]*n_envs, example_env.observation_space, example_env.action_space)
    num_envs = multi_env.num_envs
    ids_per_env = num_envs // num_env_ids
    #actor_delay = max(min(3*num_env_ids // num_envs, num_env_ids-2)

    example_adder = adder_fn()
    dones = np.zeros(num_envs,dtype=np.uint8)
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

    if act_steps_until_learn is None:
        act_steps_until_learn = data_store_size//2

    learner = learner_fn()
    #actor = actor_fn()
    exec_batch_size = min(8,max(num_envs//4,1))
    actor = BatchActor(policy_fn(), num_envs, exec_batch_size)
    prev_time = time.time()/log_frequency

    multi_env.reset_all_async()
    learn_steps = 0
    total_act_steps = 0

    env_actor_delay = (num_env_ids//2) * ids_per_env
    actions_taken = [None]*num_envs
    action_initiated = [False]*num_envs
    # for env_idx in range(env_actor_delay):
    #     obs, rew, done = multi_env.collect_wait_id(env_idx)
    #     actor.step_async(obs, env_idx)
    #     obs,act,rew,done,info = obs, None, 0, False, {}
    #     adders[env_idx].add(obs,act,rew,done,info)
    #     log_adders[env_idx].add(obs,act,rew,done,info)
    torch.set_num_interop_threads(4)

    global tot_time,start_time
    for train_step in range(1000000):
        policy_delayer.learn_step(learner.policy)
        policy_delayer.actor_step(actor.policy)

        cur_act_steps = max(1,batch_size//num_envs)
        for i in range(cur_act_steps):
            for env_idx in range(num_envs):
                obs, rew, done = multi_env.collect_wait_id(env_idx)

                start = time.time()
                actor.step_async(obs, env_idx)
                end = time.time()
                action_initiated[env_idx] = True
                act = actions_taken[env_idx]
                info = {}
                adders[env_idx].add(obs,act,rew,done,info)
                log_adders[env_idx].add(obs,act,rew,done,info)

                act_idx = (env_idx + env_actor_delay) % num_envs
                if action_initiated[act_idx]:
                    action = actor.step_wait(act_idx)
                    actions_taken[act_idx] = action
                    multi_env.step_async(act_idx, action)
                    tot_time += end - start
                    if random.random() < 0.001:
                        print(tot_time / (time.time() - start_time))
                        tot_time = 0
                        start_time = time.time()

            data_manager.receive_new_entries()

        total_act_steps += cur_act_steps * num_envs

        if total_act_steps >= act_steps_until_learn:
            print("learning")
            learn_idxs, learn_weights, learn_batch = data_manager.sample_data(batch_size)
            if learn_batch is not None:
                learner.learn_step(learn_idxs, learn_batch, learn_weights)
                learn_steps += 1

                density_result = priority_updater.fetch_densities()
                if density_result is not None:
                    ids, priorities = density_result
                    data_manager.sample_scheme.update_priorities(ids, priorities)
                    data_manager.removal_scheme.update_priorities(ids, priorities)

            if learn_steps >= max_learn_steps:
                break

        if time.time()/log_frequency > prev_time:
            logger.dump()
            logger.record("total_act_steps",total_act_steps)
            saver.checkpoint(learner.policy)
            log_callback(learner)
            prev_time += 1
