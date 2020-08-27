import multiprocessing as mp
from rlflow.utils.shared_array import SharedArray
import numpy as np
import ctypes
import multiprocessing as mp

def get_learn_policy_info_mp(learner_policy_fn,queue):
    learner_policy = learner_policy_fn()
    policy_weights = learner_policy.get_params()
    param_info = []
    for weight in policy_weights:
        assert isinstance(weight, np.ndarray)
        param_info.append((weight.shape, weight.dtype))
    queue.put(param_info)


def get_learn_policy_info(learner_policy_fn):
    queue = mp.Queue()
    proc = mp.Process(target=get_learn_policy_info_mp, args=(learner_policy_fn, queue))
    proc.start()
    proc.join()
    result = queue.get()
    return result

class OccasionalUpdate:
    def __init__(self, steps_to_update, learner_policy_fn):
        self.steps_to_update = steps_to_update
        self.train_steps = 0
        self.lock = mp.Lock()
        self.stored_version = mp.Value(ctypes.c_long, lock=False)
        self.my_version = -10
        self.params = []
        for shape, dtype in get_learn_policy_info(learner_policy_fn):
            self.params.append(SharedArray(shape, dtype))

    def learn_step(self, learner_policy):
        if self.train_steps % self.steps_to_update == 0:
            with self.lock:
                policy_weights = learner_policy.get_params()
                for val, store in zip(policy_weights, self.params):
                    store.np_arr[:] = val

                self.stored_version.value += 1

    def actor_step(self, actor_policy):
        with self.lock:
            cur_version = int(self.stored_version.value)
            if self.my_version < cur_version:
                params_to_set = [param.np_arr for param in self.params]
                actor_policy.set_params(params_to_set)

                self.my_version = cur_version
