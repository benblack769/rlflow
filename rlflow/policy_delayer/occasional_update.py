import multiprocessing as mp
from rlflow.utils.shared_array import SharedArray
import numpy as np
import ctypes

class OccasionalUpdate:
    def __init__(self, steps_to_update, learner_policy):
        self.steps_to_update = steps_to_update
        self.train_steps = 0
        self.lock = mp.Lock()
        self.stored_version = mp.Value(ctypes.c_long, lock=False)
        self.my_version = -10
        policy_weights = learner_policy.get_params()
        self.params = []
        for weight in policy_weights:
            assert isinstance(weight, np.ndarray)
            self.params.append(SharedArray(weight.shape, weight.dtype))

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
