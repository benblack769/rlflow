import numpy as np
import os
import re
import shutil
import math
import warnings

def save_arrays(folder_name, arrays):
    os.makedirs(folder_name, exist_ok=True)
    for i,arr in enumerate(arrays):
        np.save(os.path.join(folder_name, f"{i:06}.npy"), arr, allow_pickle=False)

def save_policy(folder_name, policy):
    save_arrays(folder_name, policy.get_params())

def load_folder(folder_name, policy):
    fnames = list(os.listdir(folder_name))
    fnames.sort()
    npy_list = [np.load(fname) for fname in fnames if re.fullmatch("[0-9]{6}.py",fname)]
    policy_params = policy.get_params()
    for p1,p2 in zip(npy_list, policy_params):
        assert p1.shape == p2.shape, "tried to load policy from bad save data"
    policy.set_params(npy_list)

def load_latest(base_folder, policy):
    latest_fname = os.path.join(base_folder, "latest.txt")
    if os.path.exists(latest_fname):
        latest_num = open(latest_fname).read().strip()
        fold_name = os.path.join(base_folder, latest_num)
        load_folder(fold_name, policy)
    else:
        warnings.warn(f"cannot load policy from latest, '{latest_fname}' file missing")

class Saver:
    def __init__(self, base_folder, keep_history=True, max_history_save=50, decay_rate=1.5):
        os.makedirs(base_folder, exist_ok=True)
        self.base_folder = base_folder
        self.decay_rate = decay_rate
        self.max_history_save = max_history_save
        latest_fname = os.path.join(base_folder, "latest.txt")
        self.latest_fname = latest_fname
        if not keep_history:
            if os.path.exists(latest_fname):
                os.remove(latest_fname)
            for fname in os.listdir(base_folder):
                if re.fullmatch("[0-9]+",fname):
                    shutil.rmtree(os.path.join(base_folder,fname))
        if os.path.exists(latest_fname):
            self.next_checkpoint = 1 + int(open(latest_fname).read().strip())
        else:
            self.next_checkpoint = 0

        self.start_checkpoint = self.next_checkpoint
        # don't delete old checkpoints
        self.latest_checkpoints = []
        self.decayed_checkpoints = []

    def checkpoint(self, policy):
        checkpoint_str = str(self.next_checkpoint).zfill(6)
        save_policy(os.path.join(self.base_folder, checkpoint_str), policy)
        self.latest_checkpoints.append(self.next_checkpoint)
        open(self.latest_fname,'w').write(checkpoint_str+"\n")
        self._clean_checkpoints()
        self.next_checkpoint += 1

    def _clean_checkpoints(self):
        if len(self.latest_checkpoints) > self.max_history_save:
            if not self.decayed_checkpoints or int(math.log(self.latest_checkpoints[0] - self.start_checkpoint + 1, self.decay_rate)) > int(math.log(self.decayed_checkpoints[-1] - self.start_checkpoint + 1, self.decay_rate)):
                self.decayed_checkpoints.append(self.latest_checkpoints[0])
            else:
                shutil.rmtree(os.path.join(self.base_folder, str(self.latest_checkpoints[0]).zfill(6)))
            self.latest_checkpoints.pop(0)
