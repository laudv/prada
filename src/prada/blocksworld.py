import os
import pickle
import numpy as np
import pandas as pd

from .dataset import DATA_DIR
from .dataset import Dataset, MulticlassMixin, Task

COST_AWARE_DIR = "blocksworld/cost_aware"
COST_IGNORE_DIR = "blocksworld/cost_ignore"

def _unpickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def _create_blocksworld(nblocks, nnsize, cost_aware):
    cost_aware_str = "cost_aware" if cost_aware else "cost_ignore"
    cost_aware_short = "CA" if cost_aware else "CI"
    cost_aware_path = COST_AWARE_DIR if cost_aware else COST_IGNORE_DIR
    directory = f"blocksworld_{nblocks}_{nblocks-1}"
    fname = f"blocksworld_{nblocks}_{nblocks-1}_{nnsize}_{nnsize}_{cost_aware_str}_app"
    fname_states = f"{fname}_states.pkl"
    fname_actions = f"{fname}_actions.pkl"

    fpath_states = os.path.join(DATA_DIR, cost_aware_path, directory, fname_states)
    fpath_actions = os.path.join(DATA_DIR, cost_aware_path, directory, fname_actions)

    class_name = f"Blocksworld_{nblocks}_{nnsize}_{cost_aware_short}"
    sup = (Dataset, MulticlassMixin)
    task = Task.MULTICLASS
    cls = type(class_name, sup, {})

    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(task, *args, **kwargs)
        self.fpath_states = fpath_states
        self.fpath_actions = fpath_actions

    def load_dataset(self):
        if self.X is None or self.y is None:
            X = np.array(_unpickle(self.fpath_states), dtype=np.float64)
            self.X = pd.DataFrame(X, columns=[f"s{i}" for i in range(X.shape[1])])
            
            y = np.array(_unpickle(self.fpath_actions), dtype=int)
            self.num_classes = len(np.unique(y))
            self.y = pd.Series(y)

            super(cls, self).load_dataset()

    cls.__init__ = __init__
    cls.load_dataset = load_dataset

    return cls

for cost_aware in [False, True]:
    for nblocks in [4, 6, 8]:
        for nnsize in [32, 64]:
            cls = _create_blocksworld(nblocks, nnsize, cost_aware)
            globals()[cls.__name__] = cls


