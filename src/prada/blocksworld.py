import os
import pickle
import numpy as np
import pandas as pd

from .dataset import DATA_DIR, DTYPE
from .dataset import MultiTargetRegression

_BLOCKSWORLD_DIR = "blocksworld"
BLOCKSWORLD_DATASETS = []

def _unpickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def _create_blocksworld(nblocks, nnsize, cost_aware):
    cost_aware_short = "CA" if cost_aware else "CI"
    fname = f"blocksworld_{nblocks}_{nblocks-1}_{nnsize}_{nnsize}_{cost_aware_short}_app"
    fname_states = f"{fname}_states.pkl"
    fname_actions = f"{fname}_actions.pkl"

    fpath_states = os.path.join(DATA_DIR, _BLOCKSWORLD_DIR, fname_states)
    fpath_actions = os.path.join(DATA_DIR, _BLOCKSWORLD_DIR, fname_actions)

    class_name = f"Blocksworld_{nblocks}_{nnsize}_{cost_aware_short}"
    cls = type(class_name, (MultiTargetRegression,), {})

    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(nblocks+1, *args, **kwargs)
        self.fpath_states = fpath_states
        self.fpath_actions = fpath_actions

    def load_dataset(self):
        if self.X is None or self.y is None:
            X = np.array(_unpickle(self.fpath_states), dtype=DTYPE)
            self.X = pd.DataFrame(X, columns=[f"s{i}" for i in range(X.shape[1])])
            
            y = np.array(_unpickle(self.fpath_actions), dtype=DTYPE)
            assert y.shape[1] == self.num_targets
            self.y = pd.DataFrame(y, columns=[f"a{i}" for i in range(y.shape[1])])

            super(cls, self).load_dataset()

    cls.__init__ = __init__
    cls.load_dataset = load_dataset

    BLOCKSWORLD_DATASETS.append(cls)

    return cls

for cost_aware in [False, True]:
    for nblocks in [4, 6, 8]:
        for nnsize in [16, 32, 64]:
            if nnsize == 16 and cost_aware:
                continue
            if nblocks > 4 and nnsize != 64:
                continue
            cls = _create_blocksworld(nblocks, nnsize, cost_aware)
            globals()[cls.__name__] = cls



