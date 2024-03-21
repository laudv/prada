from .dataset import Dataset, Task, RegressionMixin, MulticlassMixin

import os
import numpy as np
import pandas as pd

# https://github.com/laudv/veritas/raw/c1808327b285facd112f651397552a54f90a6a4b/tests/data/img.npy
class Img(Dataset, RegressionMixin):
    dataset_name = "img.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, Img.dataset_name)
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            super().load_dataset()

    def read_from_img(self, fname):
        import imageio
        img = imageio.imread(fname)
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])

        df = pd.DataFrame(X, columns=["a0", "a1"])
        dfy = pd.Series(y)

        data_path = os.path.join(self.data_dir, Img.dataset_name)
        df.to_hdf(data_path, key='X', mode='w') 
        dfy.to_hdf(data_path, key='y', mode='a') 

class Chaahat(Dataset, MulticlassMixin):
    dataset_name = "chaahat_blocks4_"

    def __init__(self, **kwargs):
        super().__init__(Task.MULTICLASS, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            Xs, ys = [], []
            for xname, yname in [("trainX", "trainY"), ("testX", "testY")]:
                fnamex = os.path.join(self.data_dir, f"{Chaahat.dataset_name}{xname}")
                fnamey = os.path.join(self.data_dir, f"{Chaahat.dataset_name}{yname}")
                Xs.append(pd.read_hdf(fnamex))
                ys.append(pd.read_hdf(fnamey))
            self.X = pd.concat(Xs, ignore_index=True)
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.yscores = pd.concat(ys, ignore_index=True)
            self.y = self.yscores.idxmax(axis=1)
            super().load_dataset()
            self.num_classes = self.yscores.shape[1]

    def to_singletarget(self, target):
        assert self.are_X_y_set()
        y = self.yscores.iloc[:, target]
        sup = (Dataset, RegressionMixin)
        suffix = f"RegTarget{target}"
        task = Task.REGRESSION
        name = self.name() + suffix

        cls = type(name, sup, {})
        d = cls(task, nfolds=self.nfolds, seed=self.seed)
        d.target = target
        d.multitarget = self

        # Simulate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        d.X = self.X
        d.y = y
        d.perm = self.perm

        return d
