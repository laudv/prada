from .dataset import Dataset, Task, RegressionMixin

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
            #self.threshold = np.median(self.yreal)
            #self.y = self.yreal >= self.threshold
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
