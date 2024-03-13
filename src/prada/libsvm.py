import os
import pandas as pd

from .dataset import Dataset, Task, BinaryMixin

# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

class Ijcnn1(Dataset, BinaryMixin):
    dataset_name = "ijcnn1.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.BINARY, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            ijcnn1_data_path = os.path.join(self.data_dir, Ijcnn1.dataset_name)

            # we choose new train/test subsets in 'train_and_test_set'
            Xtrain = pd.read_hdf(ijcnn1_data_path, "Xtrain")
            Xtest = pd.read_hdf(ijcnn1_data_path, "Xtest")
            ytrain = pd.read_hdf(ijcnn1_data_path, "ytrain")
            ytest = pd.read_hdf(ijcnn1_data_path, "ytest")

            self.X = pd.concat((Xtrain, Xtest), axis=0, ignore_index=True)
            self.y = pd.concat((ytrain, ytest), axis=0, ignore_index=True)
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]

            self.minmax_normalize()
            super().load_dataset()

class Webspam(Dataset, BinaryMixin):
    dataset_name = "webspam_wc_normalized_unigram.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.BINARY, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, Webspam.dataset_name)
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            self.minmax_normalize()
            super().load_dataset()
