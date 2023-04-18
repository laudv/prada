from .dataset import *
from .regression import *

class CalhouseClf(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("calhouse", data_id=537)
            self.threshold = np.median(self.y)
            self.y = self.y > self.threshold
            super().load_dataset()

class AllstateClf(Dataset):
    dataset_name = "allstate.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            allstate_data_path = os.path.join(self.data_dir, Allstate.dataset_name)
            data = pd.read_hdf(allstate_data_path)
            self.X = data.drop(columns=["loss"])
            self.threshold = np.median(data.loss)
            self.y = data.loss > self.threshold
            super().load_dataset()
