from .dataset import *

class SoccerFRA(Dataset):
    dataset_name = "spadl-whoscored-FRA-xg.h5"
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, self.dataset_name)
            self.X = pd.read_hdf(data_path, key="X").reset_index(drop=True)
            self.y = pd.read_hdf(data_path, key="y").reset_index(drop=True)
            self.minmax_normalize()
            super().load_dataset()

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.8
        return params

    def _apply_rev_map(self, series, m):
        values = self.X[series].unique()
        values.sort()
        if len(values) == len(m): # default value of 0.0 never used
            print("yo")
        elif len(values) == len(m)+1: # default value was used
            print("yoyo")
        else:
            raise RuntimeError("should not happen", len(values), len(m))

    def map_to_categories(self, X):
        actiongroup0 = {
            "shot": 1.0,
            "dribble": 2.0,
            "pass": 3.0,
            "cross": 4.0,
            "clearance": 5.0,
            "tackle": 6.0,
            "take_on": 7.0,
            "interception": 8.0,
        }
        actiongroup1 = {
            "keeper_punch": 1.0,
            "keeper_pick_up": 2.0,
            "keeper_claim": 3.0,
            "keeper_save": 4.0,
        }

        actiongroup2 = {
            "throw_in": 1.0,
            "corner_short": 2.0,
            "corner_crossed": 3.0,
        }

        actiongroup3 = {
            "bad_touch": 1.0,
            "foul": 2.0,
            "shot_freekick": 3.0,
            "goalkick": 4.0,
            "freekick_crossed": 5.0,
            "freekick_short": 6.0,
            "shot_penalty": 7.0,
        }

        bodypart = {
            "foot": 1.0,
            "other": 2.0,
            "head/other": 2.5,
            "head": 3.0,
        }

        self._apply_rev_map("actiongroup0_a1", actiongroup0)


class NormalVsAdversarial(Dataset):

    def __init__(self, dataset, nreal, min_nadv, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)
        self.dataset = dataset
        self.nreal = nreal
        self.nadv = -1
        self.min_nadv = min_nadv

        self.normals = None
        self.adversarials = None

    def load_dataset(self):
        self.dataset.load_dataset()
        self.X = self.dataset.X
        self.y = self.dataset.y

    def load_sample(self, fold):
        self.load_dataset()
        Xtrain, ytrain, Xtest, ytest = self.dataset.train_and_test_set(fold)
        Xtrain = Xtrain.to_numpy()

        if self.min_nadv+self.nreal > Xtrain.shape[0]:
            print(f"WARNING: not enough training set examples for "
                  f"{self.dataset.name()} NormalVsAdversarial")
            print("Changing nreal to",  Xtrain.shape[0] - self.min_nadv)
            self.nreal = Xtrain.shape[0] - self.min_nadv

        self.normals = Xtrain[:self.nreal, :]
        self.adversarials_basis = Xtrain[self.nreal:, :]

    def set_adversarials(self, adversarials):
        self.adversarials = adversarials
        self.nadv = adversarials.shape[0]

    def train_and_test_set(self, fold):
        if self.normals is None:
            raise RuntimeError("use load_sample first")
        if self.adversarials is None or self.nadv == -1:
            raise RuntimeError("set adversarials first (based on self.adversarials_basis)")

        _, _, Xtest, _ = self.dataset.train_and_test_set(fold)

        Xtrain = np.vstack((self.normals, self.adversarials))
        ytrain = np.hstack((np.zeros(self.nreal), np.ones(self.nadv)))
        ytest = np.zeros(Xtest.shape[0])

        return pd.DataFrame(Xtrain, columns=Xtest.columns), \
            pd.Series(ytrain), Xtest, pd.Series(ytest)

    def name(self):
        return f"{self.dataset.name()}vsAdv{self.nreal}"
