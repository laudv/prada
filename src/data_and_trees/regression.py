import os
import numpy as np
import pandas as pd

from .dataset import Dataset, Task

class Diagonal(Dataset):
    def __init__(self, num_samples=100, noise=0.0, **kwargs):
        self.num_samples = num_samples
        self.noise = noise
        super().__init__(Task.REGRESSION, **kwargs)

    def get_model_name(self, fold, model_type, num_trees, tree_depth, **kwargs):
        return super().get_model_name(fold, model_type, num_trees, tree_depth,
                                      num_samples=self.num_samples,
                                      noise=self.noise,
                                      **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            rng = np.random.default_rng(self.seed)
            floats = rng.random(self.num_samples)
            self.X = pd.DataFrame(floats, columns=["x"])
            self.y = pd.Series(floats + self.noise*rng.random(self.num_samples))
            super().load_dataset()

class Calhouse(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("calhouse", data_id=537)
            self.y = np.log(self.y)
            super().load_dataset()

class CPUSmall(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("cpusmall", data_id=227)
            super().load_dataset()

class Diamonds(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("diamonds", data_id=42225)
            super().load_dataset()

    def _transform_X_y(self, X, y):
        X = pd.get_dummies(X, columns=["cut", "color", "clarity"], drop_first=False)
        y = np.log(y)
        return X, y

class Allstate(Dataset):
    dataset_name = "allstate.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            allstate_data_path = os.path.join(self.data_dir, Allstate.dataset_name)
            data = pd.read_hdf(allstate_data_path)
            self.X = data.drop(columns=["loss"])
            self.y = data.loss
            super().load_dataset()

class KddCup99(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        X["service_ecr_i"] = (X.service == "ecr_i")
        X["service_priv"] = (X.service == "private")
        X["service_http"] = (X.service == "http")
        X["service_smtp"] = (X.service == "smtp")
        X["flag_sf"] = (X.flag == "SF")
        X["flag_s0"] = (X.flag == "S0")
        X["flag_rej"] = (X.flag == "rej")
        X = pd.get_dummies(X, columns=["protocol_type"], drop_first=True)
        X.drop(inplace=True, columns=["service", "flag", "land", "urgent"])
        y = (y=="normal")
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("kkdcup99", data_id=1113)
            self.minmax_normalize()
            super().load_dataset()

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.8
        return params

class AmesHousing(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("ames_housing", data_id=42165)
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, X, y):
        XX = pd.get_dummies(X, columns=['MSZoning', 'Street', 'Alley',
                                        'LotShape', 'LandContour', 'Utilities',
                                        'LotConfig', 'LandSlope',
                                        'Neighborhood', 'Condition1',
                                        'Condition2', 'BldgType', 'HouseStyle',
                                        'RoofStyle', 'RoofMatl', 'Exterior1st',
                                        'Exterior2nd', 'MasVnrType',
                                        'ExterQual', 'ExterCond', 'Foundation',
                                        'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                        'BsmtFinType1', 'BsmtFinType2',
                                        'Heating', 'HeatingQC', 'CentralAir',
                                        'Electrical', 'KitchenQual',
                                        'Functional', 'FireplaceQu',
                                        'GarageType', 'GarageFinish',
                                        'GarageQual', 'GarageCond',
                                        'PavedDrive', 'PoolQC', 'Fence',
                                        'MiscFeature', 'SaleType',
                                        'SaleCondition'], drop_first=False)
        XX.drop(columns=["LotFrontage"], inplace=True) # too many missing
        XX.dropna(inplace=True)
        y = np.log(y)
        return XX, y
