import os, json
import joblib
import enum
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KDTree
from functools import partial

import xgboost as xgb

VERITAS_SUPPORT = False
try: 
    import veritas
    VERITAS_SUPPORT = True
finally: pass

MODEL_DIR = os.environ["DATA_AND_TREES_MODEL_DIR"]
DATA_DIR = os.environ["DATA_AND_TREES_DATA_DIR"]
NTHREADS = os.cpu_count()

class Task(enum.Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTI_CLASSIFICATION = 3

class Dataset:
    def __init__(self, task, name_suffix=""):
        self.task = task
        self.model_dir = MODEL_DIR
        self.data_dir = DATA_DIR
        self.nthreads = NTHREADS

        self.name_suffix = name_suffix # special parameters, name indication
        self.X = None
        self.y = None
        self.Itrain = None
        self.Itest = None
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None

    def name(self):
        return type(self).__name__

    def xgb_params(self, task, custom_params={}):
        if task == Task.REGRESSION:
            params = { # defaults
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "seed": 14,
                "nthread": self.nthreads,
            }
        elif task == Task.CLASSIFICATION:
            params = { # defaults
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
                "seed": 220,
                "nthread": self.nthreads,
            }
        elif task == Task.MULTI_CLASSIFICATION:
            params = {
                "num_class": 0,
                "objective": "multi:softmax",
                "tree_method": "hist",
                "eval_metric": "merror",
                "seed": 53589,
                "nthread": self.nthreads,
            }
        else:
            raise RuntimeError("unknown task")
        params.update(custom_params)
        return params

    def rf_params(self, custom_params):
        params = custom_params.copy()
        params["n_jobs"] = self.nthreads
        return params

    def extra_trees_params(self, custom_params):
        params = custom_params.copy()
        params["n_jobs"] = self.nthreads
        return params

    def load_dataset(self): # populate X, y
        raise RuntimeError("not implemented")

    def to_float32(self):
        if self.X is not None: self.X = self.X.astype(np.float32)
        if self.y is not None: self.y = self.y.astype(np.float32)
        if self.Xtrain is not None: self.Xtrain = self.Xtrain.astype(np.float32)
        if self.ytrain is not None: self.ytrain = self.ytrain.astype(np.float32)
        if self.Xtest is not None: self.Xtest = self.Xtest.astype(np.float32)
        if self.ytest is not None: self.ytest = self.ytest.astype(np.float32)

    def train_and_test_set(self, seed=39482, split_fraction=0.9, force=False):
        if self.X is None or self.y is None or force:
            raise RuntimeError("data not loaded")

        if self.Itrain is None or self.Itest is None or force:
            np.random.seed(seed)
            indices = np.random.permutation(self.X.shape[0])

            m = int(self.X.shape[0]*split_fraction)
            self.Itrain = indices[0:m]
            self.Itest = indices[m:]

        if self.Xtrain is None or self.ytrain is None or force:
            self.Xtrain = self.X.iloc[self.Itrain]
            self.ytrain = self.y[self.Itrain]

        if self.Xtest is None or self.ytest is None or force:
            self.Xtest = self.X.iloc[self.Itest]
            self.ytest = self.y[self.Itest]

    def cross_validation(self, Itrain, Itest, force=False):
        if self.X is None or self.y is None:
            raise RuntimeError("data not loaded")

        if self.Itrain is None or self.Itest is None or force:
            self.Itrain = Itrain
            self.Itest = Itest

        if self.Xtrain is None or self.ytrain is None or force:
            self.Xtrain = self.X.iloc[self.Itrain]
            self.ytrain = self.y[self.Itrain]

        if self.Xtest is None or self.ytest is None or force:
            self.Xtest = self.X.iloc[self.Itest]
            self.ytest = self.y[self.Itest]

    def get_addtree(self, model, meta):
        if not VERITAS_SUPPORT:
            raise RuntimeError("Veritas not installed")
        if isinstance(model, xgb.Booster):
            feat2id_dict = { s: i for i, s in enumerate(meta["columns"]) }
            feat2id = feat2id_dict.__getitem__
            if self.task == Task.MULTI_CLASSIFICATION:
                nclasses = self.num_classes
                return veritas.addtrees_from_multiclass_xgb_model(model,
                        nclasses, feat2id)
            else:
                return veritas.addtree_from_xgb_model(model, feat2id)
        else:
            if self.task == Task.MULTI_CLASSIFICATION:
                nclasses = self.num_classes
                return veritas.addtrees_from_multiclass_sklearn_ensemble(model,
                        nclasses)
            else:
                return veritas.addtree_from_sklearn_ensemble(model)

    def _load_openml(self, name, data_id, force=False, y_type=np.float32):
        if not os.path.exists(f"{self.data_dir}/{name}.h5") or force:
            print(f"loading {name} with fetch_openml")
            X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
            X = X.astype(np.float32)
            y = y.astype(y_type)
            X.to_hdf(f"{self.data_dir}/{name}.h5", key="X", complevel=9)
            y.to_hdf(f"{self.data_dir}/{name}.h5", key="y", complevel=9)

        print(f"loading {name} h5 file")
        X = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="X")
        y = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="y")

        return X, y

    ## model_cmp: (model, best_metric) -> `best_metric` if not better, new metric value if better
    #  best metric can be None
    def _get_xgb_model(self, num_trees, tree_depth,
            model_cmp, metric_name, custom_params={}):
        model_name = self.get_model_name("xgb", num_trees, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading XGB model from file: {model_name}")
            model, meta = joblib.load(model_path)
        else: # train model
            self.load_dataset()
            self.train_and_test_set()

            self.dtrain = xgb.DMatrix(self.Xtrain, self.ytrain, missing=None)
            self.dtest = xgb.DMatrix(self.Xtest, self.ytest, missing=None)

            params = self.xgb_params(self.task, custom_params)
            params["max_depth"] = tree_depth

            # Looking for best learning rate
            best_metric, best_model, best_lr = None, None, None
            for lr in np.linspace(0, 1, 5)[1:]:
                print("(1) LEARNING_RATE =", lr)
                params["learning_rate"] = lr
                model = xgb.train(params, self.dtrain, num_boost_round=num_trees,
                                  evals=[(self.dtrain, "train"), (self.dtest, "test")])
                metric = model_cmp(model, best_metric)
                if metric != best_metric:
                    best_metric, best_model, best_lr = metric, model, lr

            for lr in np.linspace(best_lr - 0.25, best_lr + 0.25, 7)[1:-1]:
                if lr <= 0.0 or lr >= 1.0: continue
                if lr in np.linspace(0, 1, 5)[1:]: continue
                print("(2) LEARNING_RATE =", lr)
                params["learning_rate"] = lr
                model = xgb.train(params, self.dtrain, num_boost_round=num_trees,
                                  evals=[(self.dtrain, "train"), (self.dtest, "test")])
                metric = model_cmp(model, best_metric)
                if metric != best_metric:
                    best_metric, best_model, best_lr = metric, model, lr
            print(f"(*) best metric = {best_metric} for lr = {best_lr}")

            model = best_model
            params["num_trees"] = num_trees
            meta = {
                "params": params,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, best_metric),
                "lr": best_lr,
            }
            joblib.dump((best_model, meta), model_path)

            del self.dtrain
            del self.dtest

        return model, meta

    def get_xgb_model(self, num_trees, tree_depth):
        # call _get_xgb_model with model comparison for lr optimization
        raise RuntimeError("override in subclass")

    def get_rf_model(self, num_trees, tree_depth):
        model_name = self.get_model_name("rf", num_trees, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading RF model from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            self.train_and_test_set()

            custom_params = {
                "n_estimators": num_trees,
                "max_depth": tree_depth,
            }
            params = self.rf_params(custom_params)

            if self.task == Task.REGRESSION:
                model = RandomForestRegressor(**params).fit(self.Xtrain, self.ytrain)
                metric = metrics.mean_squared_error(model.predict(self.Xtest), self.ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                model = RandomForestClassifier(**params).fit(self.Xtrain, self.ytrain)
                metric = metrics.accuracy_score(model.predict(self.Xtest), self.ytest)
                metric_name = "acc"

            meta = {
                "params": params,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
            }
            joblib.dump((model, meta), model_path)
        return model, meta

    def get_extra_trees_model(self, num_trees, tree_depth):
        model_name = self.get_model_name("et", num_trees, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading ExtraTrees from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            self.train_and_test_set()

            custom_params = {
                "n_estimators": num_trees,
                "max_depth": tree_depth,
                "random_state": 0,
            }
            params = self.extra_trees_params(custom_params)

            if self.task == Task.REGRESSION:
                model = ExtraTreesRegressor(**params).fit(self.Xtrain, self.ytrain)
                metric = metrics.mean_squared_error(model.predict(self.Xtest), self.ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                model = ExtraTreesClassifier(**params).fit(self.Xtrain, self.ytrain)
                metric = metrics.accuracy_score(model.predict(self.Xtest), self.ytest)
                metric_name = "acc"
            meta = {
                "params": params,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
            }
            joblib.dump((model, meta), model_path)
        return model, meta

    def get_kdtree(self):
        model_name = self.get_model_name("kdtree", 0, 0)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading KDTree from file: {model_name}")
            kdtree = joblib.load(model_path)
        else:
            self.load_dataset()
            self.train_and_test_set()
            kdtree = KDTree(self.Xtrain)
            joblib.dump(kdtree, model_path)
        return kdtree
        
    def get_model_name(self, model_type, num_trees, tree_depth):
        return f"{self.name()}{self.name_suffix}-{num_trees}-{tree_depth}.{model_type}"

    def minmax_normalize(self):
        if self.X is None:
            raise RuntimeError("data not loaded")

        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

def _rmse_metric(self, model, best_m):
    yhat = model.predict(self.dtest, output_margin=True)
    m = metrics.mean_squared_error(yhat, self.ytest)
    m = np.sqrt(m)
    return m if best_m is None or m < best_m else best_m

def _acc_metric(self, model, best_m):
    yhat = model.predict(self.dtest, output_margin=True)
    m = metrics.accuracy_score(yhat > 0.0, self.ytest)
    return m if best_m is None or m > best_m else best_m

def _multi_acc_metric(self, model, best_m):
    yhat = model.predict(self.dtest)
    m = metrics.accuracy_score(yhat, self.ytest)
    return m if best_m is None or m > best_m else best_m

class MulticlassDataset(Dataset):
    def __init__(self, num_classes):
        super().__init__(Task.MULTI_CLASSIFICATION)
        self.num_classes = num_classes

    def get_class(self, cls):
        self.load_dataset()
        mask = np.zeros(self.X.shape[0])
        if isinstance(cls, tuple):
            for c in cls:
                if c not in range(self.num_classes):
                    raise ValueError(f"invalid class {c}")
                mask = np.bitwise_or(mask, self.y == c)
        else:
            if cls not in range(self.num_classes):
                raise ValueError(f"invalid class {cls}")
            mask = (self.y == cls)
        X = self.X.loc[mask, :]
        y = self.y[mask]
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        return X, y

class MultiBinClassDataset(Dataset):
    def __init__(self, multiclass_dataset, class1, class2):
        super().__init__(Task.CLASSIFICATION)
        self.multi_dataset = multiclass_dataset
        if not isinstance(self.multi_dataset, MulticlassDataset):
            raise ValueError("not a multiclass dataset:",
                    self.multi_dataset.name())
        if class1 not in range(self.multi_dataset.num_classes):
            raise ValueError("invalid class1")
        if class2 not in range(self.multi_dataset.num_classes):
            raise ValueError("invalid class2")
        if class1 >= class2:
            raise ValueError("take class1 < class2")
        self.class1 = class1
        self.class2 = class2

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.multi_dataset.load_dataset()
            X, y = self.multi_dataset.get_class((self.class1, self.class2))
            self.X = X
            self.y = (y==self.class2)

    def name(self):
        return f"{super().name()}{self.class1}v{self.class2}"

class Calhouse(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("calhouse", data_id=537)
            self.y = np.log(self.y)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_rmse_metric, self), "rmse", custom_params)

class Allstate(Dataset):
    dataset_name = "allstate.h5"

    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            allstate_data_path = os.path.join(self.data_dir, Allstate.dataset_name)
            data = pd.read_hdf(allstate_data_path)
            self.X = data.drop(columns=["loss"])
            self.y = data.loss

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_rmse_metric, self), "rmse", custom_params)

class Covtype(Dataset):
    def __init__(self):
        super().__init__(Task.CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("covtype", data_id=1596)
            self.y = (self.y==2)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class CovtypeNormalized(Covtype):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.minmax_normalize()

class Higgs(Dataset):
    def __init__(self):
        super().__init__(Task.CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(self.data_dir, "higgs.h5")
            self.X = pd.read_hdf(higgs_data_path, "X")
            self.y = pd.read_hdf(higgs_data_path, "y")

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class LargeHiggs(Dataset):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(self.data_dir, "higgs_large.h5")
            data = pd.read_hdf(higgs_data_path)
            self.y = data[0]
            self.X = data.drop(columns=[0])
            columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.X.columns = columns
            self.minmax_normalize()

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class Mnist(MulticlassDataset):
    def __init__(self):
        super().__init__(num_classes=10)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("mnist", data_id=554)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {
            "num_class": self.num_classes,
        }
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_multi_acc_metric, self), "macc", custom_params)

#class MnistNormalized(Mnist):
#    def __init__(self):
#        super().__init__()
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            super().load_dataset()
#            self.minmax_normalize()

class MnistBinClass(MultiBinClassDataset):
    def __init__(self, class1, class2):
        super().__init__(Mnist(), class1, class2)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {
            "subsample": 0.5,
            "colsample_bytree": 0.8,
        }
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

# 0  T-shirt/top
# 1  Trouser
# 2  Pullover
# 3  Dress
# 4  Coat
# 5  Sandal
# 6  Shirt
# 7  Sneaker
# 8  Bag
# 9  Ankle boot
class FashionMnist(MulticlassDataset):
    def __init__(self):
        super().__init__(num_classes=10)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("fashion_mnist", data_id=40996)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {
            "num_class": self.num_classes,
        }
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_multi_acc_metric, self), "macc", custom_params)

class FashionMnistBinClass(MultiBinClassDataset):
    def __init__(self, class1, class2):
        super().__init__(FashionMnist(), class1, class2)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {
            "subsample": 0.5,
            "colsample_bytree": 0.8,
        }
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class Ijcnn1(Dataset):
    dataset_name = "ijcnn1.h5"

    def __init__(self):
        super().__init__(Task.CLASSIFICATION)

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

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = { }
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class Webspam(Dataset):
    dataset_name = "webspam_wc_normalized_unigram.h5"

    def __init__(self):
        super().__init__(Task.CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, Webspam.dataset_name)
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            self.minmax_normalize()

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = { }
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class BreastCancer(Dataset):
    def __init__(self):
        super().__init__(Task.CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("breast-w", data_id=15, y_type=str)
            self.y = (self.y == 'malignant')
            self.X.fillna(self.X.mean(), inplace=True)
            self.y = self.y.astype(np.float32)

    def get_xgb_model(self, num_trees, tree_depth):
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(_acc_metric, self), "acc", custom_params)

class BreastCancerNormalized(BreastCancer):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.minmax_normalize()
