import os, json
import joblib
import enum
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from functools import partial

import xgboost as xgb

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

    def xgb_params(self, task, custom_params={}):
        if task == Task.REGRESSION:
            params = { # defaults
                "objective": "reg:squarederror",
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
        params.update(custom_params)
        return params

    def rf_params(self, custom_params):
        params = custom_params.copy()
        return params

    def load_dataset(self): # populate X, y
        raise RuntimeError("not implemented")

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

    def _load_openml(self, name, data_id, force=False):
        if not os.path.exists(f"{self.data_dir}/{name}.h5") or force:
            print(f"loading {name} with fetch_openml")
            X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            X.to_hdf(f"{self.data_dir}/{name}.h5", key="X", complevel=9)
            y.to_hdf(f"{self.data_dir}/{name}.h5", key="y", complevel=9)

        print(f"loading {name} h5 file")
        X = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="X")
        y = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="y")

        return X, y

    ## model_cmp: (model, best_metric) -> `best_metric` if not better, new metric value if better
    #  best metric can be None
    def _get_xgb_model(self, num_trees, tree_depth,
            model_cmp, custom_params={}):
        model_name = self.get_model_name("xgb", num_trees, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading model from file: {model_name}")
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
                print(f"(1) best metric = {best_metric}")

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
                print(f"(2) best metric = {best_metric}")

            model = best_model
            params["num_trees"] = num_trees
            meta = {"params": params, "columns": self.X.columns}
            joblib.dump((best_model, meta), model_path)

            del self.dtrain
            del self.dtest

        return model, meta

    def get_xgb_model(self, num_trees, tree_depth):
        # call _get_xgb_model with model comparison for lr optimization
        raise RuntimeError("override in subclass")

    def get_model_name(self, model_type, num_trees, tree_depth):
        return f"{type(self).__name__}{self.name_suffix}-{num_trees}-{tree_depth}.{model_type}"

    def minmax_normalize(self):
        if X is None:
            raise RuntimeError("data not loaded")

        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df
        self.name_suffix = f"-normalized{self.name_suffix}"

class Calhouse(Dataset):

    def __init__(self):
        super().__init__(Task.REGRESSION)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("calhouse", data_id=537)
            self.y = np.log(self.y)

    def get_xgb_model(self, num_trees, tree_depth):
        def metric(self, model, best_m):
            yhat = model.predict(self.dtest, output_margin=True)
            m = metrics.mean_squared_error(yhat, self.ytest)
            return m if best_m is None or m < best_m else best_m
        custom_params = {}
        return super()._get_xgb_model(num_trees, tree_depth,
                partial(metric, self), custom_params)

#class Allstate(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "reg:squarederror",
#            "tree_method": "hist",
#            "seed": 14,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            allstate_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "allstate.h5")
#            data = pd.read_hdf(allstate_data_path)
#            self.X = data.drop(columns=["loss"])
#            self.y = data.loss
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat): #maximized
#                return -metrics.mean_squared_error(y, raw_yhat)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#
#class Covtype(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 235,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            self.X, self.y = util.load_openml("covtype", data_id=1596)
#            self.y = (self.y==2)
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#        
#class CovtypeNormalized(Covtype):
#    def __init__(self):
#        super().__init__()
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            super().load_dataset()
#            self.minmax_normalize()
#
#class Higgs(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 220,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            higgs_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "higgs.h5")
#            self.X = pd.read_hdf(higgs_data_path, "X")
#            self.y = pd.read_hdf(higgs_data_path, "y")
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#
#class LargeHiggs(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 220,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            higgs_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "higgs_large.h5")
#            data = pd.read_hdf(higgs_data_path)
#            self.y = data[0]
#            self.X = data.drop(columns=[0])
#            columns = [f"a{i}" for i in range(self.X.shape[1])]
#            self.X.columns = columns
#            self.minmax_normalize()
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#
#
#class Mnist(Dataset):
#
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "num_class": 10,
#            "objective": "multi:softmax",
#            "tree_method": "hist",
#            "eval_metric": "merror",
#            "seed": 53589,
#            "nthread": 4,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            self.X, self.y = util.load_openml("mnist", data_id=554)
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, yhat): #maximized
#                return metrics.accuracy_score(y, yhat)
#            
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtrees_from_multiclass_xgb_model(self.model, 10, feat2id_map=self.feat2id)
#        for at in self.at:
#            at.base_score = 0
#
#class MnistNormalized(Mnist):
#    def __init__(self):
#        super().__init__()
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            super().load_dataset()
#            self.minmax_normalize()
#
#class Mnist2v6(Mnist):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 235,
#            "nthread": 4,
#            "subsample": 0.5,
#            "colsample_bytree": 0.8,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            super().load_dataset()
#            self.X = self.X.loc[(self.y==2) | (self.y==6), :]
#            self.y = self.y[(self.y==2) | (self.y==6)]
#            self.y = (self.y == 2.0).astype(float)
#            self.X.reset_index(inplace=True, drop=True)
#            self.y.reset_index(inplace=True, drop=True)
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#
#class FashionMnist(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "num_class": 10,
#            "objective": "multi:softmax",
#            "tree_method": "hist",
#            "eval_metric": "merror",
#            "seed": 132955,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            self.X, self.y = util.load_openml("fashion_mnist", data_id=40996)
#            #self.minmax_normalize()
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, yhat): #maximized
#                return metrics.accuracy_score(y, yhat)
#            
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtrees_from_multiclass_xgb_model(self.model, 10, feat2id_map=self.feat2id)
#        for at in self.at:
#            at.base_score = 0
#
#class FashionMnist2v6(FashionMnist):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 235,
#            "nthread": 4,
#            "subsample": 0.5,
#            "colsample_bytree": 0.8,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            super().load_dataset()
#            self.X = self.X.loc[(self.y==2) | (self.y==6), :]
#            self.y = self.y[(self.y==2) | (self.y==6)]
#            self.y = (self.y == 2.0).astype(float)
#            self.X.reset_index(inplace=True, drop=True)
#            self.y.reset_index(inplace=True, drop=True)
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#
#class Ijcnn1(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 235,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            ijcnn1_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "ijcnn1.h5")
#            self.X = pd.read_hdf(ijcnn1_data_path, "Xtrain")
#            self.Xtest = pd.read_hdf(ijcnn1_data_path, "Xtest")
#            columns = [f"a{i}" for i in range(self.X.shape[1])]
#            self.X.columns = columns
#            self.Xtest.columns = columns
#            self.y = pd.read_hdf(ijcnn1_data_path, "ytrain")
#            self.ytest = pd.read_hdf(ijcnn1_data_path, "ytest")
#            self.minmax_normalize()
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
#
#class Webspam(Dataset):
#    def __init__(self):
#        super().__init__()
#        self.params = {
#            "objective": "binary:logistic",
#            "eval_metric": "error",
#            "tree_method": "hist",
#            "seed": 732,
#            "nthread": 1,
#        }
#
#    def load_dataset(self):
#        if self.X is None or self.y is None:
#            data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "webspam_wc_normalized_unigram.h5")
#            self.X = pd.read_hdf(data_path, "X")
#            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
#            self.y = pd.read_hdf(data_path, "y")
#            self.minmax_normalize()
#
#    def load_model(self, num_trees, tree_depth):
#        model_name = self.get_model_name(num_trees, tree_depth)
#        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
#            self.load_dataset()
#            print(f"training model depth={tree_depth}, num_trees={num_trees}")
#
#            def metric(y, raw_yhat):
#                return metrics.accuracy_score(y, raw_yhat > 0)
#
#            self.params["max_depth"] = tree_depth
#            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
#                    self.y, self.params, num_trees, metric)
#
#            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}
#
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
#                pickle.dump(self.model, f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
#                json.dump(self.meta, f)
#        else:
#            print(f"loading model from file: {model_name}")
#            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
#                self.model = pickle.load(f)
#            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
#                self.meta = json.load(f)
#
#        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
#        self.feat2id = lambda x: feat2id_dict[x]
#        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
#        self.at.base_score = 0
