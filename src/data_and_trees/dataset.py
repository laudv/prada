import os
import time
import joblib
import enum
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KDTree, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import xgboost as xgb

try: 
    import veritas
    VERITAS_SUPPORT = True
except ModuleNotFoundError:
    VERITAS_SUPPORT = False

try:
    import groot.model
    GROOT_SUPPORT = True
except ModuleNotFoundError:
    GROOT_SUPPORT = False

MODEL_DIR = os.environ["DATA_AND_TREES_MODEL_DIR"]
DATA_DIR = os.environ["DATA_AND_TREES_DATA_DIR"]
NTHREADS = os.cpu_count()

class Task(enum.Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTI_CLASSIFICATION = 3

class Dataset:
    def __init__(self, task, nfolds=10, seed=18):
        self.task = task
        self.model_dir = MODEL_DIR
        self.data_dir = DATA_DIR
        self.nthreads = NTHREADS
        self.nfolds = nfolds
        self.seed = seed

        self.X = None
        self.y = None

    def name(self):
        return type(self).__name__

    def xgb_params(self, task):
        if task == Task.REGRESSION:
            params = { # defaults
                "base_score": 0.0,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "seed": self.seed,
                "nthread": self.nthreads,
            }
        elif task == Task.CLASSIFICATION:
            params = { # defaults
                #"base_score": 0.5,
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
                "seed": self.seed,
                "nthread": self.nthreads,
            }
        elif task == Task.MULTI_CLASSIFICATION:
            params = {
                #"base_score": 0.0,
                "num_class": 0,
                "objective": "multi:softmax",
                "tree_method": "hist",
                "eval_metric": "merror",
                "seed": self.seed,
                "nthread": self.nthreads,
            }
        else:
            raise RuntimeError("unknown task")
        return params

    def rf_params(self):
        return {
            "n_jobs": self.nthreads,
            "max_depth": None,
            "max_leaf_nodes": 254,
        }

    def groot_params(self):
        return { "n_jobs": self.nthreads, "min_samples_leaf": 2 }

    def extra_trees_params(self):
        return { "n_jobs": self.nthreads }

    def decision_tree_params(self):
        return { }

    def linear_params(self):
        return { "n_jobs": self.nthreads }

    def load_dataset(self): # populate X, y
        if self.X is None or self.y is None:
            raise RuntimeError("override this and call after loading X and y")
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)
        N = self.X.shape[0]

        rand = np.random.RandomState(self.seed)
        self.Is = rand.permutation(N)

        fold_size = self.Is.shape[0] / self.nfolds
        self.Ifolds = [self.Is[int(i*fold_size):int((i+1)*fold_size)]
                            for i in range(self.nfolds)]

    def train_and_test_set(self, fold):
        Itrain = np.hstack([self.Ifolds[j] for j in range(self.nfolds) if fold!=j])
        Xtrain = self.X.iloc[Itrain, :]
        ytrain = self.y[Itrain]
        Itest = self.Ifolds[fold]
        Xtest = self.X.iloc[Itest, :]
        ytest = self.y[Itest]
        return Xtrain, ytrain, Xtest, ytest

    def get_addtree(self, model, meta):
        if not VERITAS_SUPPORT:
            raise RuntimeError("Veritas not installed")
        if isinstance(model, xgb.XGBModel):
            model = model.get_booster()
        if isinstance(model, xgb.Booster):
            feat2id_dict = { s: i for i, s in enumerate(meta["columns"]) }
            feat2id = feat2id_dict.__getitem__
            if self.task == Task.MULTI_CLASSIFICATION:
                nclasses = self.num_classes
                return veritas.addtrees_from_multiclass_xgb_model(model,
                        nclasses, feat2id)
            elif self.task == Task.REGRESSION:
                return veritas.addtree_from_xgb_model(model, feat2id,
                                                      base_score=0.5)
            else:
                return veritas.addtree_from_xgb_model(model, feat2id)
        elif GROOT_SUPPORT and \
                isinstance(model, groot.model.GrootRandomForestClassifier):
            return veritas.addtree_from_groot_ensemble(model)
        else:
            if self.task == Task.MULTI_CLASSIFICATION:
                nclasses = self.num_classes
                return veritas.addtrees_from_multiclass_sklearn_ensemble(model,
                        nclasses)
            else:
                return veritas.addtree_from_sklearn_ensemble(model)

    def _load_openml(self, name, data_id, force=False):
        if not os.path.exists(f"{self.data_dir}/{name}.h5") or force:
            print(f"loading {name} with fetch_openml")
            X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
            X, y = self._transform_X_y(X, y)
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            X.to_hdf(f"{self.data_dir}/{name}.h5", key="X", complevel=9)
            y.to_hdf(f"{self.data_dir}/{name}.h5", key="y", complevel=9)

        print(f"loading {name} h5 file")
        X = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="X")
        y = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="y")

        return X, y

    def _transform_X_y(self, X, y): # override if necessary
        return X, y

    ## model_cmp: (model, best_metric) -> `best_metric` if not better, new metric value if better
    #  best metric can be None
    def get_xgb_model(self, fold, learning_rate, num_trees, tree_depth):
        lr = learning_rate
        model_name = self.get_model_name(fold, "xgb", num_trees, tree_depth, lr=f"{lr*100:.0f}")
        model_path = os.path.join(self.model_dir, model_name)
        print("xgb_model_path", model_path)
        if os.path.isfile(model_path):
            print(f"loading XGB model from file: {model_path}")
            model, meta = joblib.load(model_path)
        else: # train model
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)

            dtrain = xgb.DMatrix(Xtrain, ytrain, missing=None)
            dtest = xgb.DMatrix(Xtest, ytest, missing=None)

            params = self.xgb_params(self.task)
            params["max_depth"] = tree_depth
            params["learning_rate"] = learning_rate

            t = time.time()
            model = xgb.train(params, dtrain, num_boost_round=num_trees,
                              evals=[(dtrain, "train"), (dtest, "test")])
            t = time.time() - t

            if self.task == Task.REGRESSION:
                metric = metrics.mean_squared_error(model.predict(dtest), ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                metric = metrics.accuracy_score(model.predict(dtest)>0.5, ytest)
                metric_name = "acc"

            params["num_trees"] = num_trees
            meta = {
                "params": params,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
                "lr": lr,
                "training_time": t,
            }
            joblib.dump((model, meta), model_path)
            return model, meta

            ## Looking for best learning rate
            #best_metric, best_model, best_lr = None, None, None
            #for lr in np.linspace(0, 1, 5)[1:]:
            #    print("(1) LEARNING_RATE =", lr)
            #    params["learning_rate"] = lr
            #    model = xgb.train(params, self.dtrain, num_boost_round=num_trees,
            #                      evals=[(self.dtrain, "train"), (self.dtest, "test")])
            #    metric = model_cmp(model, best_metric)
            #    if metric != best_metric:
            #        best_metric, best_model, best_lr = metric, model, lr

            #for lr in np.linspace(best_lr - 0.25, best_lr + 0.25, 7)[1:-1]:
            #    if lr <= 0.0 or lr >= 1.0: continue
            #    if lr in np.linspace(0, 1, 5)[1:]: continue
            #    print("(2) LEARNING_RATE =", lr)
            #    params["learning_rate"] = lr
            #    model = xgb.train(params, self.dtrain, num_boost_round=num_trees,
            #                      evals=[(self.dtrain, "train"), (self.dtest, "test")])
            #    metric = model_cmp(model, best_metric)
            #    if metric != best_metric:
            #        best_metric, best_model, best_lr = metric, model, lr
            #print(f"(*) best metric = {best_metric} for lr = {best_lr}")

            #model = best_model
            #params["num_trees"] = num_trees
            #meta = {
            #    "params": params,
            #    "num_trees": num_trees,
            #    "tree_depth": tree_depth,
            #    "columns": self.X.columns,
            #    "task": self.task,
            #    "metric": (metric_name, best_metric),
            #    "lr": best_lr,
            #}
            #joblib.dump((best_model, meta), model_path)

            #del self.dtrain
            #del self.dtest

        return model, meta

    def get_rf_model(self, fold, num_trees, tree_depth=None):
        model_name = self.get_model_name(fold, "rf", num_trees, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading RF model from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)

            params = self.rf_params()
            params["n_estimators"] = num_trees
            params["max_depth"] = tree_depth

            t = time.time()
            if self.task == Task.REGRESSION:
                model = RandomForestRegressor(**params).fit(Xtrain, ytrain)
                metric = metrics.mean_squared_error(model.predict(Xtest), ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                model = RandomForestClassifier(**params).fit(Xtrain, ytrain)
                metric = metrics.accuracy_score(model.predict(Xtest), ytest)
                metric_name = "acc"
            t = time.time() - t

            meta = {
                "params": params,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
                "training_time": t
            }
            joblib.dump((model, meta), model_path)
        return model, meta

    def get_decision_tree(self, fold, tree_depth):
        model_name = self.get_model_name(fold, "dt", 1, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading DT model from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)

            params = self.decision_tree_params()
            params["max_depth"] = tree_depth

            t = time.time()
            if self.task == Task.REGRESSION:
                model = DecisionTreeRegressor(**params).fit(Xtrain, ytrain)
                metric = metrics.mean_squared_error(model.predict(Xtest), ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                model = DecisionTreeClassifier(**params).fit(Xtrain, ytrain)
                metric = metrics.accuracy_score(model.predict(Xtest), ytest)
                metric_name = "acc"
            t = time.time() - t

            meta = {
                "params": params,
                "num_trees": 1,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
                "training_time": t
            }
            joblib.dump((model, meta), model_path)
        return model, meta

    def get_linear_model(self, fold):
        model_name = self.get_model_name(fold, "linear", 0, 0)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading linear model from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)

            params = self.linear_params()

            t = time.time()
            if self.task == Task.REGRESSION:
                model = LinearRegression(**params).fit(Xtrain, ytrain)
                metric = metrics.mean_squared_error(model.predict(Xtest), ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                model = LogisticRegression(**params).fit(Xtrain, ytrain)
                metric = metrics.accuracy_score(model.predict(Xtest), ytest)
                metric_name = "acc"
            t = time.time() - t

            meta = {
                "params": params,
                "num_trees": 0,
                "tree_depth": 0,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
                "training_time": t
            }
            joblib.dump((model, meta), model_path)
        return model, meta

    def get_groot_model(self, fold, num_trees, tree_depth, epsilon):
        if not GROOT_SUPPORT:
            raise RuntimeError("GROOT not installed")
        model_name = self.get_model_name(fold, "groot", num_trees, tree_depth,
                epsilon=str(np.round(epsilon, 4)))
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading GROOT model from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)

            params = self.groot_params()
            params["n_estimators"] = num_trees
            params["max_depth"] = tree_depth
            params["attack_model"] = np.ones(Xtrain.shape[1]) * epsilon

            t = time.time()
            if self.task == Task.REGRESSION:
                raise RuntimeError("not supported yet")
            else:
                model = groot.model.GrootRandomForestClassifier(**params)
                model.fit(Xtrain, ytrain)
                metric = metrics.accuracy_score(model.predict(Xtest), ytest)
                metric_name = "acc"
            t = time.time() - t

            meta = {
                "params": params,
                "num_trees": num_trees,
                "tree_depth": tree_depth,
                "columns": self.X.columns,
                "task": self.task,
                "metric": (metric_name, metric),
                "training_time": t,
            }
            joblib.dump((model, meta), model_path)
        return model, meta

    # TODO update to fold
    def get_extra_trees_model(self, num_trees, tree_depth):
        model_name = self.get_model_name("et", num_trees, tree_depth)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading ExtraTrees from file: {model_name}")
            model, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            self.train_and_test_set()

            params = self.extra_trees_params()
            params["n_estimators"] = num_trees
            params["max_depth"] = tree_depth
            params["random_state"] = 0

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

    def get_kdtree(self, fold):
        model_name = self.get_model_name("kdtree", 0, 0)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading KDTree from file: {model_name}")
            kdtree, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            t = time.time()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)
            kdtree = KDTree(Xtrain)
            t = time.time()-t
            print(f"trained KDTree in {t:.2f}s");
            meta = {"training_time": t}
            joblib.dump((kdtree, meta), model_path)
        return kdtree, meta

    def get_iforest(self, fold):
        model_name = self.get_model_name(fold, "iforest", 0, 0)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading Isolation Forest from file: {model_name}")
            iforest, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)
            t = time.time()
            iforest = IsolationForest().fit(Xtrain)
            t = time.time() - t
            print(f"trained Isolation Forest in {t:.2f}s");
            meta = {"training_time": t}
            joblib.dump((iforest, meta), model_path)
        return iforest, meta

    def get_lof(self, fold):
        model_name = self.get_model_name(fold, "lof", 0, 0)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading LocalOutlierFactor from file: {model_name}")
            lof, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)
            t = time.time()
            lof = LocalOutlierFactor(n_neighbors=10)
            lof.fit(Xtrain)
            t = time.time() - t
            print(f"trained LocalOutlierFactor in {t:.2f}s");
            meta = {"training_time": t}
            joblib.dump((lof, meta), model_path)
        return lof, meta

    def get_oneclasssvm(self, fold):
        model_name = self.get_model_name(fold, "ocsvm", 0, 0)
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.isfile(model_path):
            print(f"loading OneClassSVM from file: {model_name}")
            svm, meta = joblib.load(model_path)
        else:
            self.load_dataset()
            Xtrain, ytrain, Xtest, ytest = self.train_and_test_set(fold)
            t = time.time()
            svm = OneClassSVM()
            svm.fit(Xtrain)
            t = time.time() - t
            print(f"trained OneClassSVM in {t:.2f}s");
            meta = {"training_time": t}
            joblib.dump((svm, meta), model_path)
        return svm, meta
        
    def get_model_name(self, fold, model_type, num_trees, tree_depth, **kwargs):
        a = "-".join(f"{k}{v}" for k, v in kwargs.items())
        return f"{self.name()}-seed{self.seed}-fold{fold}_{num_trees}-{tree_depth}{a}.{model_type}"

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
    def __init__(self, num_classes, **kwargs):
        super().__init__(Task.MULTI_CLASSIFICATION, **kwargs)
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
    def __init__(self, multiclass_dataset, class1, class2, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)
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
            super().load_dataset()

    def name(self):
        return f"{super().name()}{self.class1}v{self.class2}"

class OneVsAllDataset(Dataset):
    def __init__(self, multiclass_dataset, class1, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)
        self.multi_dataset = multiclass_dataset
        if not isinstance(self.multi_dataset, MulticlassDataset):
            raise ValueError("not a multiclass dataset:",
                    self.multi_dataset.name())
        if class1 not in range(self.multi_dataset.num_classes):
            raise ValueError("invalid class1")
        self.class1 = class1

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.multi_dataset.load_dataset()
            self.X = self.multi_dataset.X
            self.y = (self.multi_dataset.y==self.class1)
            super().load_dataset()

    def name(self):
        return f"{self.multi_dataset.name()}{self.class1}vAll"
