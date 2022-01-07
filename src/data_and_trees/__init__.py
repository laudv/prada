import os, json, time
import joblib
import enum
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KDTree, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
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
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "seed": self.seed,
                "nthread": self.nthreads,
            }
        elif task == Task.CLASSIFICATION:
            params = { # defaults
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
                "seed": self.seed,
                "nthread": self.nthreads,
            }
        elif task == Task.MULTI_CLASSIFICATION:
            params = {
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
        return { "n_jobs": self.nthreads }

    def extra_trees_params(self):
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

            model = xgb.train(params, dtrain, num_boost_round=num_trees,
                              evals=[(dtrain, "train"), (dtest, "test")])

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

    def get_rf_model(self, fold, num_trees, tree_depth):
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

            if self.task == Task.REGRESSION:
                model = RandomForestRegressor(**params).fit(Xtrain, ytrain)
                metric = metrics.mean_squared_error(model.predict(Xtest), ytest)
                metric = np.sqrt(metric)
                metric_name = "rmse"
            else:
                model = RandomForestClassifier(**params).fit(Xtrain, ytrain)
                metric = metrics.accuracy_score(model.predict(Xtest), ytest)
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
            lof = LocalOutlierFactor()
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

class Calhouse(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.REGRESSION, **kwargs)
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("calhouse", data_id=537)
            self.y = np.log(self.y)
            super().load_dataset()

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

class Covtype(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("covtype", data_id=1596)
            self.y = (self.y==2)
            super().load_dataset()

class CovtypeNormalized(Covtype):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.minmax_normalize()

class Higgs(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(self.data_dir, "higgs.h5")
            self.X = pd.read_hdf(higgs_data_path, "X")
            self.y = pd.read_hdf(higgs_data_path, "y")
            super().load_dataset()

class LargeHiggs(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(self.data_dir, "higgs_large.h5")
            data = pd.read_hdf(higgs_data_path)
            self.y = data[0]
            self.X = data.drop(columns=[0])
            columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.X.columns = columns
            self.minmax_normalize()
            super().load_dataset()

class Mnist(MulticlassDataset):
    def __init__(self, **kwargs):
        super().__init__(num_classes=10, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("mnist", data_id=554)
            super().load_dataset()

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        return params

class MnistBinClass(MultiBinClassDataset):
    def __init__(self, class1, class2, **kwargs):
        super().__init__(Mnist(), class1, class2, **kwargs)

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.8
        return params

class MnistKvAll(OneVsAllDataset):
    def __init__(self, class1, **kwargs):
        super().__init__(Mnist(), class1, **kwargs)

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        return params

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
    def __init__(self, **kwargs):
        super().__init__(num_classes=10, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("fashion_mnist", data_id=40996)
            super().load_dataset()

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        return params

class FashionMnistBinClass(MultiBinClassDataset):
    def __init__(self, class1, class2, **kwargs):
        super().__init__(FashionMnist(), class1, class2, **kwargs)


    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.8
        return params

class FashionMnistKvAll(OneVsAllDataset):
    def __init__(self, class1, **kwargs):
        super().__init__(FashionMnist(), class1, **kwargs)

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        return params

class Ijcnn1(Dataset):
    dataset_name = "ijcnn1.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

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

class Webspam(Dataset):
    dataset_name = "webspam_wc_normalized_unigram.h5"

    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, Webspam.dataset_name)
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            self.minmax_normalize()
            super().load_dataset()

class BreastCancer(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        y = (y == 'malignant')
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("breast-w", data_id=15)
            self.X.fillna(self.X.mean(), inplace=True)
            super().load_dataset()

class BreastCancerNormalized(BreastCancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.minmax_normalize()

class Adult(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        X["workclass"] = (X.workclass=="private")
        X["male"] = (X.sex=="Male")
        X["from_us"] = (X["native-country"]=="United-States")
        X["marital-status"] = \
            (X["marital-status"]=="Married-vic-spouse") * 4.0\
            + (X["marital-status"]=="Never-married") * 3.0\
            + (X["marital-status"]=="Divorced") * 2.0\
            + (X["marital-status"]=="Separated") * 1.0
        X = pd.get_dummies(X, columns=["occupation", "relationship", "race"], drop_first=True)
        X.drop(inplace=True, columns=["education", "sex", "native-country"])
        y = (y == ">50K")
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("adult", data_id=179)
            self.minmax_normalize()
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

class Vehicle(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        y = (y == "bus") | (y == "van")
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("vehicle", data_id=54)
            self.minmax_normalize()
            super().load_dataset()

class Phoneme(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        y = (y == "2") # y values are in ['1', '2'] -> transform to binary
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("phoneme", data_id=1489)
            self.minmax_normalize()
            super().load_dataset()

class Spambase(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        y = (y == "1") # y values are in ['0', '1'] -> transform to binary
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("spambase", data_id=44)
            self.minmax_normalize()
            super().load_dataset()

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
