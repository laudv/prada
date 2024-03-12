import os
import time
import enum
import numpy as np
import pandas as pd

try:
    DATA_DIR = os.environ["DATA_AND_TREES_DATA_DIR"]
except KeyError as e:
    print()
    print("| Environment variable DATA_AND_TREES_DATA_DIR not set. This is the")
    print("| path to the folder where DATA_AND_TREES stores the cached and")
    print("| compressed HDF5 files. Set the variable like this (unix):")
    print("|     `export DATA_AND_TREES_DATA_DIR=/path/to/data/dir`")
    print()
    raise e

NFOLDS = 5
SEED = 2537254
DTYPE = np.float32

class Task(enum.Enum):
    REGRESSION = 1
    BINARY = 2
    MULTICLASS = 3

class Dataset:
    def __init__(self, task, nfolds=NFOLDS, seed=SEED):
        self.task = task
        self.data_dir = DATA_DIR
        self.nfolds = nfolds
        self.seed = seed
        self.source = "manual"
        self.url = "unknown"

        self.X = None
        self.y = None

    def name(self):
        return type(self).__name__

    def is_regression(self):
        return self.task == Task.REGRESSION

    def is_binary(self):
        return self.task == Task.BINARY

    def is_multiclass(self):
        return self.task == Task.MULTICLASS

    def are_X_y_set(self):
        return self.X is not None and self.y is not None

    def load_dataset(self): # populate X, y
        if self.X is None or self.y is None:
            raise RuntimeError("override this and call after loading X and y")
        N = self.X.shape[0]

        rng = np.random.default_rng(self.seed)
        self.perm = rng.permutation(N)

        #fold_size = self.Is.shape[0] / self.nfolds
        #self.Ifolds = [self.Is[int(i*fold_size):int((i+1)*fold_size)]
        #               for i in range(self.nfolds)]

    def train_and_test_set(self, fold_index):
        #Itrain = np.hstack([self.Ifolds[j] for j in range(self.nfolds) if fold!=j])
        #Xtrain = self.X.iloc[Itrain, :]
        #ytrain = self.y[Itrain]
        #Itest = self.Ifolds[fold]
        #Xtest = self.X.iloc[Itest, :]
        #ytest = self.y[Itest]
        dtrain, dtest = self.train_and_test_fold(fold_index)
        return dtrain.X, dtrain.y, dtest.X, dtest.y

    def train_and_test_fold(self, fold_index, nfolds=None):
        fold = Fold(self, fold_index, nfolds=nfolds)

        mro = tuple(set(type(self).__mro__).intersection(
            {RegressionMixin, BinaryMixin, MulticlassMixin, Dataset}))
        assert type(self) not in mro

        # Create subtype to ensure presence of relevant mixin
        train_fold_type = type(f"{self.name()}_TrnFold{fold_index}",
                               (TrainFold,) + mro, {})
        train_fold = train_fold_type(fold)

        test_fold_type = type(f"{self.name()}_TstFold{fold_index}",
                               (TestFold,) + mro, {})
        test_fold = test_fold_type(fold)

        for f in self.task_fields():
            setattr(train_fold, f, getattr(self, f))
            setattr(test_fold, f, getattr(self, f))

        return train_fold, test_fold

    def astype(self, dtype):
        self.X = self.X.astype(dtype)
        self.y = self.y.astype(dtype)

    def _cached_hdf5_name(self):
        return os.path.join(self.data_dir, f"{self.name()}.h5")

    def hdf5_exists(self):
        h5file = self._cached_hdf5_name()
        return os.path.exists(h5file)

    def load_hdf5(self):
        h5file = self._cached_hdf5_name()
        print(f"loading cached {h5file}")
        X = pd.read_hdf(h5file, key="X")
        y = pd.read_hdf(h5file, key="y")
        return X, y

    def store_hdf5(self, X, y):
        h5file = self._cached_hdf5_name()
        X.to_hdf(h5file, key="X", complevel=9)
        y.to_hdf(h5file, key="y", complevel=9)

    def _load_openml(self, name, data_id, force=False):
        from sklearn.datasets import fetch_openml

        if not self.hdf5_exists() or force:
            print(f"loading {name} with fetch_openml")
            X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
            X, y = self._transform_X_y(X, y)
            X, y = self._cast_X_y(X, y)
            self.store_hdf5(X, y)
        else:
            X, y = self.load_hdf5()
        return X, y

    def _load_arff(self, name, force=False):
        from scipy.io.arff import loadarff

        if not self.hdf5_exists() or force:
            print(f"loading {name} from arff file")
            data = loadarff(os.path.join(self.data_dir, name))
            X, y = self._transform_X_y(data, None)
            X, y = self._cast_X_y(X, y)
            self.store_hdf5(X, y)
        else:
            X, y = self.load_hdf5()
        return X, y

    def _load_csv_gz(self, name, force=False, read_csv_kwargs={}):
        import gzip

        if not self.hdf5_exists() or force:
            print(f"loading {name} from gzipped csv file")
            with gzip.open(os.path.join(self.data_dir, name), "rb") as f:
                data = pd.read_csv(f, **read_csv_kwargs)
            X, y = self._transform_X_y(data, None)
            X, y = self._cast_X_y(X, y)
            self.store_hdf5(X, y)
        else:
            X, y = self.load_hdf5()
        return X, y

    def _download_data_temporarily(self, url):
        try:
            import requests
        except ModuleNotFoundError as e:
            print("To download UCI datasets, the `requests` is required")
            print("    `pip install requests`")
            raise e

        # TODO complete

    def _transform_X_y(self, X, y): # override if necessary
        return X, y

    def _cast_X_y(self, X, y):
        from sklearn.preprocessing import OrdinalEncoder
        if self.task != Task.REGRESSION and not np.isreal(y[0]):
            self.target_encoder = OrdinalEncoder(dtype=DTYPE)
            y = self.target_encoder.fit_transform(y.to_numpy().reshape(-1, 1))
            return X.astype(DTYPE), pd.Series(y.ravel())
        else:
            return X.astype(DTYPE), y.astype(DTYPE)

    def minmax_normalize(self):
        if self.X is None:
            raise RuntimeError("data not loaded")

        from sklearn import preprocessing

        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

    def robust_normalize(self, quantile_lo=.1, quantile_hi=.9):
        if self.X is None:
            raise RuntimeError("data not loaded")

        from sklearn import preprocessing

        X = self.X.values
        min_max_scaler = preprocessing.RobustScaler(
                quantile_range=(quantile_lo*100.0, quantile_hi*100.0))
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

    def scale_target(self, quantile_lo=.1, quantile_hi=.9):
        if self.X is None:
            raise RuntimeError("data not loaded")
        if not self.is_regression():
            return

        from sklearn import preprocessing

        min_max_scaler = preprocessing.RobustScaler(
                quantile_range=(quantile_lo*100.0, quantile_hi*100.0))
        y = self.y.values.reshape(-1, 1)
        y_scaled = min_max_scaler.fit_transform(y)
        df = pd.Series(y_scaled.ravel())
        self.y = df

    def transform_target(self):
        if self.X is None:
            raise RuntimeError("data not loaded")
        if not self.is_regression():
            return

        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=100)
        y = self.y.values.reshape(-1, 1)
        y_scaled = qt.fit_transform(y)
        df = pd.Series(y_scaled.ravel())
        self.y = df

    def paramgrid(self, **kwargs):
        import itertools

        param_lists = {k: (v if isinstance(v, list) else [v])
                       for k, v in kwargs.items()}
        param_names = list(param_lists.keys())

        for instance in itertools.product(*param_lists.values()):
            params = dict(zip(param_names, instance))
            yield params

    def train(self, model_class, params):
        if not isinstance(self, TrainFold):
            raise RuntimeError("train on a TrainFold "
                               "(use `Dataset.train_and_test_fold()`)")
        train_time = time.time()
        clf = model_class(**params)
        clf.fit(self.X, self.y)
        train_time = time.time() - train_time

        return clf, train_time

    def hyperparam(self, model_class, **kwargs):
        for params in self.paramgrid(**kwargs):
            yield self.train(model_class, params)

    def metric(self, *args):
        try:
            import veritas
            VERITAS_EXISTS = True
        except ModuleNotFoundError:
            VERITAS_EXISTS = False

        if len(args) == 1:
            ytrue = self.y
            ypred_or_clf, = args
            if isinstance(ypred_or_clf, np.ndarray):
                ypred = ypred_or_clf
            elif VERITAS_EXISTS and isinstance(ypred_or_clf, veritas.AddTree):
                at = ypred_or_clf
                if self.is_binary():
                    ypred = at.predict(self.X) > 0.5
                elif self.is_regression():
                    ypred = at.predict(self.X)
                else:
                    ypred = np.argmax(at.predict(self.X), axis=1)
            else:
                clf = ypred_or_clf
                try:
                    ypred = clf.predict(self.X)
                except AttributeError:
                    raise ValueError("metric(ypred:np.ndarray) or metric(clf)")
        elif len(args) == 2:
            ytrue, ypred = args
        else:
            raise ValueError()
        return self._metric(ytrue, ypred)
            
class Fold:
    def __init__(self, dataset, fold_index, nfolds=None):
        if not isinstance(dataset, Dataset):
            raise ValueError("Not a dataset")
        if isinstance(dataset, TestFold):
            print("Warning: fold on TestFold")

        self.nfolds = nfolds
        if nfolds is None:
            self.nfolds = dataset.nfolds

        if fold_index >= self.nfolds or fold_index < 0:
            raise IndexError(f"Invalid fold index: {fold_index} / {self.nfolds}")

        self.dataset = dataset
        self.fold_index = fold_index

        fold_size = dataset.perm.shape[0] / self.nfolds
        test_start = int(fold_index * fold_size)
        test_end = int((fold_index+1) * fold_size)

        self.perm_test = dataset.perm[test_start:test_end]
        self.perm_train = np.hstack((
            dataset.perm[:test_start],
            dataset.perm[test_end:]))
        
        self.Xtrain = dataset.X.loc[self.perm_train, :]
        self.ytrain = dataset.y.loc[self.perm_train]
        self.Xtest = dataset.X.loc[self.perm_test, :]
        self.ytest = dataset.y.loc[self.perm_test]

class TrainFold:
    def __init__(self, fold):
        self.parent = fold.dataset
        self.fold = fold

        # Dataset __init__
        super().__init__(self.parent.task,
                         nfolds=fold.nfolds - 1,
                         seed=self.parent.seed)

        # should set fields as `DataSet.load_dataset`

        self.perm = fold.perm_train
        self.X = fold.Xtrain
        self.y = fold.ytrain

    def load_dataset(self):
        raise RuntimeError("Cannot load TrainFold")

class TestFold:
    def __init__(self, fold):
        self.parent = fold.dataset
        self.fold = fold

        # Dataset __init__
        super().__init__(self.parent.task,
                         nfolds=1,
                         seed=self.parent.seed)

        self.perm = fold.perm_test
        self.X = fold.Xtest
        self.y = fold.ytest

    def load_dataset(self):
        raise RuntimeError("Cannot load TestFold")

class RegressionMixin:

    def to_binary(self, frac_positive=0.5, right=False):
        return self.to_multiclass([frac_positive], right)

    def to_multiclass(self, quantiles, right=False):
        assert self.are_X_y_set()
        quantiles = np.quantile(self.y, quantiles)
        y = np.digitize(self.y, quantiles, right=right)
        y = y.astype(DTYPE)

        num_classes = len(quantiles) + 1
        binary = num_classes == 2
        sup = (Dataset, BinaryMixin) if binary else (Dataset, MulticlassMixin)
        suffix = "BinClf" if binary else f"MultClf{num_classes}"
        task = Task.BINARY if binary else Task.MULTICLASS
        name = self.name() + suffix

        cls = type(name, sup, {})
        d = cls(task, nfolds=self.nfolds, seed=self.seed)
        d.num_classes = num_classes
        d.class_edges = quantiles
        d.regression = self

        # Simulate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        d.X = self.X
        d.y = pd.Series(y)
        d.perm = self.perm

        return d

    def _metric(self, ytrue, ypred): # lower is better
        from sklearn.metrics import root_mean_squared_error
        return root_mean_squared_error(ytrue, ypred)

    def metric_name(self):
        return "rmse"

    def get_model_class(self, model_type):
        if model_type == "xgb":
            import xgboost as xgb
            return xgb.XGBRegressor

        if model_type == "rf":
            import sklearn.ensemble
            return sklearn.ensemble.RandomForestRegressor

        if model_type == "lgb":
            import lightgbm as lgb
            return lgb.LGBMRegressor

        raise ValueError(f"Unknown model_type {model_type}")

    def task_fields(self):
        return []

class BinaryMixin:

    def _metric(self, ytrue, ypred): # higher is better
        from sklearn.metrics import accuracy_score
        return accuracy_score(ytrue, ypred)

    def metric_name(self):
        return "accuracy"

    def get_model_class(self, model_type):
        if model_type == "xgb":
            import xgboost as xgb
            return xgb.XGBClassifier

        if model_type == "rf":
            import sklearn.ensemble
            return sklearn.ensemble.RandomForestClassifier

        if model_type == "lgb":
            import lightgbm as lgb
            return lgb.LGBMClassifier

        raise ValueError(f"Unknown model_type {model_type}")

    def task_fields(self):
        return []

class MulticlassMixin:

    def one_vs_other(self, class1, class2):
        assert self.is_multiclass()

        if class1 not in range(self.num_classes):
            raise ValueError("invalid class1")
        if class2 not in range(self.num_classes):
            raise ValueError("invalid class2")
        if class1 >= class2:
            raise ValueError("take class1 < class2")

        mask = (self.y == class1) | (self.y == class2)
        def class1_predicate(y):
            return y == class2

        d = self.to_binary(f"{class1}v{class2}", class1_predicate, mask)
        d.class1 = class1
        d.class2 = class2
        return d

    def one_vs_rest(self, class1):
        assert self.is_multiclass()

        if class1 not in range(self.num_classes):
            raise ValueError("invalid class1")

        def class1_predicate(y):
            return y == class1

        d = self.to_binary(f"{class1}vRest", class1_predicate)
        d.class1 = class1

        return d

    def multi_vs_rest(self, suffix, trueclasses):
        assert self.is_multiclass()

        N = self.X.shape[0]
        num_classes = self.num_classes

        def class1_predicate(y):
            ynew = np.zeros(N, dtype=DTYPE)
            for c in trueclasses:
                if c not in range(num_classes):
                    raise ValueError(f"invalid class {c}")
                ynew[y==c] = 1.0
            return pd.Series(ynew)

        d = self.to_binary(suffix, class1_predicate)
        d.trueclasses = trueclasses

        return d

    def lessthan_c_vs_rest(self, c):
        return self.multi_vs_rest(f"Lt{c}", range(c))

    def to_binary(self, suffix, class1_predicate, mask=None):
        assert self.is_multiclass()

        task = Task.BINARY
        cls = type(f"{self.name()}{suffix}", (Dataset, BinaryMixin), {})
        d = cls(task, nfolds=self.nfolds, seed=self.seed)
        d.multiclass = self
        d.mask = mask

        # Similate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        if mask is not None:
            d.X = self.X[mask].reset_index(drop=True)
            d.y = class1_predicate(self.y.loc[mask].reset_index(drop=True))
            d.load_dataset() # reset self.perm
        else:
            d.X = self.X
            d.y = class1_predicate(self.y)
            d.perm = self.perm # reuse permutation

        d.y = d.y.astype(DTYPE)

        return d

    def _metric(self, ytrue, ypred): # higher is better
        from sklearn.metrics import accuracy_score
        return accuracy_score(ytrue, ypred)

    def metric_name(self):
        return "accuracy"

    def get_model_class(self, model_type):
        if model_type == "xgb":
            import xgboost as xgb
            return xgb.XGBClassifier

        if model_type == "rf":
            import sklearn.ensemble
            return sklearn.ensemble.RandomForestClassifier

        if model_type == "lgb":
            import lightgbm as lgb
            return lgb.LGBMClassifier

        raise ValueError(f"Unknown model_type {model_type}")

    def task_fields(self):
        return ["num_classes"]

