import os
import time
import copy
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder

try:
    DATA_DIR = os.environ["PRADA_DATA_DIR"]
except KeyError as e:
    print()
    print("| Environment variable PRADA_DATA_DIR not set. This is the")
    print("| path to the folder where PraDa stores the cached and")
    print("| compressed HDF5 files. Set the variable like this (unix):")
    print("|     `export PRADA_DATA_DIR=/path/to/data/dir`")
    print()
    raise e

SEED = 2537254
DTYPE = np.float32


class Dataset:
    def __init__(self, metric, seed=SEED, silent=False):
        self.data_dir = DATA_DIR
        self.seed = seed
        self.silent = silent
        self.source = "manual"
        self.url = "unknown"

        self.num_targets = -1
        self.num_classes = -1

        self.X = None
        self.y = None
        self.perm = None

        # (ytrue, ypredicted) -> metric value
        self._metric = metric
        self.metric_name = "?"

    def name(self):
        return type(self).__name__


    def is_regression(self):
        raise RuntimeError(f"undef in {self.name()} (subclass order!)")

    def is_classification(self):
        raise RuntimeError(f"undef in {self.name()} (subclass order!)")

    def is_binary(self):
        return self.is_classification() and self.num_classes == 2

    def is_multiclass(self):
        return self.is_classification() and self.num_classes > 2

    def is_multitarget_regression(self):
        return self.is_regression() and self.num_targets > 1

    def _metric_transform_y(self):
        # return self.y > 0.0
        return self.y

    def _at_predict(self, at):
        # return at.predict(self.X)
        raise RuntimeError("define in subclass")

    def _clf_predict(self, clf):
        # return clf.predict(self.X)
        raise RuntimeError("define in subclass")


    def is_data_loaded(self):
        return self.X is not None and self.y is not None and self.perm is not None

    def load_dataset(self):  # populate X, y, and perm
        if self.X is None or self.y is None:
            raise RuntimeError("override this and call after setting self.X and self.y")
        N = self.X.shape[0]

        rng = np.random.default_rng(self.seed)
        self.perm = rng.permutation(N)

    def train_and_test_set(self, fold_index_or_fraction, nfolds=None):
        dtrain, dtest = self.split(fold_index_or_fraction, nfolds)
        return dtrain.X, dtrain.y, dtest.X, dtest.y

    def train_and_test_fold(self, fold_index, nfolds):
        return self.split(fold_index, nfolds)

    def split(self, fold_index_or_fraction, nfolds=None):
        assert self.is_data_loaded()

        if isinstance(fold_index_or_fraction, float):
            assert nfolds is None, "fractional DataSplit does not take nfolds arg"
            fraction = fold_index_or_fraction
            self.test_fraction = fraction
            test_end = int(np.round(fraction * self.perm.shape[0]))
            self.perm_test = self.perm[:test_end]
            self.perm_train = self.perm[test_end:]

        else:
            fold_index = fold_index_or_fraction
            assert nfolds is not None, "fold DataSplit without nfolds"
            assert isinstance(fold_index, int), "fold_index must be int"
            assert isinstance(nfolds, int), "fold_index must be int"
            if fold_index >= nfolds or fold_index < 0:
                raise IndexError(f"Invalid fold index: {fold_index} / {nfolds}")

            fold_size = self.perm.shape[0] / nfolds
            test_start = int(fold_index * fold_size)
            test_end = int((fold_index + 1) * fold_size)

            self.fold_index = fold_index
            self.perm_test = self.perm[test_start:test_end]
            self.perm_train = np.hstack(
                (self.perm[:test_start], self.perm[test_end:])
            )

        dtrain = copy.copy(self)
        dtest = copy.copy(self)

        # These should be shallow copies:
        assert self.X is dtrain.X
        assert self.X is dtest.X

        dtrain.X = self.X.loc[self.perm_train, :]
        dtrain.y = self.y.loc[self.perm_train]
        dtrain.perm = self.perm_train

        dtest.X = self.X.loc[self.perm_test, :]
        dtest.y = self.y.loc[self.perm_test]
        dtest.perm = self.perm_test

        return dtrain, dtest

    def astype(self, dtype):
        self.X = self.X.astype(dtype)
        #self.y = self.y.astype(dtype)

    def _cached_hdf5_name(self):
        return os.path.join(self.data_dir, f"{self.name()}.h5")

    def hdf5_exists(self):
        h5file = self._cached_hdf5_name()
        return os.path.exists(h5file)

    def load_hdf5(self):
        h5file = self._cached_hdf5_name()
        if not self.silent:
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
            if not self.silent:
                print(f"loading {name} with fetch_openml")
            X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
            X, y = self._transform_X_y(X, y)
            #for k in X.columns:
            #    print(k)
            #    print(X[k].unique())
            #    print()
            #print(y.unique())
            X, y = self._cast_X_y(X, y)
            self.store_hdf5(X, y)
        else:
            X, y = self.load_hdf5()
        return X, y

    def _load_arff(self, name, force=False):
        from scipy.io.arff import loadarff

        if not self.hdf5_exists() or force:
            if not self.silent:
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
            if not self.silent:
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

    def _transform_X_y(self, X, y):  # override if necessary
        return X, y

    def _cast_X_y(self, X, y):
        if self.is_classification():
            return X.astype(DTYPE), y.astype(int)
        else:
            return X.astype(DTYPE), y.astype(DTYPE)
        #from sklearn.preprocessing import OrdinalEncoder

        #if self.task != Task.REGRESSION and not np.isreal(y[0]):
        #    self.target_encoder = OrdinalEncoder(dtype=DTYPE)
        #    y = self.target_encoder.fit_transform(y.to_numpy().reshape(-1, 1))
        #    return X.astype(DTYPE), pd.Series(y.ravel())
        #else:
        #    return X.astype(DTYPE), y.astype(DTYPE)

    def minmax_normalize(self):
        if self.X is None:
            raise RuntimeError("data not loaded")

        from sklearn import preprocessing

        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

    def robust_normalize(self, quantile_lo=0.1, quantile_hi=0.9):
        if self.X is None:
            raise RuntimeError("data not loaded")

        from sklearn import preprocessing

        X = self.X.values
        min_max_scaler = preprocessing.RobustScaler(
            quantile_range=(quantile_lo * 100.0, quantile_hi * 100.0)
        )
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

    def scale_target(self, quantile_lo=0.1, quantile_hi=0.9):
        if self.X is None:
            raise RuntimeError("data not loaded")
        if not self.is_regression():
            return

        from sklearn import preprocessing

        min_max_scaler = preprocessing.RobustScaler(
            quantile_range=(quantile_lo * 100.0, quantile_hi * 100.0)
        )
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

        param_lists = {
            k: (v if isinstance(v, list) else [v]) for k, v in kwargs.items()
        }
        param_names = list(param_lists.keys())

        for instance in itertools.product(*param_lists.values()):
            params = dict(zip(param_names, instance))
            yield params

    def train(self, model_class, params):
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
            ytrue = self._metric_transform_y()
            (ypred_or_clf,) = args
            if isinstance(ypred_or_clf, np.ndarray):
                ypred = ypred_or_clf
            elif VERITAS_EXISTS and isinstance(ypred_or_clf, veritas.AddTree):
                at = ypred_or_clf
                ypred = self._at_predict(at)
                #if self.is_binary():
                #    ypred = at.predict(self.X) > 0.5
                #elif self.is_regression():
                #    ypred = at.predict(self.X)
                #else:
                #    ypred = np.argmax(at.predict(self.X), axis=1)
            else:
                clf = ypred_or_clf
                try:
                    ypred = self._clf_predict(clf)
                except AttributeError:
                    raise ValueError("metric(ypred:np.ndarray) or metric(clf)")
        elif len(args) == 2:
            ytrue, ypred = args
        else:
            raise ValueError()
        return self._metric(ytrue, ypred)


class BalancedAccuracyMixin:
    def use_balanced_accuracy(self):
        self._metric = balanced_accuracy_score
        self.metric_name = "balanced_accuracy"


class ClassificationMixin(BalancedAccuracyMixin):
    def is_classification(self):
        return True

    def is_regression(self):
        return False

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

        if model_type == "dt":
            import sklearn.tree

            return sklearn.tree.DecisionTreeClassifier

        raise ValueError(f"Unknown model_type {model_type}")

    def as_regression_problem(self):
        assert self.is_data_loaded()

        suffix = "AsReg"
        name = self.name() + suffix

        if self.num_classes > 2:
            cls = type(name, (MulticlassAsMTRegr,), {})
            d = cls(self.num_classes, seed=self.seed, silent=self.silent)
            enc = OneHotEncoder(sparse_output=False, dtype=DTYPE)
            y = pd.DataFrame(enc.fit_transform(self.y.values.reshape(-1, 1)) * 2.0 - 1.0)
        else:
            cls = type(name, (BinaryAsRegr,), {})
            d = cls(seed=self.seed, silent=self.silent)
            y = pd.Series((self.y.values * 2.0 - 1.0).astype(DTYPE))

        # Simulate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        d.X = self.X
        d.y = y
        d.perm = self.perm

        d.classification = self

        return d


class Binary(ClassificationMixin, Dataset):
    def __init__(self, seed=SEED, silent=False):
        super().__init__(accuracy_score, seed, silent)
        self.threshold = 0.5
        self.num_classes = 2
        self.metric_name = "accuracy"

    def _at_predict(self, at):
        return at.predict(self.X) > self.threshold

    def _clf_predict(self, clf):
        return clf.predict_proba(self.X)[:, 1] > self.threshold


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
                ynew[y == c] = 1.0
            return pd.Series(ynew)

        d = self.to_binary(suffix, class1_predicate)
        d.trueclasses = trueclasses

        return d

    def lessthan_c_vs_rest(self, c):
        return self.multi_vs_rest(f"Lt{c}", range(c))

    def to_binary(self, suffix, class1_predicate, mask=None):
        assert self.is_multiclass()

        cls = type(f"{self.name()}{suffix}", (Binary,), {})
        d = cls(seed=self.seed, silent=self.silent)
        d.multiclass = self
        d.mask = mask

        # Similate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        if mask is not None:
            d.X = self.X[mask].reset_index(drop=True)
            d.y = class1_predicate(self.y.loc[mask].reset_index(drop=True))
            d.load_dataset()  # reset self.perm
        else:
            d.X = self.X
            d.y = class1_predicate(self.y)
            d.perm = self.perm  # reuse permutation

        d.y = d.y.astype(DTYPE)

        return d


class Multiclass(ClassificationMixin, MulticlassMixin, Dataset):
    def __init__(self, num_classes, seed=SEED, silent=False):
        super().__init__(accuracy_score, seed, silent)
        self.num_classes = num_classes
        self.metric_name = "accuracy"

    def _at_predict(self, at):
        return np.argmax(at.predict(self.X), axis=1)

    def _clf_predict(self, clf):
        return clf.predict(self.X)


class RegressionMixin:
    def is_regression(self):
        return True

    def is_classification(self):
        return False

    def to_binary(self, frac_positive=0.5, right=False):
        return self.to_multiclass([frac_positive], right)

    def to_multiclass(self, quantiles, right=False):
        assert self.is_data_loaded()
        quantiles = np.quantile(self.y, quantiles)
        y = np.digitize(self.y, quantiles, right=right)
        y = y.astype(DTYPE)

        num_classes = len(quantiles) + 1
        binary = num_classes == 2
        sup = (Binary,) if binary else (Multiclass,)
        suffix = "BinClf" if binary else f"MultClf{num_classes}"
        name = self.name() + suffix

        cls = type(name, sup, {})
        d = cls(seed=self.seed, silent=self.silent)
        d.num_classes = num_classes
        d.class_edges = quantiles
        d.regression = self

        # Simulate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        d.X = self.X
        d.y = pd.Series(y)
        d.perm = self.perm

        return d

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

        if model_type == "dt":
            import sklearn.tree

            return sklearn.tree.DecisionTreeRegressor

        raise ValueError(f"Unknown model_type {model_type}")

    def task_fields(self):
        return []


class Regression(RegressionMixin, Dataset):
    def __init__(self, seed=SEED, silent=False):
        super().__init__(root_mean_squared_error, seed, silent)
        self.num_targets = 1
        self.metric_name = "rmse"

    def _at_predict(self, at):
        return at.eval(self.X)

    def _clf_predict(self, clf):
        return clf.predict(self.X)


class MultiTargetRegression(RegressionMixin, Dataset):
    def __init__(self, num_targets, seed=SEED, silent=False):
        super().__init__(root_mean_squared_error, seed, silent)
        self.num_targets = num_targets
        self.metric_name = "rmse"

    def _at_predict(self, at):
        return at.eval(self.X)

    def _clf_predict(self, clf):
        return clf.predict(self.X)

    def to_argmax_multiclass(self):
        suffix = "ArgMaxMulticlass"
        cls = type(f"{self.name()}{suffix}", (Multiclass,), {})
        d = cls(self.num_targets, seed=self.seed, silent=self.silent)
        d.multitarget_regression = self

        # Similate load dataset
        # This needs to set the same fields as Dataset.load_dataset!
        d.X = self.X
        d.y = pd.Series(np.argmax(self.y, axis=1).astype(int))
        d.perm = self.perm  # reuse permutation

        return d


class BinaryAsRegr(RegressionMixin, BalancedAccuracyMixin, Dataset):
    def __init__(self, seed=SEED, silent=False):
        super().__init__(accuracy_score, seed, silent)
        self.num_targets = 2
        self.num_classes = 1
        self.metric_name = "accuracy"
        self.threshold = 0.0

    def _metric_transform_y(self):
        return self.y > 0.0

    def _at_predict(self, at):
        return at.eval(self.X) > 0.0

    def _clf_predict(self, clf):
        return clf.predict(self.X) > 0.0


class MulticlassAsMTRegr(RegressionMixin, BalancedAccuracyMixin, Dataset):
    def __init__(self, num_classes, seed=SEED, silent=False):
        super().__init__(accuracy_score, seed, silent)
        self.num_targets = num_classes
        self.num_classes = num_classes
        self.metric_name = "accuracy"

    def _metric_transform_y(self):
        return np.argmax(self.y, axis=1)

    def _at_predict(self, at):
        return np.argmax(at.eval(self.X), axis=1)

    def _clf_predict(self, clf):
        return np.argmax(clf.predict(self.X), axis=1)
