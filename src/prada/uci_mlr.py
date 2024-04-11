from .dataset import Multiclass

import numpy as np
import pandas as pd

UCI_DATASETS = []

#def _create_uci_mlr(name, url, data, task, fields):
#    if task == Task.MULTICLASS:
#        sup = (Dataset, MulticlassMixin)
#    elif task == Task.REGRESSION:
#        sup = (Dataset, RegressionMixin)
#    else:
#        sup = (Dataset,)
#    cls = type(name, sup, {})
#
#    def __init__(self, *args, **kwargs):
#        super(cls, self).__init__(task, *args, **kwargs)
#        self.source = "uci"
#        self.url = url
#        self.data = data
#        for k, v in fields.items():
#            setattr(self, k, v)
#
#    def load_dataset(self):
#        if self.X is not None and self.y is not None:
#            return
#
#
#        #self.X, self.y = self._load_openml(self.name(), self.openml_id)
#
#        super(cls, self).load_dataset()
#
#    cls.__init__ = __init__
#    cls.load_dataset = load_dataset
#
#    UCI_DATASETS.append(cls)
#
#    return cls

# https://archive.ics.uci.edu/datasets?search=Dry%20Bean%20Dataset
class DryBean(Multiclass):
    dataset_name = "Dry_Bean_Dataset.arff"

    def __init__(self, *args, **kwargs):
        super().__init__(7, *args, **kwargs)

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_arff(DryBean.dataset_name, force)
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        num_feat = len(data[0].dtype) - 1
        num_ex = len(data[0])

        X = np.zeros((num_ex, num_feat), dtype=np.float32)

        print(data[0].shape)
        for k in range(num_feat):
            X[:, k] = [v[k] for v in data[0]]
        y = [v[num_feat] for v in data[0]]

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        columns = [x for x in data[1]][0:num_feat]
        X = pd.DataFrame(X, columns=columns)
        y = pd.Series(y)

        return X, y

# https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis
class SensorlessDriveDiagnosis(Multiclass):
    dataset_name = "Sensorless_drive_diagnosis.txt.gz"

    def __init__(self, *args, **kwargs):
        super().__init__(11, *args, **kwargs)

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                SensorlessDriveDiagnosis.dataset_name,
                read_csv_kwargs={"sep": " ", "header": None},
                force=force
            )
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        X = data.iloc[:, 0:-1]
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = data.iloc[:, -1]

        vmap = {v: i for i, v in enumerate(sorted(y.unique()))}
        y = pd.Series([vmap[x] for x in y], name="class")

        return X, y

# https://archive.ics.uci.edu/dataset/59/letter+recognition
class LetterRecognition(Multiclass):
    dataset_name = "letter-recognition.data.gz"

    def __init__(self, *args, **kwargs):
        super().__init__(26, *args, **kwargs)

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                LetterRecognition.dataset_name,
                read_csv_kwargs={"header": None},
                force=force
            )
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        letters = data.iloc[:, 0]
        lmap = {v: i for i, v in enumerate(sorted(letters.unique()))}
        y = pd.Series([lmap[x] for x in letters], name="letter")
        data.drop(inplace=True, columns=[0])
        X = data.astype(float)
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        return X, y

# https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
class PenDigits(Multiclass):
    dataset_name = "pendigits.csv.gz"

    def __init__(self, *args, **kwargs):
        super().__init__(10, *args, **kwargs)

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                PenDigits.dataset_name,
                read_csv_kwargs={"header": None},
                force=force
            )
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        data.columns = [f"f{i}" for i in range(data.shape[1])]
        digits = data.iloc[:, -1]
        dmap = {v: i for i, v in enumerate(sorted(digits.unique()))}
        y = pd.Series([dmap[x] for x in digits], name="digits")
        data.drop(inplace=True, columns=[digits.name])
        X = data.astype(float)
        return X, y



# Anuran Calls https://archive.ics.uci.edu/dataset/406/anuran+calls+mfccs

# IDA2016 https://archive.ics.uci.edu/dataset/414/ida2016challenge

