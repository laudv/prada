import os
import pandas as pd
import numpy as np

from .dataset import \
    Dataset, \
    Task, \
    MulticlassDataset, \
    MultiBinClassDataset, \
    OneVsAllDataset

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
        X = pd.get_dummies(X, columns=["occupation", "relationship", "race"],
                           drop_first=True)
        X.drop(inplace=True, columns=["education", "sex", "native-country"])
        y = (y == ">50K")
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("adult", data_id=179)
            self.minmax_normalize()
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

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        #params["subsample"] = 0.25
        #params["colsample_bytree"] = 0.5
        return params

    def rf_params(self):
        params = Dataset.rf_params(self)
        #Lori - additional params from GROOT paper
        params["min_samples_split"] = 10
        params["min_samples_leaf"] = 5

        return params

    def groot_params(self):
        params = Dataset.groot_params(self)
        #Lori - additional params from GROOT paper
        params["min_samples_split"] = 10
        params["min_samples_leaf"] = 5

        return params

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

class EMnist(MulticlassDataset):
    def __init__(self, **kwargs):
        super().__init__(num_classes=47, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("emnist", data_id=41039)
            super().load_dataset()

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        return params

class MnistLt5(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("mnist", data_id=554)
            self.y = self.y < 5
            super().load_dataset()

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
        params["subsample"] = 0.4
        params["colsample_bytree"] = 0.5
        return params

class FashionMnistKvAll(OneVsAllDataset):
    def __init__(self, class1, **kwargs):
        super().__init__(FashionMnist(), class1, **kwargs)

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        return params

class FashionMnistLt5(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("fashion_mnist", data_id=40996)
            self.y = self.y<5
            super().load_dataset()

    def name(self):
        return f"{super().name()}"

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
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, Webspam.dataset_name)
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            self.minmax_normalize()
            super().load_dataset()

class Higgs(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(self.data_dir, "higgs.h5")
            self.X = pd.read_hdf(higgs_data_path, "X")
            self.y = pd.read_hdf(higgs_data_path, "y")
            self.minmax_normalize()
            super().load_dataset()

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        return params

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

class Electricity(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        y = (y == "UP") # y values are in ['UP', 'DOWN'] -> transform to binary
        return X, y

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["subsample"] = 0.6
        return params

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("electricity", data_id=151)
            self.minmax_normalize()
            super().load_dataset()

class Banknote(Dataset):
    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def _transform_X_y(self, X, y):
        y = (y == "2") # y values are in ['1', '2'] -> transform to binary
        return X, y

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml("banknote-authentication", data_id=1462)
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

class DryBean(MulticlassDataset):
    dataset_name = "Dry_Bean_Dataset.arff"

    def __init__(self, **kwargs):
        super().__init__(num_classes=7, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_arff(DryBean.dataset_name)
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

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        #params["subsample"] = 0.5
        #params["colsample_bytree"] = 0.5
        return params

# https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis
class SensorlessDriveDiagnosis(MulticlassDataset):
    dataset_name = "Sensorless_drive_diagnosis.txt.gz"

    def __init__(self, **kwargs):
        super().__init__(num_classes=11, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                    SensorlessDriveDiagnosis.dataset_name,
                    read_csv_kwargs={"sep": " ", "header": None})
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        X = data.iloc[:, 0:-1]
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = data.iloc[:, -1]

        vmap = {v: i for i, v in enumerate(sorted(y.unique()))}
        y = pd.Series([vmap[x] for x in y], name="class")

        return X, y

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        #params["subsample"] = 0.5
        #params["colsample_bytree"] = 0.5
        return params

class SensorlessDriveDiagnosisLt6(Dataset):
    dataset_name = "Sensorless_drive_diagnosis.txt.gz"

    def __init__(self, **kwargs):
        super().__init__(Task.CLASSIFICATION, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                    SensorlessDriveDiagnosis.dataset_name,
                    read_csv_kwargs={"sep": " ", "header": None})
            self.y = self.y<6
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        X = data.iloc[:, 0:-1]
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = data.iloc[:, -1]

        return X, y

    def name(self):
        return f"{super().name()}"

# https://archive.ics.uci.edu/dataset/186/wine+quality
class WineQuality(MulticlassDataset):
    dataset_name = "WineQuality.csv.gz"

    def __init__(self, **kwargs):
        super().__init__(num_classes=7, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                    WineQuality.dataset_name)
            super().load_dataset()
            self.minmax_normalize()

    def _transform_X_y(self, data, _target_still_in_data):
        quality = data["quality"]
        qvalmap = {v: i for i, v in enumerate(sorted(quality.unique()))}
        y = pd.Series([qvalmap[x] for x in quality], name="quality")
        is_red = data["Type"] == "Red Wine"
        data.drop(inplace=True, columns=["Unnamed: 0", "Type", "quality"])
        data["IsRed"] = is_red
        X = data.astype(float)

        return X, y

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        #params["subsample"] = 0.5
        #params["colsample_bytree"] = 0.5
        return params

# https://archive.ics.uci.edu/dataset/59/letter+recognition
class LetterRecognition(MulticlassDataset):
    dataset_name = "letter-recognition.data.gz"

    def __init__(self, **kwargs):
        super().__init__(num_classes=26, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                    LetterRecognition.dataset_name,
                    read_csv_kwargs={"header": None})
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

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        #params["subsample"] = 0.5
        #params["colsample_bytree"] = 0.5
        return params

# https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
class PenDigits(MulticlassDataset):
    dataset_name = "pendigits.csv.gz"

    def __init__(self, **kwargs):
        super().__init__(num_classes=10, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_csv_gz(
                    PenDigits.dataset_name,
                    read_csv_kwargs={"header": None})
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

    def xgb_params(self, task):
        params = Dataset.xgb_params(self, task)
        params["num_class"] = self.num_classes
        #params["subsample"] = 0.5
        #params["colsample_bytree"] = 0.5
        return params

class ImgClf(MulticlassDataset):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)

        self.buckets = np.linspace(0, 1, num_classes+1)[1:-1]

        from .regression import Img
        self.img = Img()

    def load_dataset(self):
        self.img.load_dataset()
        self.X, self.yreg = self.img.X, self.img.y
        self.thresholds = np.quantile(self.yreg, self.buckets)
        y = np.digitize(self.yreg, self.thresholds)
        self.y = pd.Series(y, name="intensity")
        super().load_dataset()

