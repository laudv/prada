from .dataset import Dataset, Task
from .dataset import MulticlassMixin, RegressionMixin, BinaryMixin

import numpy as np
import pandas as pd

OPENML_DATASETS = []

# https://huggingface.co/datasets/inria-soda/tabular-benchmark

def _create_openml(name, openml_id, task, fields, transform_y=None):
    if task == Task.MULTICLASS:
        sup = (Dataset, MulticlassMixin)
    elif task == Task.REGRESSION:
        sup = (Dataset, RegressionMixin)
    else:
        sup = (Dataset, BinaryMixin)
    cls = type(name, sup, {})

    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(task, *args, **kwargs)
        self.source = "openml"
        self.openml_id = openml_id
        self.url = f"https://www.openml.org/d/{openml_id}"
        for k, v in fields.items():
            setattr(self, k, v)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_openml(self.name(), self.openml_id)
            super(cls, self).load_dataset()

    def _transform_X_y(self, X, y):
        if transform_y is not None:
            return X, transform_y(y)
        else:
            return X, y

    cls.__init__ = __init__
    cls.load_dataset = load_dataset
    cls._transform_X_y = _transform_X_y

    OPENML_DATASETS.append(cls)

    return cls


# -- REGRESSION -------------------------------------------------------------- #
for tup in [
        # Tabular benchmark versions
        ("CpuAct",          44132),
        ("Pol",             44133),
        ("Elevators",       44134),
        ("WineQuality",     44136),
        ("Ailerons",        44137),
        ("Yprop41",         45032),
        ("Houses",          44138), # calhouse
        ("House16H",        44139),
        ("Sulfur",          44145),
        ("MiamiHousing2016",44147),
        ("Superconduct",    44148),

        ("Topo21",                      45041),
        ("AnalCatData",                 44055), # http://www.stern.nyu.edu/~jsimonof/AnalCatData
        ("VisualizingSoil",             44056),
        ("DelaysZurich",                45045),
        ("Diamonds",                    44059),
        ("AllstateClaimsSeverity",      45046),
        ("MercedesBenzManufacturing",   44061),
        ("BrazilianHouses",             44062),
        ("BikeSharingDemand",           44063),
        ("AirlinesDelay1M",             45047),
        ("NycTaxi",                     44065),
        ("Abalone",                     45042),
        ("HouseSales",                  44066),
        ("Seattlecrime6",               45043),
        ("MedicalCharges",              45048),
        ("ParticulateMatterUkair17",    44068),
        ("SGEMM_GPU_kernelperf",        44069),

        # https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
        ("Year",    44027),
        ("Yolanda", 42705), # https://automl.chalearn.org/data
        ("CpuSmall", 227),

    ]:

    if len(tup) == 2:
        name, openml_id = tup
        transform_y = None
    else:
        name, openml_id, transform_y = tup

    cls = _create_openml(name, openml_id, Task.REGRESSION, {}, transform_y)
    globals()[name] = cls

# -- BINARY ------------------------------------------------------------------ #
for tup in [
        # Tabular benchmark versions
        ("Electricity",      44156),
        ("CovtypeNumeric",   44121),
        ("Covtype",          44159),
        #("Pol",              44122), REGRESSION
        ("MagicTelescope",   44125),
        ("BankMarketing",    44126),
        ("Bioresponse",      45019),
        ("MiniBooNE",        44128),
        ("DefaultCreditCardClients", 45036),
        ("HiggsBig",         44129),
        ("EyeMovements",     44157),
        ("Diabetes130US",    45022),
        ("Jannis",           45021),
        ("Heloc",            45026),
        ("Credit",           44089),
        ("California",       45028),
        ("Albert",           45035),
        ("CompasTwoYears",   45039),
        ("RoadSafety",       45038),

        # Other OpenML
        ("AtlasHiggs",       45549),
        ("SantanderCustomerSatisfaction", 45566),
        #("VehicleSensIt",    357),  # http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
        #("Prostate",         45672), # https://github.com/slds-lmu/paper_2023_ci_for_ge
        # https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
        ("CensusIncomeKDD",  42750),
        ("BreastCancer", 15, lambda y: y=="malignant"),
        ("Vehicle", 54, lambda y: (y == "bus") | (y == "van")),
        ("Spambase", 44, lambda y: (y == "1")),
        ("Phoneme", 1489, lambda y: (y == "1")),
        ("Nomao", 45078, lambda y: (y == 2.0)),
        ("Banknote", 1462, lambda y: (y == "1")),
    ]:

    if len(tup) == 2:
        name, openml_id = tup
        transform_y = None
    else:
        name, openml_id, transform_y = tup
    cls = _create_openml(name, openml_id, Task.BINARY, {})
    globals()[name] = cls

# -- MULTICLASS -------------------------------------------------------------- #
for name, openml_id, num_classes in [
        ("Mnist",        554,   10),
        ("EMnist",       41039, 47),

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
        ("FashionMnist", 40996, 10),

        # https://github.com/EpistasisLab/penn-ml-benchmarks/tree/master/datasets/classification/fars
        ("Fars",    40672, 8),
        ("Volkert", 41166, 10), # https://competitions.codalab.org/competitions/2321
    ]:
    fields = {"num_classes": num_classes}
    cls = _create_openml(name, openml_id, Task.MULTICLASS, fields)
    globals()[name] = cls




# -- DATASETS WITH TRANSFORMATIONS ------------------------------------------- #

def _Adult_transform_X_y(self, X, y):
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
    print(X, X.dtypes)
    print(y, y.dtype)
    return X, y
Adult = _create_openml("Adult", 179, Task.BINARY, {})
setattr(Adult, "_transform_X_y", _Adult_transform_X_y)

def _KddCup99_transform_X_y(self, X, y):
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
KddCup99 = _create_openml("KddCup99", 1113, Task.BINARY, {})
setattr(KddCup99, "_transform_X_y", _KddCup99_transform_X_y)

def _AmesHousing_transform_X_y(self, X, y):
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
AmesHousing = _create_openml("AmesHousing", 42165, Task.REGRESSION, {})
setattr(AmesHousing, "_transform_X_y", _AmesHousing_transform_X_y)
