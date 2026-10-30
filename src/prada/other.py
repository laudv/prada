from .dataset import Binary, Regression

import os
import numpy as np
import pandas as pd

# https://github.com/laudv/veritas/raw/c1808327b285facd112f651397552a54f90a6a4b/tests/data/img.npy
class Img(Regression):
    dataset_name = "img.h5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(self.data_dir, Img.dataset_name)
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            super().load_dataset()

    def read_from_img(self, fname):
        import imageio
        img = imageio.imread(fname)
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])

        df = pd.DataFrame(X, columns=["a0", "a1"])
        dfy = pd.Series(y)

        data_path = os.path.join(self.data_dir, Img.dataset_name)
        df.to_hdf(data_path, key='X', mode='w') 
        dfy.to_hdf(data_path, key='y', mode='a') 


# https://github.com/tailequy/fairness_dataset/tree/main/Dutch_census
class DutchCensus(Binary):
    dataset_name = "dutch_census_2001.arff"
    dataset_url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/4d225df30598951985c31d81e1544dba504286a8/Dutch_census/dutch_census_2001.arff"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_arff(DutchCensus.dataset_name, force, url=DutchCensus.dataset_url)
            super().load_dataset()

    def _transform_X_y(self, data, _):
        df = pd.DataFrame(data[0])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode("utf-8")


        y = (df["occupation"] == "2_1")
        X = df.drop(columns=["occupation"])

        # Binary
        X["male"] = (X["sex"] == "1")

        # Age groups (1–17) and education level (0–5, map unknown 9→0): keep as numeric
        X["age"] = X["age"].astype(int)
        X["edu_level"] = X["edu_level"].replace("9", "0").astype(int)

        # Marital status: manual ordinal (same logic as Adult/CensusIncomeKDD)
        X["Marital_status"] = \
            (X["Marital_status"] == "2") * 3.0 \
            + (X["Marital_status"] == "1") * 2.0 \
            + (X["Marital_status"] == "3") * 1.0
        # widowed (4) -->  0

        X = pd.get_dummies(X, columns=[
            "household_position", "household_size",
            "prev_residence_place", "citizenship", "country_birth",
            "economic_status", "cur_eco_activity",
        ], drop_first=True)

        X.drop(columns=["sex"], inplace=True)

        return X, y


# https://github.com/tailequy/fairness_dataset/tree/main/Law_school
class LawSchool(Binary):
    dataset_name = "law_dataset.arff"
    dataset_url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/4d225df30598951985c31d81e1544dba504286a8/Law_school/law_dataset.arff"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_arff(LawSchool.dataset_name, force, url=LawSchool.dataset_url)
            super().load_dataset()

    def _transform_X_y(self, data, _):
        df = pd.DataFrame(data[0])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode("utf-8")

        y = (df["pass_bar"] == "1")
        X = df.drop(columns=["pass_bar"])

        # Categorical cols declared in ARFF --> convert to numeric
        X["racetxt"] = X["racetxt"].astype(int)   # protected: 1=white, 0=non-white
        X["tier"] = X["tier"].astype(int)         # ordinal: 1=best tier, 6=worst

        # fulltime: 1.0=full-time, 2.0=part-time → binary
        X["fulltime"] = (X["fulltime"] == 1.0)

        # male, decile1b, decile3, lsat, ugpa, zfygpa, zgpa, fam_inc: already numeric
        return X, y
