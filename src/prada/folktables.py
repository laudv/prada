from .dataset import Binary

# https://github.com/zykls/folktables
# Ding et al., "Retiring Adult: New Datasets for Fair Machine Learning", NeurIPS 2021
#
# All tasks use California (CA) 2018 data, all races included.
# TODO: can be expanded to other years/states

ACS_SURVEY_YEAR = "2018"
ACS_STATES = ["CA"]


def _load_folktables(self, acs_task, force=False):
    if not self.hdf5_exists() or force:
        from folktables import ACSDataSource
        if not self.silent:
            print(f"loading {self.name()} from folktables "
                  f"(states={ACS_STATES}, year={ACS_SURVEY_YEAR})")
        data_source = ACSDataSource(
            survey_year=ACS_SURVEY_YEAR, horizon="1-Year", survey="person",
            root_dir=self.data_dir
        )
        acs_data = data_source.get_data(states=ACS_STATES, download=True)
        X, y, _ = acs_task.df_to_pandas(acs_data)
        y = y.squeeze()  # df_to_pandas returns y as single-column DataFrame
        X, y = self._transform_X_y(X, y)
        X, y = self._cast_X_y(X, y)
        self.store_hdf5(X, y)
    else:
        X, y = self.load_hdf5()
    return X, y


class ACSIncome(Binary):
    # Predict whether income >$50k (CA 2018).
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = "folktables"
        self.url = "https://github.com/zykls/folktables"

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            from folktables import ACSIncome as _ACSIncome
            self.X, self.y = _load_folktables(self, _ACSIncome, force)
            super().load_dataset()


class ACSEmployment(Binary):
    # Predict whether employed (CA 2018).
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = "folktables"
        self.url = "https://github.com/zykls/folktables"

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            from folktables import ACSEmployment as _ACSEmployment
            self.X, self.y = _load_folktables(self, _ACSEmployment, force)
            super().load_dataset()


class ACSPublicCoverage(Binary):
    # Predict whether has public health insurance (CA 2018).
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = "folktables"
        self.url = "https://github.com/zykls/folktables"

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            from folktables import ACSPublicCoverage as _ACSPublicCoverage
            self.X, self.y = _load_folktables(self, _ACSPublicCoverage, force)
            super().load_dataset()


class ACSMobility(Binary):
    # Predict whether moved in past year (CA 2018).
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = "folktables"
        self.url = "https://github.com/zykls/folktables"

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            from folktables import ACSMobility as _ACSMobility
            self.X, self.y = _load_folktables(self, _ACSMobility, force)
            super().load_dataset()


class ACSTravelTime(Binary):
    # Predict whether commute >20 min (CA 2018).
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = "folktables"
        self.url = "https://github.com/zykls/folktables"

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            from folktables import ACSTravelTime as _ACSTravelTime
            self.X, self.y = _load_folktables(self, _ACSTravelTime, force)
            super().load_dataset()
