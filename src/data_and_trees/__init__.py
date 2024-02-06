# ruff: noqa: F403
# ruff: noqa: F405

from .dataset import *

from .openml import *
from .other import *
from .uci_mlr import *


ALL_DATASETS = [cls for cls in Dataset.__subclasses__()]
ALL_DATASET_NAMES = [cls.__name__ for cls in ALL_DATASETS]

ALL_REGRESSION = [cls for cls in ALL_DATASETS if issubclass(cls, RegressionMixin)]
ALL_REGRESSION_NAMES = [cls.__name__ for cls in ALL_REGRESSION]

ALL_BINARY = [cls for cls in ALL_DATASETS if issubclass(cls, BinaryMixin)]
ALL_BINARY_NAMES = [cls.__name__ for cls in ALL_BINARY]

ALL_MULTICLASS = [cls for cls in ALL_DATASETS if issubclass(cls, MulticlassMixin)]
ALL_MULTICLASS_NAMES = [cls.__name__ for cls in ALL_MULTICLASS]


def _get_dname_suggestion(dname):
        from difflib import SequenceMatcher
        import numpy as np

        distances = [SequenceMatcher(None, dname, sugg).ratio()
                     for sugg in ALL_DATASET_NAMES]
        return ALL_DATASET_NAMES[np.argmax(distances)]

def get_dataset(dname, *args, **kwargs):
    import inspect

    try:
        cls = globals()[dname]
    except KeyError:
        cls = None

    if inspect.isclass(cls) and issubclass(cls, Dataset):
        return cls(*args, **kwargs)
    else:
        dname_suggestion = _get_dname_suggestion(dname)

        print()
        print("----------------------------------------")
        print(f"Invalid dataset `{dname}`. Did you mean `{dname_suggestion}`?")
        print("Valid dataset names are:", end="\n\n")
        print("REGRESSION: ", ", ".join(ALL_REGRESSION_NAMES), end="\n\n")
        print("BINARY: ", ", ".join(ALL_BINARY_NAMES), end="\n\n")
        print("MULTICLASS: ", ", ".join(ALL_MULTICLASS_NAMES), end="\n\n")
        print("----------------------------------------")

        raise ValueError(f"Invalid dataset `{dname}`. Did you mean {dname_suggestion}?")

