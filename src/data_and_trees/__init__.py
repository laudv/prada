# ruff: noqa: F403
# ruff: noqa: F405

from .dataset import *

from .openml import *
from .other import *
from .uci_mlr import *
from .libsvm import *


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

def parse_dataset_name(dname_and_options):
    try:
        i0 = dname_and_options.index('[')
        i1 = dname_and_options.index(']')
    except ValueError: # substring not found
        return dname_and_options, None
    dataset_name = dname_and_options[0:i0]
    options_str = dname_and_options[i0+1:i1]

    return dataset_name, options_str

def derive_dataset(d, options_str):
    if options_str is None:
        return d

    # multiclass one-vs-rest transformation (e.g. Mnist[2v4])
    if 'vRest' in options_str:
        class1 = int(options_str[:-len("vRest")])
        d.load_dataset()
        return d.one_vs_rest(class1)

    # multiclass one-vs-other transformation (e.g. Mnist[2v4])
    if 'v' in options_str:
        class1, class2 = tuple(map(int, options_str.split('v')))
        d.load_dataset()
        return d.one_vs_other(class1, class2)

    # TODO multiclass multi_vs_rest

    # regression: turn into binary classification
    if 'bin' in options_str:
        assert d.is_regression()
        d.load_dataset()
        return d.to_binary()

    # TODO multiclass to_multiclass

    raise ValueError(f"invalid option string {options_str}")

def get_dataset(dname_and_options, *args, **kwargs):
    import inspect

    dname, options_str = parse_dataset_name(dname_and_options) 

    try:
        cls = globals()[dname]
    except KeyError:
        cls = None

    if inspect.isclass(cls) and issubclass(cls, Dataset):
        return derive_dataset(cls(*args, **kwargs), options_str)
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

