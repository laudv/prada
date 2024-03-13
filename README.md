# *PraDa* :gem: : A fine selection of **Pra**ctically useful tabular **Da**tasets

*PraDa* automatically downloads (only OpenML for now) and locally stores its fine
selection of datasets on your computer in the HDF5 format for faster loading
times.

Use this environment variable to inform *PraDa* where to store these HDF5
datasets:

```sh
export PRADA_DATA_DIR=/your/data/cache/directory
```

## Install

*PraDa* is currently not a PyPI package, but you can install it straight from
Github (https://pip.pypa.io/en/stable/topics/vcs-support/)
```sh
pip install git+https://github.com/laudv/prada.git
```

## Loading a dataset

List all the datasets:
```python
import prada
prada.ALL_DATASETS
prada.ALL_DATASET_NAMES
prada.ALL_BINARY
prada.ALL_REGRESSION
prada.ALL_MULTICLASS
```

Load a dataset using the class name:
```python
d = prada.Phoneme()
d.load_dataset()

d.X.shape # (5404, 5)
d.y.shape # (5404,)
```

Load a dataset using a name of a dataset:
```python
d = prada.get_dataset("Phoneme")
```

Load a multiclass dataset and turn it into a binary dataset by comparing only
two classes:
```python
d = prada.Mnist()
d.load_dataset()
d2v4 = d.one_vs_other(2, 4)

# or

d = prada.get_dataset("Mnist[2v4]")
```

Similarly, turn a regression dataset into a (binary) classification dataset:
```python
d = prada.get_dataset("WineQuality[bin]")

# or

d = prada.WineQuality()
d.load_dataset()
d.to_binary(frac_positive=0.5)
```

For more of these functions, have a look at the `RegressionMixin` and
`MulticlassMixin` mixins.


## Hyper-parameter optimization

Iterate over a grid of parameters:
```python
d = prada.Spambase()
d.load_dataset()

param_dict = {"n_estimators": [10, 20], "eta": [0.5, 0.9] }
for i, params in enumerate(d.paramgrid(**param_dict)):
    print(i, params)
```

This prints:
```
0 {'n_estimators': 10, 'eta': 0.5}
1 {'n_estimators': 10, 'eta': 0.9}
2 {'n_estimators': 20, 'eta': 0.5}
3 {'n_estimators': 20, 'eta': 0.9}
```

Train a model for a given parameter set:
```python
dtrain, dtest = d.train_and_test_fold(fold)
dtrain, dvalid = dtrain.train_and_test_fold(0)

# `model_class` can be any sklearn compatible classifier.
# There is built-in support for
#   - rf:  sklearn RandomForest
#   - xgb: xgboost
#   - lgb: lightgbm
model_type = "xgb" # or "rf", "lgb"
model_class = d.get_model_class(model_type)
clf, train_time = dtrain.train(model_class, params)

mtrain = dtrain.metric(clf)
mtest  = dtest.metric(clf)
mvalid = dvalid.metric(clf)
```


## Utility functions

```python
d = prada.Banknote(nfolds=5, seed=5232, silent=True)

d.name() # Banknote

d.source    # openml
d.openml_id # only for openml datasets
d.url       # only for openml datasets

d.is_regression() # False
d.is_binary()     # True
d.is_multiclass() # False
d.astype(np.float32) # cast d.X and d.y
d.minmax_normalize() # sklearn.MinMaxScaler
d.robust_normalize() # sklearn.RobustScaler
d.scale_target() # for regression problems

# Metric: RMSE for regression, Accuracy for classification
# either evaluates a given classifier on `d.X` ...
d.metric(clf: sklearn_compatible)
d.metric(at: veritas.AddTree)
# ... or just applies the relevant metric to the given values
d.metric(ytrue, ypred)
```

## Using it a `click` command line interface

```python
import prada
import click

@click.group()
def cli():
    pass

@cli.command("my_command")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb"]),
              default="rf")
@click.option("--fold", default=0)
@click.option("--nfolds", default=5)
@click.option("--relerr", default=0.01)
@click.option("--seed", default=123456)
def test_idea_cmd(dname, model_type, fold, nfolds, seed):
    d = prada.get_dataset(dname, nfolds=nfolds, seed=seed)
    d.load_dataset()
    d.robust_normalize()
    d.scale_target() # only for regression datasets

    # Do what you need to do here...

if __name__ == "__main__":
    cli()
```
