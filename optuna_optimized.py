"""
Optuna example that demonstrates a pruner for XGBoost.


We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python xgboost_integration.py

"""

import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tqdm import tqdm
from file_utility import FileUtility
from load_file import YahooDataLoader
from line_printer import LinePrinter
from load_batch_data import BatchDataLoader

def load_data():
    data_path = '../Drop_Box/Dropbox'
    sentence_length = 31
    batch_size = 1000

    file_utility_input = ile_info = {'source_data_path': data_path,
                                     'save_destination_path': 'results',
                                     'file_formats_to_load': 'csv',
                                     'file_format_to_save': 'csv',
                                     'verbose': True
                                     }
    batch_data_loader_input = {'batch_size': batch_size,
                               'sentence_length': sentence_length,
                               'include_volatility': True,
                               'data_path': data_path,
                               'file_utility_input': file_utility_input,
                               'intervals': 4
                               }
    batch_data_loader = BatchDataLoader(**batch_data_loader_input)
    data, done = batch_data_loader.fetch_batch()
    # data_zero_and_one = data[(data['action']==1) | (data['action']==0)]
    data_zero_and_one = data
    data_zero_and_one.loc[data_zero_and_one['action'] ==-1,'action'] = 2
    final_data = data_zero_and_one[data_zero_and_one.columns[:-2]]
    target = data_zero_and_one.action

    return final_data, target

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    data, target = load_data()
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        # "objective": "binary:logistic",
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    accuracy = accuracy_score(valid_y, pred_labels)
    return accuracy


# if __name__ == "__main__":
if __name__ == "__main__":
    # data_path = '../Data_Source/Yahoo/Processed_Yahoo_Data/Stock_Binary_tolerance_half_std/ETFs'

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
