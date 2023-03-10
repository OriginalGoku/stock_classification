"""
Optuna example that demonstrates a pruner for XGBoost.


We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python xgboost_integration.py

"""

import numpy as np
import optuna
import pandas as pd

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tqdm import tqdm
from file_utility import FileUtility
from load_file import YahooDataLoader
from load_batch_data import BatchDataLoader
from line_printer import LinePrinter

from optuna_pruning_sklearn import StopWhenTrialKeepBeingPrunedCallback
from sklearn.multiclass import OneVsRestClassifier

import joblib



class OptunaOptimizer:
    def __init__(self, batch_size, interval):

        self.rounding_precision = 4
        self.test_percent = 0.25
        #self.data_path = '..\Data_Source\Yahoo\Processed_Yahoo_Data\Stock_Binary_tolerance_half_std\ETF'
        self.data_path = '../Data_Source/Yahoo/Processed_Yahoo_Data/Stock_Binary_tolerance_half_std/ETF'
        #self.data_path = '../Data_Source/Dropbox'

        self.sentence_length = 31
        self.batch_size = batch_size

        self.pruning_threshold = 5
        self.load_positive_actions = True

        self.pruner = StopWhenTrialKeepBeingPrunedCallback(self.pruning_threshold)

        self.file_utility_input = {'source_data_path': self.data_path,
                                   'save_destination_path': 'results',
                                   'file_formats_to_load': 'csv',
                                   'file_format_to_save': 'csv',
                                   'verbose': True
                                   }
        batch_data_loader_input = {'batch_size': self.batch_size,
                                   'sentence_length': self.sentence_length,
                                   'include_volatility': True,
                                   'data_path': self.data_path,
                                   'file_utility_input': self.file_utility_input,
                                   'intervals': interval
                                   }
        self.batch_data_loader = BatchDataLoader(**batch_data_loader_input)
        self.data = None
        self.target = None
        self.line_printer = LinePrinter()

    def load_data(self, fetch_randomize_data):

        if fetch_randomize_data:
            #self.line_printer.print_text("We are HERE ")
            data = self.batch_data_loader.fetch_batch_randomized(self.load_positive_actions)
        else:
            data, done = self.batch_data_loader.fetch_batch(self.load_positive_actions)


        # data_zero_and_one = data
        #data_zero_and_one.loc[data_zero_and_one['action'] == -1, 'action'] = 2
        # final_data = data_zero_and_one[data_zero_and_one.columns[:-2]]
        # target = data_zero_and_one.action

        self.data = data[data.columns[:-2]]
        self.target = data['action']

        # print(self.target.groupby(self.target).size())


        # return final_data, target


    def random_forest_objective(self, trial):
        #self.load_data(True)
        train_x, valid_x, train_y, valid_y = train_test_split(self.data, self.target, test_size=0.25)
        param = {
            #"verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 150, 200, log=True),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 2, 150, log=True),

            #"min_samples_split": trial.suggest_int("min_samples_split", 10, 20, log=True),
            #"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, log=True),
            #"bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "bootstrap": False,
            #"ccp_alpha": trial.suggest_float("ccp_alpha", 0.01, 1, log=True),
            #"eval_metric": "auc",
        }
        #if param["bootstrap"]:
         #   param["max_samples"] = trial.suggest_float("max_samples", 0.01, 0.99, log=True)

        model = OneVsRestClassifier(RandomForestClassifier(**param))

        fitted_model = model.fit(train_x, train_y)
        y_pred = model.predict(valid_x)

        model_y_score = fitted_model.predict_proba(valid_x)
        # print("len model_y_score: ", len(model_y_score))
        # print("Model_y_score")

        #model_y_score_df = pd.DataFrame(model_y_score)



        # print('valid_y: ', valid_y)
        # print("model_y_score_df[model_y_score_df[2]>0.6].index: ", model_y_score_df[model_y_score_df[1]>0.6].index.values)
        # # print(valid_y.reset_index(drop=True))
        # model_y_score_df['action']=valid_y.reset_index(drop=True).iloc[model_y_score_df[model_y_score_df[1]>0.6].index]
        # print(model_y_score_df[model_y_score_df[1]>0.6])
        # print('=====')
        # print(valid_y.loc[model_y_score_df.[model_y_score_df[1]>0.6].index])

        # print('model_y_score: ', model_y_score)
        report = classification_report(valid_y.to_numpy(), y_pred, output_dict=True, digits=self.rounding_precision)

        # target_names = self.target_names,
        report_accuracy = report['accuracy']
        #f1_score = 0
        #for i in range(0, 2):
         #   f1_score += report[str(i)]['f1-score']

        #return report_accuracy, f1_score

        trial.set_user_attr(key="best_booster", value=model)

        return report_accuracy

    def find_best_model_callback(self, study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])

    # FYI: Objective functions can take additional arguments
    # (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
    def xg_boost_objective(self, trial):
        # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
        # data, target = load_data()
        # print(self.data)
        self.load_data(True)
        train_x, valid_x, train_y, valid_y = train_test_split(self.data, self.target, test_size=self.test_percent)
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

        #self.line_printer.print_text('Starting Optuna with data length: '+ str(len(self.data)))
        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
        accuracy = accuracy_score(valid_y, pred_labels)
        return accuracy

    # def random_forest_objective(self, trial):
    #
    #     self.load_data(True)
    #     train_x, valid_x, train_y, valid_y = train_test_split(self.data, self.target, test_size=0.25)
    #
    #     param = {
    #         "verbosity": 0,
    #         # "objective": "binary:logistic",
    #         "objective": "multi:softmax",
    #         "num_class": 3,
    #         "eval_metric": "auc",
    #         "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
    #         "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
    #         "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    #     }
    #
    #     model = RandomForestClassifier()

    def run_optuna(self, algorithm, n_trials):
        #study = optuna.create_study(directions=["maximize", "maximize"])
        #study = optuna.create_study(
            # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
        #)
        self.load_data(False)
        if (algorithm == 'gx_boosx_objective'):
            study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
                 )
            print("XGBoost_Objective")
            study.optimize(self.xg_boost_objective, n_trials=n_trials)
            print("Number of finished trials: ", len(study.trials))
            trial = study.best_trial
            print("Best trial:")

            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

        if (algorithm == 'Random Forest'):

            print("Starting Random Forest Optimizer")
            study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize"
            )
            study.optimize(self.random_forest_objective, n_trials=n_trials, callbacks=[self.find_best_model_callback],
                           gc_after_trial=True)
            best_model = study.user_attrs["best_booster"]
            joblib.dump(best_model, 'models/Best_Random_Forest_Model.joblib')

            fig = optuna.visualization.plot_param_importances(study)
            fig.show()

            #print("Number of finished trials: ", len(study.trials))
            trial = study.best_trial
            print("Best trial:")

            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))




        # Multiple output
        # if (algorithm == 'Random Forest'):
        #     print("Random Forest")
        #     study.optimize(self.random_forest_objective, n_trials=n_trials, callbacks=[self.pruner],
        #                    show_progress_bar=True)
        #     print("Number of finished trials: ", len(study.trials))
        #     trial = study.best_trials
        #     fig = optuna.visualization.plot_pareto_front(study, target_names=["accuracy", "f1_score"])
        #     fig.show()
        #
        #     fig_2 = optuna.visualization.plot_param_importances(
        #         study, target=lambda t: t.values[0], target_name="accuracy"
        #     )
        #     fig_2.show()
        #     fig_3 = optuna.visualization.plot_param_importances(
        #         study, target=lambda t: t.values[1], target_name="f1_score"
        #     )
        #     fig_3.show()
        #
        #
        #
        #     trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
        #     print(f"Trial with highest accuracy: ")
        #     print(f"\tnumber: {trial_with_highest_accuracy.number}")
        #     print(f"\tparams: {trial_with_highest_accuracy.params}")
        #     print(f"\tvalues: {trial_with_highest_accuracy.values}")
        #
        #     trial_with_highest_f1_score = max(study.best_trials, key=lambda t: t.values[1])
        #     print(f"f1_score with highest accuracy: ")
        #     print(f"\tnumber: {trial_with_highest_f1_score.number}")
        #     print(f"\tparams: {trial_with_highest_f1_score.params}")
        #     print(f"\tvalues: {trial_with_highest_f1_score.values}")
        #
        #     # save data:
        #     joblib.dump(study, 'Random_Forest_'+str(n_trials)+'.pkl')
        #
        #     # load data:
        #     # loaded_data = joblib.load(...file name)

if __name__ == "__main__":
    # data_path = '../Data_Source/Yahoo/Processed_Yahoo_Data/Stock_Binary_tolerance_half_std/ETFs'
    optuna_optimizer = OptunaOptimizer(20000, 4)
    # optuna_optimizer.load_data()
    optuna_optimizer.run_optuna("Random Forest", n_trials=100)

