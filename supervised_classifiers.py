import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

import xgboost as xgb

# To save and load models
from joblib import dump, load

# from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from line_printer import LinePrinter
from collections import Counter

import os

global_run_counter = 0


def check_and_create_folder(path_):
    if not os.path.isdir(path_):
        os.makedirs(path_)


# Testing Git updates

class SupervisedClassifier:
    def __init__(self, actions: [int], action_description: [str], include_volatility: bool, model_path='models',
                 plot_path='plots', report_path='reports'):
        """
        Sequence in this class:
        1. call generate_test_train
        2. call generate_y_one_hot

        :param data_columns:
        :param no_of_actions:
        """
        check_and_create_folder(model_path)
        check_and_create_folder(plot_path)
        check_and_create_folder(report_path)

        self.model_pipeline = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_score = []
        self.y_onehot_test = None

        # self.model_names = ['MLP', 'Gradient Boosting', 'Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree',
        #                     'Gaussian', 'SVM','XGBoost']
        self.model_names = ['Random Forest']
        self.xgb_params = {'booster': 'gbtree', 'lambda': 3.12800432790444e-08, 'alpha': 1.9119332110454214e-06,
                           'max_depth': 7,
                           'eta': 0.39440064197237373, 'gamma': 0.11824872282374695, 'grow_policy': 'depthwise'}

        self.model_path = model_path
        self.plot_path = plot_path
        self.report_path = report_path
        # self.acc_list = []
        # self.auc_list = []
        self.accuracy_dic = {model: [] for model in self.model_names}
        # self.confusion_matrix_list = []

        self.n_classes = len(action_description)
        self.target_names = action_description
        self.actions = actions
        self.random_state = 123
        self.rounding_precision = 2
        self.include_volatility = include_volatility

        self.line_printer = LinePrinter()

    def generate_test_train(self, train_data, test_data):
        if ((len(train_data) == 0) | (len(test_data) == 0)):
            return False

        train_data_shuffled = train_data.copy()
        test_data_shuffled = test_data.copy()

        train_data_shuffled = train_data_shuffled.iloc[np.random.permutation(len(train_data_shuffled))]
        test_data_shuffled = test_data_shuffled.iloc[np.random.permutation(len(test_data_shuffled))]

        print("Generating test_train data")
        columns_to_include = -3
        if self.include_volatility:
            columns_to_include = -2
        X_train = train_data_shuffled[train_data_shuffled.columns[:columns_to_include]]
        X_test = test_data_shuffled[test_data_shuffled.columns[:columns_to_include]]
        y_train = train_data_shuffled['action']
        y_test = test_data_shuffled['action']

        # print(y_test)

        self.X_train, _, self.y_train, __ = train_test_split(X_train, y_train, test_size=0.01,
                                                             random_state=self.random_state)
        _, self.X_test, __, self.y_test = train_test_split(X_test, y_test, test_size=0.99,
                                                           random_state=self.random_state)

        self.y_onehot_test = self.generate_y_one_hot()
        # print('len(Counter(self.y_test): ', len(Counter(self.y_test)))
        has_enough_data = (len(Counter(self.y_test)) == self.n_classes)
        return has_enough_data

    def generate_y_one_hot(self):
        label_binarizer = LabelBinarizer().fit(self.y_train)
        y_onehot_test = label_binarizer.transform(self.y_test)
        return y_onehot_test

    def generate_pipeline(self):

        # self.model_pipeline.append(MLPClassifier(early_stopping=False))
        # self.model_pipeline.append(GradientBoostingClassifier())
        # self.model_pipeline.append(LogisticRegression(max_iter=2000))
        # self.model_pipeline.append(KNeighborsClassifier())
        self.model_pipeline.append(RandomForestClassifier(n_estimators=183, criterion='log_loss', max_depth=28,
                                                          bootstrap=True))

        # self.model_pipeline.append(DecisionTreeClassifier())
        # self.model_pipeline.append(GaussianNB())
        # self.model_pipeline.append(SGDClassifier(loss='log_loss'))

        # self.model_pipeline.append(xgb.Booster(**params))

        # self.model_pipeline.append(xgb.XGBClassifier(**self.xgb_params))
    def load_random_forest_and_evaluate(self):
        global global_run_counter
        self.line_printer.print_text("Loading Random Forest Model")
        model_name = 'Random Forest_Optimized'
        model = load('models/best_random_forest_model.joblib')
        y_pred = model.predict(self.X_test)
        # model_y_score = fitted_model.predict_proba(self.X_test)

        # print('model_y_score: ', model_y_score)
        report = classification_report(self.y_test.to_numpy(), y_pred, output_dict=True,
                                       digits=self.rounding_precision)
        # target_names = self.target_names,
        report_accuracy = report['accuracy']
        # print('report[accuracy]: ', round(report_accuracy, 2))
        # self.accuracy_dic[model_name].append(round(report_accuracy, self.rounding_precision))

        # print('Generating Confusion Matrix')
        confusion = confusion_matrix(self.y_test.to_numpy(), y_pred, labels=self.actions)
        (pd.DataFrame(confusion)).to_csv(self.report_path + "/" + model_name + '_Run_' + str(global_run_counter)
                                         + '_confusion_matrix.csv')

        print("Saving model results as: ", self.report_path + "/" + model_name + '_Run_' + str(global_run_counter)
              + '_report.csv')
        report_df = pd.DataFrame(report)
        (round(report_df, self.rounding_precision).T).to_csv(
            self.report_path + "/" + model_name + '_Run_' + str(global_run_counter)
            + '_report.csv')

        # self.y_score.append(model_y_score)

        # print("Plotting model results for ", model_name + " Run " + str(global_run_counter))
        # self.plot_all_OvR_ROC(model_name + "_Run_" + str(global_run_counter), model_y_score)

        self.line_printer.print_line()

        pd.DataFrame(self.accuracy_dic).to_csv(self.report_path + '/Accuracy_Run_' + str(global_run_counter) + '.csv')
        global_run_counter += 1
        return self.y_score

    def evaluate_model(self):

        global global_run_counter
        self.line_printer.print_text('Starting Run ' + str(global_run_counter))

        for model_counter in tqdm(range(len(self.model_pipeline))):
            # run the models for the first time
            model_name = self.model_names[model_counter]
            self.line_printer.print_line()
            print("Starting Model: ", model_name, ' with test_size: ', len(self.X_test), ' and train size: ',
                  len(self.X_train))

            model_file_name = self.model_path + "/" + model_name + '.joblib'
            if global_run_counter == 0:
                model = self.model_pipeline[model_counter]
            # Load the models
            else:
                print('Loading model: ', model_name)
                model = load(model_file_name)

            fitted_model = model.fit(self.X_train, self.y_train)
            print("Saving model ", model_name)
            dump(fitted_model, model_file_name)

            y_pred = model.predict(self.X_test)
            #model_y_score = fitted_model.predict_proba(self.X_test)

            # print('model_y_score: ', model_y_score)
            report = classification_report(self.y_test.to_numpy(), y_pred, output_dict=True,
                                           digits=self.rounding_precision)
            # target_names = self.target_names,
            report_accuracy = report['accuracy']
            # print('report[accuracy]: ', round(report_accuracy, 2))
            self.accuracy_dic[model_name].append(round(report_accuracy, self.rounding_precision))

            # print('Generating Confusion Matrix')
            confusion = confusion_matrix(self.y_test.to_numpy(), y_pred, labels=self.actions)
            (pd.DataFrame(confusion)).to_csv(self.report_path + "/" + model_name + '_Run_' + str(global_run_counter)
                                              + '_confusion_matrix.csv')

            print("Saving model results as: ", self.report_path + "/" + model_name + '_Run_' + str(global_run_counter)
                  + '_report.csv')
            report_df = pd.DataFrame(report)
            (round(report_df, self.rounding_precision).T).to_csv(
                self.report_path + "/" + model_name + '_Run_' + str(global_run_counter)
                + '_report.csv')

            # self.y_score.append(model_y_score)

            # print("Plotting model results for ", model_name + " Run " + str(global_run_counter))
            # self.plot_all_OvR_ROC(model_name + "_Run_" + str(global_run_counter), model_y_score)

            self.line_printer.print_line()

        pd.DataFrame(self.accuracy_dic).to_csv(self.report_path + '/Accuracy_Run_' + str(global_run_counter) + '.csv')
        global_run_counter += 1
        return self.y_score

    def plot_confusion_matrix(self):
        fig = plt.figure(figsize=(50, 25))
        for i in range(len(self.confusion_matrix_list)):
            confusion = self.confusion_matrix_list[i]
            model_name = self.model_list[i]
            sub = fig.add_subplot(2, 3, i + 1).set_title(model_name)
            confusion_plot = sns.heatmap(confusion, annot=True, cmap='Blues_r')
            confusion_plot.set_xlabel('Predicted Values')
            confusion_plot.set_ylabel('Actual Values')

    # def accuracy_results(self):
    #     return pd.DataFrame({'Model': self.model_list, ' Accuracy': self.acc_list, 'AUC': self.auc_list})

    def calculate_ROC_value_using_micro_averaged_OvR(self):
        # store the fpr, tpr, and roc_auc for all averaging strategies
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    def generate_fpr_tpr_for_ROC_curve_using_the_OvR_macro_average(self, y_score):
        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(self.n_classes):
            # print("self.y_onehot_test[:,", i,"]: ", self.y_onehot_test[:, i])
            # print("y_score[:, i]: ", y_score[:, i])
            fpr[i], tpr[i], _ = roc_curve(self.y_onehot_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(self.n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        return fpr["macro"], tpr["macro"], roc_auc["macro"]

    def calculate_ROC_curve_using_OvR_average(self, average_type='micro'):
        if ((average_type == 'micro') | (average_type == 'macro')):
            the_roc_auc_ovr = roc_auc_score(
                self.y_test,
                y_score,
                multi_class="ovr",
                average=average_type,
            )
            print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{the_roc_auc_ovr:.2f}")

            return the_roc_auc_ovr
        else:
            raise ValueError("Average type can only be micro or macro")

    def plot_ROC_curve_for__specific_action(self, action):
        class_id = np.flatnonzero(label_binarizer.classes_ == action)[0]

        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
        plt.legend()
        plt.show()

    def plot_ROC_curve_using_micro_averaged_OvR(self):
        RocCurveDisplay.from_predictions(
            y_onehot_test.ravel(),
            y_score.ravel(),
            name="micro-average OvR",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
        plt.legend()
        plt.show()

    def plot_all_OvR_ROC(self, plot_name: str, y_score, show_plot=False, save_plot=True, plot_micro=True,
                         plot_macro=True, plot_all=False):
        from itertools import cycle

        fig, ax = plt.subplots(figsize=(20, 15))

        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        # self.line_printer.print_text('len(y_score): '+str(len(y_score.ravel())))
        # self.line_printer.print_text('len(y_onehot_test.ravel()): ' + str(len(self.y_onehot_test.ravel())))
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_onehot_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        if plot_micro | plot_all:
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )
        macro_fpr, macro_tpr, roc_ = self.generate_fpr_tpr_for_ROC_curve_using_the_OvR_macro_average(y_score)
        if plot_macro | plot_all:
            plt.plot(
                macro_fpr,
                macro_tpr,
                label=f"macro-average ROC curve (AUC = {roc_:.2f})",
                color="navy",
                linestyle=":",
                linewidth=4,
            )

        if plot_all:
            colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "black", "brown", "grey"])
            for class_id, color in zip(range(self.n_classes), colors):
                RocCurveDisplay.from_predictions(
                    self.y_onehot_test[:, class_id],
                    y_score[:, class_id],
                    name=f"ROC curve for {self.target_names[class_id]}",
                    color=color,
                    ax=ax,
                )

        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend()

        if save_plot:
            plt.savefig(self.plot_path + '/' + plot_name + '.png')
        if show_plot:
            plt.show()

        plt.close()
