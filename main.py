# import numpy as np
import pandas as pd
# import gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.ppo import MlpPolicy

# from imitation.algorithms import bc
# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper
from supervised_classifiers import SupervisedClassifier
from file_utility import FileUtility
from load_file import YahooDataLoader
from tqdm import tqdm
from line_printer import LinePrinter
import timeit


def run_classifier(classifiers, train_data, test_data, extra_classification_counter):
    if classifiers.generate_test_train(train_data, test_data):
        y_scores = classifiers.evaluate_model()
        extra_classification_counter = 0

    else:
        extra_classification_counter += 1

    return extra_classification_counter

if __name__ == '__main__':
    starting_time = timeit.default_timer()
    print("Start time :", starting_time)

    data_path = '../Data_Source/Yahoo/Processed_Yahoo_Data/Stock'
    # folder_path = 'AMS'
    # This parameter is used in order to prevent memory problems. If a folder contains more than 1000 files,
    # then the system will load 1000 files first, do the training and then continue with the rest of the files

    line_printer = LinePrinter()
    # no_of_files_to_start_training = 1000
    min_df_size_to_start_training = 5000

    batch_size_to_load_if_min_df_size_does_not_have_enough_sample_data = 1000

    file_info = {'source_data_path': data_path,
                 'save_destination_path': 'results',
                 'file_formats_to_load': 'csv',
                 'file_format_to_save': 'csv',
                 'verbose': True
                 }
    data_col = ['-' + str(i) for i in range(62, 1, -1)]
    data_col.append('0')
    # % of the data to be used for testing
    # for example 75% of the files in each folder for training and the rest of the folders will be used for test
    train_percent = 0.75

    file_util = FileUtility(**file_info)
    classifiers = SupervisedClassifier()
    classifiers.generate_pipeline()
    data_loader = YahooDataLoader()

    # Load all folders and iterate through each folder and perform trainining:
    list_of_folders = file_util.load_all_sub_directories()

    # This counter will count the number of batch_size_to_load_if_min_df_size_does_not_have_enough_sample_data
    # it will be multiplied with the above variable and then added to min_df_size_to_start_training in order to
    # make sure there is enough data before calling the classifier
    extra_classification_counter = 0

    test_data = pd.DataFrame()
    train_data = pd.DataFrame()

    total_row_counter = 0

    for folder_counter in tqdm(range(len(list_of_folders))):
        folder_name = list_of_folders[folder_counter]
        list_of_files = file_util.load_file_names_in_directory(folder_name)
        no_of_files_in_folder = len(list_of_files)
        for file_counter in tqdm(range(no_of_files_in_folder)):
        # for file_counter in tqdm(range(3)):
            loaded_data = data_loader.load_file(data_path + "/" + folder_name, list_of_files[file_counter], interval=60)
            total_row_counter += len(loaded_data)

            # if there is information available in the file we have just loaded then proceed
            if len(loaded_data) > 0:
                # threshold is the minimum number of rows of data required before entering starting the classifier
                threshold = (min_df_size_to_start_training + extra_classification_counter *
                             batch_size_to_load_if_min_df_size_does_not_have_enough_sample_data)

                # Generate Test Data
                # first check if we have enough train data, then collect test data
                if len(train_data) < threshold * train_percent:
                    train_data = pd.concat([loaded_data, train_data])
                    # print('Train DATA::::::')
                    # print(train_data)
                    # line_printer.print_line()
                else:
                    # Generate Train Data
                    test_data = pd.concat([loaded_data, test_data])


                # if we have enough data, then we can run the classifier.

                if ( (len(test_data) + len(train_data))> threshold ):
                    extra_classification_counter = run_classifier(classifiers, train_data, test_data,
                                                                  extra_classification_counter)
                    # this is to free up memory. If the run_classifier manged to conduct training then we can start
                    # collecting data for a new train and test dataset
                    if (extra_classification_counter == 0):
                        train_data = pd.DataFrame()
                        test_data = pd.DataFrame()

                print("Total Row Counter: ", total_row_counter)
                print('len (train_data): ', len(train_data))
                print('len (test_data): ', len(test_data))


    print('Before Final Run len (train_data): ', len(train_data))
    print('Before Final Run len (test_data): ', len(test_data))
    final_length = len(train_data) + len(test_data)
    print('Final total: ', final_length)

    # This part processes all the data that is left after the last file is loaded but the length of test and train
    # dataset is still below the threshold
    combined_df = pd.concat([train_data,test_data])
    combined_df.reset_index(inplace=True, drop=True)

    if (final_length > 0):
        no_of_test_data_required = int(final_length * (1-train_percent))
        no_of_train_data_required = int(final_length * train_percent)


        if ( len(test_data) < (no_of_test_data_required) ):
            test_data = pd.concat([test_data, combined_df.iloc[no_of_train_data_required:]])
            train_data = combined_df.iloc[:no_of_train_data_required]

        final_classification_counter = run_classifier(classifiers, train_data, test_data, extra_classification_counter)

    print('After Final Run len (train_data): ', len(train_data))
    print('After Final Run (test_data): ', len(test_data))


    print("Total Row Counter: ", total_row_counter)

