from line_printer import LinePrinter
from file_utility import FileUtility
from load_file import YahooDataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm


class BatchDataLoader:
    def __init__(self, batch_size: int, sentence_length: int, include_volatility: bool, data_path: str,
                 file_utility_input: dict, intervals=1):

        """
        This class returns a batch of data and keeps track of the remaining folders and files
        :param batch_size: The actual batch size (number of rows of data to load)
        :param data_path: The main folder to load the data. All data must be in the sub folders from this data_path
        :param file_utility_input: dictionary containing default parameters of the file_utility class
        track previously loaded data. Data might be repetitive. This option is good Optuna optimization
        :param intervals: the interval for data to be loaded. by default a 1 means to load all the data
        """

        self.batch_size = batch_size
        self.data_path = data_path
        self.file_utility = FileUtility(**file_utility_input)
        self.data_loader = YahooDataLoader()
        self.list_of_folders = self.file_utility.load_all_sub_directories()

        self.intervals = intervals
        self.line_printer = LinePrinter()
        # the folder and file counter are used to keep track of the files and folders that are loaded
        # so in the next request, the system can start at the right place
        self.folder_counter = 0
        self.file_counter = 0
        self.sentence_length = sentence_length
        self.include_volatility = include_volatility
        data_col = ['-' + str(i) for i in range(self.sentence_length - 1, 1, -1)]
        data_col.append('0')
        self.data_col = data_col
        # Since this class loads complete files, the total number of rows of data fetched might be greater than
        # batch size. data_to_return.iloc[:batch_size]. So the portion of data_to_return.iloc[batch_size:] will be
        # kept in this variable for the next call to fetch data
        self.data_left_from_previous_call = pd.DataFrame()
        # self.randomize_output = randomize_output

    def fetch_batch(self, load_positive_actions):

        """

        :return: a dataframe with shape (minimum_batch_size, n_features. The function also returns a boolean done flag
        to determin the end of all data available in the folder. If true, then there are eno more edata left
        """

        row_counter = 0
        # load the data left over from the last function call
        data_to_return = self.data_left_from_previous_call
        print("List of folder: ", self.list_of_folders)
        # loop through the folder list starting from the last call to the functon
        for folder_counter in tqdm(range(self.folder_counter, len(self.list_of_folders))):
            folder_name = self.list_of_folders[folder_counter]

            list_of_files = self.file_utility.load_file_names_in_directory(folder_name)
            no_of_files_in_folder = len(list_of_files)

            for file_counter in tqdm(range(self.file_counter, no_of_files_in_folder)):
                loaded_data = self.data_loader.load_file(self.data_path + "/" + folder_name,
                                                         list_of_files[file_counter],
                                                         load_positive_actions=load_positive_actions,
                                                         interval=self.intervals)
                row_counter += len(loaded_data)
                data_to_return = pd.concat([data_to_return, loaded_data])

                if row_counter >= self.batch_size:
                    # if the file counter has not reached the end of the files in a folder, then increase self.file_counter
                    # so it can load the correct file in thee next call, else reset the self.file_counter to zero, if not
                    if file_counter < no_of_files_in_folder:
                        self.file_counter = file_counter + 1
                    else:
                        self.file_counter = 0
                    self.folder_counter = folder_counter
                    self.data_left_from_previous_call = data_to_return.iloc[self.batch_size:]
                    return data_to_return.iloc[:self.batch_size], False

        self.file_counter = file_counter
        self.folder_counter = folder_counter

        # since there are no more data left to load, set the dataframe to none in order to manage memory usage
        self.data_left_from_previous_call = None

        return data_to_return, True

    def fetch_batch_randomized(self, load_positive_actions):

        row_counter = 0
        # load the data left over from the last function call
        #folder_id_range = np.arange(0, len(self.list_of_folders))
        # loop through the folder list starting from the last call to the functon
        print("Total Row loaded so far: ", row_counter)
        data_to_return = pd.DataFrame()
        for idx in tqdm(range(0, len(self.list_of_folders))):

            folder_counter = np.random.choice(len(self.list_of_folders), 1)
            folder_name = self.list_of_folders[folder_counter[0]]

            list_of_files = self.file_utility.load_file_names_in_directory(folder_name)
            no_of_files_in_folder = len(list_of_files)

            for file_counter in tqdm(range(0, no_of_files_in_folder)):
                loaded_data = self.data_loader.load_file(self.data_path + "/" + folder_name,
                                                         list_of_files[file_counter],
                                                         load_positive_actions=load_positive_actions,
                                                         interval=self.intervals)
                row_counter += len(loaded_data)
                data_to_return = pd.concat([data_to_return, loaded_data])

                # if enough data was collected, then return data
                print('row_counter: ', row_counter)
                if row_counter >= self.batch_size:
                    return data_to_return.iloc[:self.batch_size]

        return data_to_return
