import pandas as pd


class YahooDataLoader:
    def __init__(self, column_name_dictionary=None, date_index_new_name=None, verbose=False):
        """

        :param column_name_dictionary: a dictionary containing name of the column in the dataset.
        The dictionary format should be:
        {current_open_column_name: desired_open_column_name , current_high_column_name: desired_high_column_name,
        current_low_column_name: desired_low_column_name, current_close_column_name: desired_close_column_name,
        current_volume_column_name: desired_volume_column_name, current_date_column_name: desired_date_column_name}
        This class will change the keys and values of this dictionary so the keys become new column names

        """
        if not column_name_dictionary:
            column_name_dictionary = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                                      'Volume': 'volume'}
        self.column_names = column_name_dictionary
        self.date_index_new_name = date_index_new_name
        self.verbose = verbose

    def rename_columns(self, original_data):
        original_data.index.rename(self.date_index_new_name, inplace=True)
        original_data.rename(columns=self.column_names, inplace=True)

    def load_file(self, path, file_name, interval = 1):
        """

        :param path: path must include '/'
        :param file_name:
        :param column_names: [open, high, low, close, volume]
        :param data_has_volume: clarify if data has volume information
        :param interval: It will return data in intervals set by this parameters. so for example if this is set to 2
        then the data will skip 2 rows for each row it returns
        :return: Pandas DataFrame with ascending sorted pandas datetime index
        """
        if interval < 1:
            raise ValueError("interval must be greater than 0")
        if (path[-1] != '/'):
            # raise Warning('Path must end with /')
            path = path+'/'

        file_format = file_name.split('.')[-1]
        try:
            if self.verbose:
                print('loading ', file_name)

            if file_format == 'csv':
                data = pd.read_csv(path + file_name, index_col=0, parse_dates=True)
            elif file_format == 'xlsx':
                data = pd.read_excel(path + file_name, index_col=0, parse_dates=True)
            else:
                raise Exception('File format must be either .csv or .xlsx')

            data.index = pd.to_datetime(data.index, utc=True)
            # to make sure index is sorted properly
            data.sort_index(inplace=True)

            data.name = file_name.replace(file_format, '')[:-1]
            data['action'] = data['action'].astype(int)

            data = data.iloc[::interval, :]
            return data
        except:
            print('Could not load ', path, file_name)
