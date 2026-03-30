import pandas as pd
import numpy as np


class Base_Model():

    def __init__(self, data_json_dict):
        self._training_samples = self._read_samples(
            data_json_dict["training_samples"])
        self._test_samples = self._read_samples(data_json_dict["test_samples"])

        basic_check_samples = self._perform_basic_check_samples()

        self._features = self._read_samples(data_json_dict['features'])

        self.analysis_name = data_json_dict['analysis_name']
        self._randomness = data_json_dict["rand"]
        self._output_dir = data_json_dict["output_dir"]

    def _read_samples(self, path_to_sample_list: str) -> list:
        '''
        @param path_to_training_samples: path to file with training samples (one sample per row, can be either cell lines or cell line - drug pairs), can also read drug file (one drug per row)
        @return list of samples
        '''
        if path_to_sample_list == "":
            return []
        with open(path_to_sample_list, "r", encoding="utf-8") as sample_file:
            samples = sample_file.read().splitlines()
        return samples

    def _perform_basic_check_samples(self):

        warning_msg = ""
        if len(self._training_samples) == 0:
            warning_msg = warning_msg + "No training samples in input file" + "\n"

        for sample in self._test_samples:
            if sample in self._training_samples:
                warning_msg = warning_msg + "Test sample " + sample + \
                    " is found in training and test set." + "\n"

        return warning_msg

    def parse_matrix(self, matrix_file):
        '''
        @param matrix_file: a file with data for a specific molecular data type (e.g. morgan fingerprints), samples in rows, features in column
        @return: a pandas DataFrame with the information
        '''
        matrix = pd.read_csv(matrix_file, sep='\t')
        matrix.set_index(matrix.columns.values[0], drop=True, inplace=True)
        if matrix.isna().any().any():
            old_rows = set(matrix.index.to_list())
            matrix.dropna(inplace=True, ignore_index=False)
            new_rows = set(matrix.index.to_list())
            to_remove = sorted(old_rows - new_rows)
            for sample in to_remove:
                if sample in self._test_samples:
                    self._test_samples.remove(sample)
                elif sample in self._training_samples:
                    self._training_samples.remove(sample)
        return matrix

    def combine_feature_types(self, feature_list: list):
        '''
        @param feaure_list: a list of feature DataFrames that should be combined such that the can be passed together to a model
        @return:  a pandas DataFrame with the combined information
        '''
        combined_features = pd.concat(feature_list, axis=1)
        return combined_features

    def remove_unnecessary_features(self, matrix: pd.DataFrame):
        new_matrix = matrix.loc[:, self._features]
        return new_matrix

    def _parse_response_file(self, response_file):
        response = pd.read_csv(response_file, sep='\t')
        response.set_index(response.columns.values[0], drop=True, inplace=True)
        return response
