import pandas as pd


class Base_FS():

    def __init__(self, data_json_dict):
        self.num_features = int(data_json_dict['num_features'])
        self.feature_file = data_json_dict['features']
        self.samples = self.read_sample_file(data_json_dict['samples'])
        self.feature_matrix = self.read_feature_matrix(self.feature_file)
        self.output_dir = data_json_dict['output_dir']
        self.feature_type = data_json_dict['feature_type']

    def read_feature_matrix(self, matrix_file):
        features = pd.read_csv(matrix_file, sep='\t')
        features.set_index('compound', drop=True, inplace=True)
        features.dropna(inplace=True, axis=1)
        features = features.loc[self.samples, :]
        return features

    def _parse_response_file(self, response_file):
        response = pd.read_csv(response_file, sep='\t')
        response.set_index(response.columns.values[0], drop=True, inplace=True)
        response = response.loc[self.samples, :]
        return response

    def read_sample_file(self, sample_file):
        with open(sample_file, 'r') as input:
            samples = input.read().splitlines()
        return samples
