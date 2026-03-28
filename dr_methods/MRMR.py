from mrmr import mrmr_classif, mrmr_regression
from Base_FS import Base_FS


class MRMR(Base_FS):

    def __init__(self, data_json_dict):
        super().__init__(data_json_dict)
        fs_specific_parameters = data_json_dict['fs_specific']
        self.response_file = fs_specific_parameters['response']

        self.response = super()._parse_response_file(self.response_file)
        self.task = fs_specific_parameters['task']
        if not self.task in ['classification', 'regression']:
            print('task not known')
            raise ValueError()
        if self.task == 'classification':
            self.selector = mrmr_classif
        else:
            self.selector = mrmr_regression

    def select_features(self):
        best_n_features = self.selector(
            X=self.feature_matrix, y=self.response['response'], K=self.num_features)

        with open(f'{self.output_dir}/{self.feature_type}_feature_names_mrmr.txt', 'w') as output:
            for n in range(0, self.num_features):
                output.write(f'{best_n_features[n]}\n')
