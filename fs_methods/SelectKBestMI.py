from sklearn.feature_selection import SelectKBest, mutual_info_classif
from Base_FS import Base_FS


class Select_MI(Base_FS):
    def __init__(self, data_json_dict):
        super().__init__(data_json_dict)
        fs_specific_parameters = data_json_dict['fs_specific']
        self.response_file = fs_specific_parameters['response']

        self.response = super()._parse_response_file(self.response_file)
        self.selector = SelectKBest(
            score_func=mutual_info_classif, k=self.num_features)

    def select_features(self):
        self.selector.set_output(transform='pandas')
        self.selector.fit(self.feature_matrix, self.response)
        feature_names = self.selector.get_feature_names_out()
        with open(f'{self.output_dir}//{self.feature_type}_feature_names_MI.txt', 'w') as output:
            for feature in feature_names:
                output.write(f'{feature}\n')
