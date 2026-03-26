from Base_FS import Base_FS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Variance_FS(Base_FS):

    def __init__(self, data_json_dict):
        super().__init__(data_json_dict)
        self.full_feature_matrix = pd.read_csv(
            data_json_dict['features'], sep='\t')
        self.full_feature_matrix.set_index('compound', drop=True, inplace=True)

    def select_features(self):
        variances = {}

        self.scaler = MinMaxScaler()
        self.matrix_scaled = pd.DataFrame(data = self.scaler.fit_transform(self.feature_matrix), columns=self.feature_matrix.columns, index=self.feature_matrix.index)


        for feature in self.matrix_scaled.columns:
            feature_vec_var = self.matrix_scaled[feature].values.var()
            variances[feature] = feature_vec_var

        sorted_features = sorted(variances, key=variances.get, reverse=True)

        with open(f'{self.output_dir}//{self.feature_type}_feature_names_variance.txt', 'w') as output:
            for n in range(self.num_features):
                output.write(f'{sorted_features[n]}\n')
