from sklearn.decomposition import PCA
from Base_FS import Base_FS
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Principal_Component_Analysis(Base_FS):

    def __init__(self, data_json_dict):
        super().__init__(data_json_dict)
        fs_specific_parameters = data_json_dict['fs_specific']
        self.random_state = fs_specific_parameters['random_state']
        self.file_name = fs_specific_parameters['file_name']
        self.full_feature_matrix = pd.read_csv(
            data_json_dict['features'], sep='\t')
        self.full_feature_matrix.set_index('compound', drop=True, inplace=True)
        self.full_feature_matrix = self.full_feature_matrix.loc[:,
                                                                self.feature_matrix.columns]
        self.fs_method = PCA(n_components=self.num_features,
                             random_state=self.random_state)

    def select_features(self):
        self.scaler = StandardScaler()
        self.matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        self.fs_method.fit(self.matrix_scaled)
        self.full_matrix_scaled = self.scaler.transform(self.full_feature_matrix)
        self.transformed_matrix = self.fs_method.transform(
            self.full_matrix_scaled)
        data = pd.DataFrame(data=self.transformed_matrix, columns=[f'PC{n}' for n in range(
            1, self.num_features + 1)], index=self.full_feature_matrix.index)
        data.to_csv(f'{self.output_dir}/{self.file_name}', sep='\t')
        with open(f'{self.output_dir}/{self.feature_type}_feature_names_pca.txt', 'w') as output:
            for n in range(1, self.num_features + 1):
                output.write(f'PC{n}\n')
