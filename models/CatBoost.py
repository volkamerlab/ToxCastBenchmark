from catboost import CatBoostClassifier, CatBoostRegressor
from Base_Model import Base_Model


class Cat_Boost(Base_Model):

    def __init__(self, data_json_dict):
        super().__init__(data_json_dict)
        model_specifc_parameters = data_json_dict['model_specific']
        morgan_fps_file = model_specifc_parameters['morgan']
        maccs_fps_file = model_specifc_parameters['maccs']
        morphological_fps_file = model_specifc_parameters['morphological']
        phys_chem_file = model_specifc_parameters['physchem']
        # check if we can do this with RF
        smiles_file = model_specifc_parameters['smiles']
        response_file = model_specifc_parameters['response']
        self._used_data_types = []

        if morgan_fps_file != '':
            self._morgan_fps = super().parse_matrix(morgan_fps_file)
            self._morgan_fps_used = True
            self._used_data_types.append(self._morgan_fps)
        else:
            self._morgan_fps_used = False

        if maccs_fps_file != '':
            self._maccs_fps = super().parse_matrix(maccs_fps_file)
            self._maccs_fps_used = True
            self._used_data_types.append(self._maccs_fps)
        else:
            self._maccs_fps_used = False

        if morphological_fps_file != '':
            self._morphological_fps = super().parse_matrix(morphological_fps_file)
            self._morphological_fps_used = True
            self._used_data_types.append(self._morphological_fps)
        else:
            self._morphological_fps_used = False

        if phys_chem_file != '':
            self.phys_chem = super().parse_matrix(phys_chem_file)
            self.phys_chem_used = True
            self._used_data_types.append(self.phys_chem)
        else:
            self.phys_chem_used = False

        if smiles_file != '':
            self.smiles = super().parse_matrix(smiles_file)
            self.smiles_used = True
            self._used_data_types.append(self.smiles)
        else:
            self.smiles_used = False

        if response_file == '':
            print("Path to response must be given")
            raise ValueError()

        self.response = super()._parse_response_file(response_file)

        if len(self._used_data_types) == 0:
            print("At least one feature type must be given")
            raise ValueError()
        elif len(self._used_data_types) > 1:
            self._feature_matrix = super().combine_feature_types(self._used_data_types)
        else:
            self._feature_matrix = self._used_data_types[0]

        self._feature_matrix = super().remove_unnecessary_features(self._feature_matrix)

        self._default_hyperparameters = {'random_state': 42}

        hyperparameter_dict = model_specifc_parameters['hyperparameters']
        if hyperparameter_dict != '':
            self.hyperparameters = hyperparameter_dict
        else:
            self.hyperparameters = self._default_hyperparameters

        if data_json_dict['task'] == 'classification':
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()
        if 'use_best_model' in self.hyperparameters.keys():
            self.hyperparameters['use_best_model'] = bool(
                self.hyperparameters['use_best_model'])
        self.model = self.model.set_params(**self.hyperparameters)

    def fit(self):

        X_train = self._feature_matrix.loc[self._training_samples, :]

        y_train = self.response.loc[self._training_samples, 'response']

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_train)
        outpath = f'{self._output_dir}/{self.analysis_name}_train_predictions.csv'
        with open(outpath, 'w') as output:
            output.write(f'compound\tpredicted\tactual\n')
            for i, sample in enumerate(X_train.index.values):
                output.write(f'{sample}\t{y_pred[i]}\t{y_train.values[i]}\n')

    def predict(self):
        X_test = self._feature_matrix.loc[self._test_samples, :]
        y_test = self.response.loc[self._test_samples, 'response']
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        outpath = f'{self._output_dir}/{self.analysis_name}_test_predictions.csv'
        with open(outpath, 'w') as output:
            output.write(f'compound\tpredicted\tactual\tp(0)\tp(1)\n')
            for i, sample in enumerate(X_test.index.values):
                output.write(f'{sample}\t{y_pred[i]}\t{y_test.values[i]}\t{y_pred_proba[i][0]}\t{y_pred_proba[i][1]}\n')

