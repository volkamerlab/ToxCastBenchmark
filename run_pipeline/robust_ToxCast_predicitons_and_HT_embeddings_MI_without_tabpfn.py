import os
import subprocess
import glob
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def main():

    rf_hyperparams = []
    for trees in range(100, 600, 200):
        for depth in range(10, 50, 10):
            for samples_leaf in range(5, 25, 5):
                rf_hyperparams.append(
                    '{ "min_samples_leaf": ' + str(samples_leaf) + ', "n_estimators": ' + str(trees) + ', "max_depth": ' + str(depth) + ', "random_state": 42, "class_weight": "balanced", "n_jobs": -1}')

    svm_hyperparams = []

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for gamma in ['scale', 'auto']:
            for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                svm_hyperparams.append(
                    '{ "kernel": "' + kernel + '", "max_iter": 1000, "gamma": "' + str(gamma) + '" , "probability": "False", "class_weight": "balanced"}')

    mlp_hyperparams = []

    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
        for activation_function in ['relu', 'tanh', 'logistic', 'identity']:
            for hidden_layer_sizes in [(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 50, 50), (100, 100, 100), (100, 100, 50)]:
                mlp_hyperparams.append(
                    '{"early_stopping": "True", "alpha": ' + str(alpha) + ', "hidden_layer_sizes" : "' + str(hidden_layer_sizes)+'", "activation" : "' + activation_function + '", "random_state": 42, "learning_rate": "adaptive", "solver": "adam", "batch_size": "auto"}')

    cat_boost_hyperparams = []
    for depth in range(4, 11):  # see https://catboost.ai/docs/en/concepts/parameter-tuning and https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning.ipynb
        for l2_leaf_reg in [0.001, 0.01, 0.1, 1]:
            for bootstrap_type in ['Bayesian', 'Bernoulli']:
                cat_boost_hyperparams.append(
                    '{"iterations": 1000, "bootstrap_type": "' + str(bootstrap_type) + '", "l2_leaf_reg" : '+str(l2_leaf_reg) + ', "depth" : '+str(depth) + ', "logging_level":"Silent"}')

    hyperparameters = {
        'mlp': mlp_hyperparams,
        'rf':  rf_hyperparams,
        'svm':  svm_hyperparams,
        'cat_boost': cat_boost_hyperparams,
        'tabpfn': ['{"device":"cuda", "max_time":600}']
    }

    task = 'classification'
    model_main = '../models/main_models.py'
    json_config_gen = "../models/json_config_generator.sh"

    feature_type = 'embeddings'
    for dr_name in ['MI']:

        subfolders = ['steroidal']
        with open('/local/lisa-marie.rolli/comptox_benchmark/already_done_embeddings.txt', 'r') as already_done_file:
            already_done_assays = already_done_file.read().splitlines()
        for subfolder in subfolders:

            input_directory = f'../model_inputs/{subfolder}/'
            for content in os.listdir(input_directory):
                if '.csv' in content:
                    continue
                if content in already_done_assays:
                    continue
                print(content)
                path_to_CV_folds = f'{input_directory}/{content}/'
                output_final_results = f'../model_outputs/{subfolder}/{content}/'
                pattern = f'{content}-*_binary_response.csv'

                matching_files = glob.glob(
                    f'../ToxCastDownloads/binary_responses_and_datasail_input_files/{subfolder}/{pattern}')

                if matching_files:
                    response = matching_files[0]
                else:
                    continue
                final_models = [False for _ in range(5)]
                for model_name in ['cat_boost', 'rf', 'mlp', 'svm']:

                    for fold in range(5):
                        if not final_models[fold]:
                            with open(f'{output_final_results}/fold{fold}/final_models_{feature_type}_{dr_name}.txt', 'w') as output:
                                output.write('')
                            final_models[fold] = True
                        # for hyperparameter combination
                        if not (model_name == 'tabpfn'):
                            best_combi = None
                            best_score = -1
                            for combi in hyperparameters[model_name]:
                                combi_dict = eval(combi)
                                combi_val = 0
                                for ht_fold in range(4):
                                    training_samples = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/train.txt'
                                    test_samples = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/test.txt'
                                    if not (dr_name == 'none'):
                                        features = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/{feature_type}_feature_names_{dr_name}.txt'
                                    else:
                                        features = f'../model_inputs/all_feature_names_{feature_type}.txt'

                                    if not dr_name == 'pca':
                                        feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/embeddings.csv'

                                    else:
                                        feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/{feature_type}_transformed_matrix_pca.csv'

                                    model_specific = '{ "physchem": "", "maccs": "", "morgan": "' + \
                                        feature_matrix_path + \
                                        '", "smiles": "", "morphological": "", "response": "' + response + \
                                        '", "hyperparameters": ' + \
                                        combi + '}'
                                    output_dir = f'../temp/{subfolder}/{content}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/'
                                    os.makedirs(output_dir, exist_ok=True)
                                    config_file_path = f'{output_dir}/{dr_name}_{model_name}_config.json'
                                    analysis_name = f'{model_name}_{feature_type}_{dr_name}'
                                    for hp in combi_dict.keys():
                                        analysis_name += f'_{hp}_{combi_dict[hp]}'
                                    subprocess.run(args=['bash', json_config_gen,
                                                         model_name, training_samples, test_samples, features, output_dir, 'deterministic', model_specific, task, analysis_name, config_file_path])
                                    subprocess.run(
                                        ['python3', model_main, config_file_path])
                                    res = pd.read_csv(
                                        f'{output_dir}/{analysis_name}_test_predictions.csv', sep='\t')
                                    mcc = matthews_corrcoef(
                                        y_pred=res['predicted'].values, y_true=res['actual'].values)
                                    combi_val += mcc
                                combi_val /= 4
                                if combi_val > best_score:
                                    best_score = combi_val
                                    best_combi = combi
                                with open(f'../temp/{subfolder}/{content}/fold{fold}//hyperparameter_tuning/{model_name}_combis_{feature_type}_{dr_name}.csv', 'a') as ht_val_file:
                                    ht_val_file.write(
                                        f'\n{analysis_name}\t{combi_val}')
                        else:
                            best_combi = hyperparameters['tabpfn'][0]
                        if model_name == 'svm':
                            str(best_combi).replace(
                                '"probability": "False"', '"probability": "True"')
                        if not dr_name == 'pca':
                            feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/embeddings.csv'

                        else:
                            feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/{feature_type}_transformed_matrix_pca.csv'
                        if not (dr_name == 'none'):
                            features = f'{path_to_CV_folds}/fold{fold}/{feature_type}_feature_names_{dr_name}.txt'
                        else:
                            features = f'../model_inputs/all_feature_names_{feature_type}.txt'

                        training_samples = f'{path_to_CV_folds}/fold{fold}/train.txt'
                        test_samples = f'{path_to_CV_folds}/fold{fold}/test.txt'
                        output_dir = f'{output_final_results}/fold{fold}/'
                        model_specific = '{ "physchem": "", "maccs": "", "morgan": "' + \
                            feature_matrix_path + \
                            '", "smiles": "", "morphological": "", "response": "' + response + \
                            '", "hyperparameters": ' + \
                            best_combi + '}'
                        analysis_name = f'{model_name}_best_combi_{feature_type}_{dr_name}'
                        config_file_path = f'{output_dir}/{dr_name}_config.json'
                        subprocess.run(args=['bash', json_config_gen,
                                             model_name, training_samples, test_samples, features, output_dir, 'deterministic', model_specific, task, analysis_name, config_file_path])

                        subprocess.run(
                            ['python3', model_main, config_file_path])
                        res = pd.read_csv(
                            f'{output_dir}/{analysis_name}_test_predictions.csv', sep='\t')
                        mcc = matthews_corrcoef(
                            y_pred=res['predicted'].values, y_true=res['actual'].values)
                        auroc = roc_auc_score(
                            y_true=res['actual'], y_score=res.loc[:, 'p(1)'].values)
                        with open(f'{output_final_results}/fold{fold}/final_models_{feature_type}_{dr_name}.txt', 'a') as output:
                            output.write(
                                f'\n{model_name}\tfold{fold}\t{mcc}\t{auroc}')


if __name__ == '__main__':
    main()
