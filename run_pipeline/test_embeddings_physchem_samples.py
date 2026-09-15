import os
import subprocess
import glob
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def main():
    maccs_physchem_samples = pd.read_csv('../model_inputs/maccs_physchem_samples.csv', sep = '\t', index_col = 0)
    physchem_samples = maccs_physchem_samples.index.tolist()
    rf_hyperparams = []
    for trees in range(100, 600, 200):
        for depth in range(10, 50, 10):
            for samples_leaf in range(5, 25, 5):
                rf_hyperparams.append(
                    '{ "min_samples_leaf": ' + str(samples_leaf) + ', "n_estimators": ' + str(trees) + ', "max_depth": ' + str(depth) + ', "random_state": 42, "class_weight": "balanced", "n_jobs": -1}')

    hyperparameters = {
        'rf':  rf_hyperparams
    }

    task = 'classification'
    model_main = '../models/main_models.py'
    json_config_gen = "../models/json_config_generator.sh"

    feature_type = 'embeddings'
    dr_name = 'none'
    with open('../model_inputs/sorted_assays.txt', 'r') as assay_file:
        sorted_assays = assay_file.read().splitlines()
    for assay in sorted_assays:

        subfolders = ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']

        
        for subfolder in subfolders:

            input_directory = f'../model_inputs/{subfolder}/'
            for content in os.listdir(input_directory):
                if '.csv' in content:
                    continue
                if content != assay:
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

                for model_name in ['rf']:

                    for fold in range(5):
                        
                        # for hyperparameter combination
                        if not (model_name == 'tabpfn'):
                            best_combi = None
                            best_score = -1
                            for combi in hyperparameters[model_name]:
                                combi_dict = eval(combi)
                                combi_val = 0
                                for ht_fold in range(4):
                                    training_samples_original = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/train.txt'
                                    test_samples_original = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/test.txt'
                                    
                                    training_samples = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/train_physchem.txt'
                                    test_samples = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/test_pyschem.txt'
                                    
                                    
                                    with open(training_samples_original, 'r') as original_train_file:
                                        train_samples_list = original_train_file.read().splitlines()
                                    final_train_samples = []
                                    for sample in train_samples_list:
                                        if sample in physchem_samples:
                                            final_train_samples.append(sample)
                                    with open(training_samples, 'w') as train_samples_final:
                                        for sample in final_train_samples:
                                            train_samples_final.write(f'{sample}\n')
                                            
                                    with open(test_samples_original, 'r') as original_test_file:
                                        test_samples_list = original_test_file.read().splitlines()
                                    final_test_samples = []
                                    for sample in test_samples_list:
                                        if sample in physchem_samples:
                                            final_test_samples.append(sample)
                                    with open(test_samples, 'w') as test_samples_final:
                                        for sample in final_test_samples:
                                            test_samples_final.write(f'{sample}\n')
                                    
                                    features = f'../model_inputs/all_feature_names_{feature_type}.txt'

                                    
                                    
                                    feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/{feature_type}.csv'

                                    model_specific = '{ "physchem": "", "maccs": "", "morgan": "' + \
                                        feature_matrix_path + \
                                        '", "smiles": "", "morphological": "", "response": "' + response + \
                                        '", "hyperparameters": ' + \
                                        combi + '}'
                                    output_dir = f'../temp/{subfolder}/{content}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/'
                                    os.makedirs(output_dir, exist_ok=True)
                                    config_file_path = f'{output_dir}/{dr_name}_{model_name}_{feature_type}_config.json'
                                    analysis_name = f'{model_name}_{feature_type}_{dr_name}_physchem_samples'
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

                        feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/embeddings.csv'

                        features = f'../model_inputs/all_feature_names_{feature_type}.txt'

                        training_samples_original = f'{path_to_CV_folds}/fold{fold}/train.txt'
                        test_samples_original = f'{path_to_CV_folds}/fold{fold}/test.txt'
                        
                        training_samples = f'{path_to_CV_folds}/fold{fold}/train_physchem.txt'
                        test_samples = f'{path_to_CV_folds}/fold{fold}/test_pyschem.txt'
                        
                        
                        with open(training_samples_original, 'r') as original_train_file:
                            train_samples_list = original_train_file.read().splitlines()
                        final_train_samples = []
                        for sample in train_samples_list:
                            if sample in physchem_samples:
                                final_train_samples.append(sample)
                        with open(training_samples, 'w') as train_samples_final:
                            for sample in final_train_samples:
                                train_samples_final.write(f'{sample}\n')
                                
                        with open(test_samples_original, 'r') as original_test_file:
                            test_samples_list = original_test_file.read().splitlines()
                        final_test_samples = []
                        for sample in test_samples_list:
                            if sample in physchem_samples:
                                final_test_samples.append(sample)
                        with open(test_samples, 'w') as test_samples_final:
                            for sample in final_test_samples:
                                test_samples_final.write(f'{sample}\n')
            
            
                        output_dir = f'{output_final_results}/fold{fold}/'
                        model_specific = '{ "physchem": "", "maccs": "", "morgan": "' + \
                            feature_matrix_path + \
                            '", "smiles": "", "morphological": "", "response": "' + response + \
                            '", "hyperparameters": ' + \
                            best_combi + '}'
                        analysis_name = f'{model_name}_best_combi_{feature_type}_{dr_name}_physchem_samples'
                        config_file_path = f'{output_dir}/{dr_name}_{feature_type}_{model_name}_config.json'
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
                        with open(f'{output_final_results}/fold{fold}/final_models_{feature_type}_{dr_name}_physchem_samples.txt', 'a') as output:
                            output.write(
                                f'\n{model_name}\tfold{fold}\t{mcc}\t{auroc}')


if __name__ == '__main__':
    main()
