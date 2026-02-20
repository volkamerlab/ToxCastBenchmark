import os
import glob
import subprocess
import sys

def main(delta):
    with open(f'/home/lisa-marie-rolli/comptox_benchmark/data_processing/considered_hormone_assays_epsilon_and_delta{delta}.txt', 'r') as relevant_assay_file:
        relevant_assays = relevant_assay_file.read().splitlines()
    files = {'physchem': '/home/lisa-marie-rolli/comptox_benchmark/automatic_ToxCast_Query/physchem_properties.csv',
             'maccs': '/home/lisa-marie-rolli/comptox_benchmark/automatic_ToxCast_Query/maccs.csv', 'morgan': '/home/lisa-marie-rolli/comptox_benchmark/automatic_ToxCast_Query/morgan.csv'}
    feature_types = ['physchem', 'maccs', 'morgan']

    task = 'classification'
    fs_main = '/home/lisa-marie-rolli/comptox_benchmark/fs_methods/main_feature_selection.py'
    json_config_gen = "/home/lisa-marie-rolli/comptox_benchmark/fs_methods/json_config_generator.sh"
    num_features = 100

    for feature_type in feature_types:
        for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:
        #for subfolder in ['steroidal']:
            directory = f'/home/lisa-marie-rolli/comptox_benchmark/ToxCast_Assays_Endpoint_Results/{subfolder}/'

            for content in os.listdir(directory):
                if not (content in relevant_assays):
                    continue
                if '.csv' in content:
                    continue
                path_to_CV_folds = f'{directory}/{content}/'
                pattern = f'{content}-*_binary_response.csv'
                
                matching_files = glob.glob(f'{directory}/{pattern}')

                if matching_files:
                    response = matching_files[0]
                else:
                    break

                model_specific = {
                    'pca': '{"random_state": 42, "file_name": ' + f'"{feature_type}_transformed_matrix_pca.csv"' + '}',
                    'mrmr':  '{ "response": "' + response + '", "task": "' + task + '"}',
                    'mi':  '{ "response": "' + response + '", "task": "' + task + '"}',
                    'variance': '{}',
                }
                for fold in range(5):

                    samples = f'{path_to_CV_folds}/fold{fold}/train.txt'
                    output_dir = f'{path_to_CV_folds}/fold{fold}/'
                    for fs_name in ['pca', 'mrmr', 'mi', 'variance']:
                        config_file_path = f'{output_dir}/{fs_name}_config.json'
                        subprocess.run(['bash', json_config_gen,
                                        fs_name, str(num_features), files[feature_type], output_dir, samples, model_specific[fs_name], feature_type, config_file_path])
                        subprocess.run(['python3', fs_main, config_file_path])

                    for ht_fold in range(4):
                        samples = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/train.txt'
                        output_dir = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/'
                        for fs_name in ['pca', 'mrmr', 'mi', 'variance']:
                            config_file_path = f'{output_dir}/{fs_name}_config.json'
                            subprocess.run(args=['bash', json_config_gen,
                                                 fs_name, str(num_features), files[feature_type], output_dir, samples, model_specific[fs_name], feature_type, config_file_path])
                            subprocess.run(
                                ['python3', fs_main, config_file_path])


if __name__ == '__main__':
    main(sys.argv[1])
