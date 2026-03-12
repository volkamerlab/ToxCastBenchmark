import os
import glob
import subprocess


def main():
    task = 'classification'
    fs_main = '../fs_methods/main_feature_selection.py'
    json_config_gen = "..//fs_methods/json_config_generator.sh"
    num_features = 100


    for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:

        directory = f'../model_inputs/{subfolder}/'

        for content in os.listdir(directory):
            if '.csv' in content:
                continue

            path_to_CV_folds = f'{directory}/{content}/'
            pattern = f'{content}-*_binary_response.csv'
            
            matching_files = glob.glob(f'../ToxCastDownloads/binary_responses_and_datasail_input_files/{subfolder}/{pattern}')
            print(matching_files)
            if matching_files:
                response = matching_files[0]
            else:
                continue

            model_specific = {
                'pca': '{"random_state": 42, "file_name": ' + f'"embeddings_transformed_matrix_pca.csv"' + '}',
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
                                    fs_name, str(num_features), f'{output_dir}/embeddings.csv', output_dir, samples, model_specific[fs_name], 'embeddings', config_file_path])
                    subprocess.run(['python3', fs_main, config_file_path])

                for ht_fold in range(4):
                    samples = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/train.txt'
                    output_dir = f'{path_to_CV_folds}/fold{fold}/hyperparameter_tuning/fold{ht_fold}/'
                    for fs_name in ['pca', 'mrmr', 'mi', 'variance']:

                        config_file_path = f'{output_dir}/{fs_name}_config.json'
                        subprocess.run(args=['bash', json_config_gen,
                                                fs_name, str(num_features), f'{output_dir}/embeddings.csv', output_dir, samples, model_specific[fs_name], 'embeddings', config_file_path])
                        subprocess.run(
                            ['python3', fs_main, config_file_path])


if __name__ == '__main__':
    main()
