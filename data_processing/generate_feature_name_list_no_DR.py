import pandas as pd


def main():
    for feature_type in ['maccs', 'morgan', 'physchem', 'embeddings']:
        if not (feature_type == 'embeddings'):
            if not (feature_type == 'physchem'):
                feature_matrix = pd.read_csv(
                    f'/local/lisa-marie.rolli/ToxCastBenchmark/model_inputs/{feature_type}.csv', sep='\t')
            else:
                feature_matrix = pd.read_csv(
                    f'/local/lisa-marie.rolli/ToxCastBenchmark/model_inputs/{feature_type}_properties.csv', sep='\t')
            with open(f'/local/lisa-marie.rolli/ToxCastBenchmark/model_inputs/all_feature_names_{feature_type}.txt', 'w') as output:
                for i in feature_matrix.columns.to_list():
                    if i == 'compound':
                        continue
                    output.write(f'{i}\n')
        else:
            with open('/local/lisa-marie.rolli/ToxCastBenchmark/model_inputs/all_feature_names_embeddings.txt', 'w') as output:
                for i in range(0, 767 + 1):
                    output.write(f'{i}\n')


if __name__ == '__main__':
    main()
