import numpy as np
import pandas as pd
from rdkit import DataStructs
import os
import glob
K_NEIGHBORS = 5


def nn_concordance(df, fps, k=K_NEIGHBORS):
    labels = df['response'].values
    n = len(fps)
    nn_sim = np.zeros(n)
    nn_same_label = np.zeros(n, dtype=bool)
    knn_active_frac = np.zeros(n)

    for i in range(n):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = -1  # exclude self
        nn_idx = np.argmax(sims)
        nn_sim[i] = sims[nn_idx]
        nn_same_label[i] = (labels[nn_idx] == labels[i])
        topk = np.argpartition(-sims, k)[:k]
        knn_active_frac[i] = labels[topk].mean()

    return {
        'mean_NN_tanimoto_sim': nn_sim.mean(),
        'NN_label_concordance_overall': nn_same_label.mean(),
        'NN_label_concordance_actives': nn_same_label[labels == 1].mean(),
        'NN_label_concordance_inactives': nn_same_label[labels == 0].mean(),
        f'{k}NN_active_frac_among_active_query': knn_active_frac[labels == 1].mean(),
        f'{k}NN_active_frac_among_inactive_query': knn_active_frac[labels == 0].mean(),
    }


def calculate_correlation(df):
    assays_with_best_performances = pd.read_csv(
        'assays_with_best_performance.csv', sep='\t')
    # Keep only assay and mcc_avg from the first dataframe
    best_performances_sub = assays_with_best_performances[['assay', 'mcc_avg']]

    # Merge with the second dataframe on assay
    merged = best_performances_sub.merge(df, on='assay', how='inner')

    # Compute correlation of mcc_avg with all other columns
    corrs = (
        merged.drop(columns='assay')
        .corr()['mcc_avg']
        .drop('mcc_avg')
        .sort_values(ascending=False)
    )

    corrs.to_csv('correlations.csv', sep='\t')


def main():
    fingerprints = pd.read_csv(
        '../model_inputs/morgan.csv', sep='\t', index_col=0)
    nn_rows = []
    for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:

        directory = f'../model_inputs/{subfolder}/'

        for name in os.listdir(directory):
            if '.csv' in name:
                continue

            pattern = f'{name}-*_datasail_input.csv'

            matching_files = glob.glob(
                f'../ToxCastDownloads/binary_responses_and_datasail_input_files/{subfolder}/{pattern}')
            if matching_files:
                response = matching_files[0]
            else:
                continue
            df = pd.read_csv(response, sep='\t')
            fps = [
                DataStructs.CreateFromBitString(''.join(map(str, row.astype(int))))for row in fingerprints.loc[df['compound'].values, :].to_numpy()]

            row = {'assay': name}
            row.update(nn_concordance(df, fps))
            nn_rows.append(row)

    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv('nn_dataframe.csv', sep='\t', index=False)

    calculate_correlation(nn_df)


if __name__ == '__main__':
    main()
