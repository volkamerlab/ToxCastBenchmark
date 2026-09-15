from PIL import Image
from rdkit.Chem import rdMMPA
from warnings import filterwarnings
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit import DataStructs
import os
import glob
from sklearn.metrics import matthews_corrcoef
import sys
filterwarnings("ignore")

def fisher_mean_mcc(mcc_values):
    """
    Compute the Fisher z-transformed mean of MCC values.
    
    Parameters
    ----------
    mcc_values : array-like
        List of MCC values in [-1, 1].

    Returns
    -------
    float
        Back-transformed mean MCC.
    """
    mcc_values = np.asarray(mcc_values, dtype=float)

    # Avoid infinities for values exactly at ±1
    eps = np.finfo(float).eps
    mcc_values = np.clip(mcc_values, -1 + eps, 1 - eps)

    z = np.arctanh(mcc_values)
    z_mean = np.mean(z)

    return np.tanh(z_mean)

def nn_concordance(df, fps, name=None, test_folds = None):
    
    labels = df['response'].values
    pred_nn = {}
    compounds = df['compound'].values
    for i, compound in enumerate(compounds):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        
        sims[i] = -1  # exclude self
        
        for fold in test_folds:
            if compound in fold:
                test_set = fold
                break
        mask = ~df['compound'].isin(test_set)
        best_idx = np.argmax(sims[mask])
        pred_nn[compound] = labels[best_idx]
        


    if not test_folds is None:
        nn_pred = pd.DataFrame(pred_nn.items(), columns = ["compound", "prediction"])
        merged = nn_pred.merge(df, on = 'compound')
        merged.set_index('compound', inplace = True, drop = True)
        mccs= []
        for test_samples in test_folds:
            sub_df = merged.loc[test_samples, ['response', 'prediction']]
            mccs = matthews_corrcoef(merged['response'].values, merged['prediction'].values)
        mean_mcc = fisher_mean_mcc(mccs)
        with open('assays_baseline_mcc_new.csv', 'a') as output:
            output.write(f'{name}\t{mean_mcc}\n')
        
        nn_pred.to_csv(f'../model_outputs/baseline_predictions/{name}_nn_pred.csv', sep ='\t', index = False)

def main():
    fingerprints = pd.read_csv(
            '../model_inputs/morgan.csv', sep='\t', index_col=0)
    with open('assays_baseline_mcc_new.csv', 'w') as output:
        output.write('assay\tmcc_avg\n')
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
            test_folds = []
            for fold in range(5):
                with open(f'../model_inputs/{subfolder}/{name}/fold{fold}/test.txt', 'r') as test_sample_file:
                    test_samples = test_sample_file.read().splitlines()
                    test_folds.append(test_samples)
            
            df = pd.read_csv(response, sep='\t')
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)
            fps = [
                DataStructs.CreateFromBitString(''.join(map(str, row.astype(int))))for row in fingerprints.loc[df['compound'].values, :].to_numpy()]
            print(df)
            nn_concordance(df, fps, name=name, test_folds = test_folds)
            
if __name__=='__main__':
    main()