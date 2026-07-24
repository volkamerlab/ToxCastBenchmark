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
K_NEIGHBORS = 5
ROWS_PER_PAGE = 20
filterwarnings("ignore")



MAX_SUB_HEAVY_ATOMS = 8      # max heavy atoms in the swapped substituent
MAX_SUB_CORE_RATIO = 0.5     # substituent must be <= this fraction of the core's size


import numpy as np

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

def _heavy_atoms(smi_with_dummy):
    m = Chem.MolFromSmiles(smi_with_dummy.replace('[*:1]', '[H]'))
    return m.GetNumHeavyAtoms() if m else 0


def get_core_substituent_pairs(mol, max_sub_heavy=MAX_SUB_HEAVY_ATOMS, max_sub_core_ratio=MAX_SUB_CORE_RATIO):
    """Single-cut MMP fragmentation of one molecule. Returns list of (core_smiles, substituent_smiles)
    for cuts where the substituent passes the 'small change' size filters."""
    try:
        frags = rdMMPA.FragmentMol(
            mol, maxCuts=1, resultsAsMols=False, maxCutBonds=20)
    except Exception:
        return []
    out = []
    for const, pair in frags:
        if '.' not in pair:
            continue
        parts = pair.split('.')
        if len(parts) != 2:
            continue
        a, b = parts
        ha_a, ha_b = _heavy_atoms(a), _heavy_atoms(b)
        if ha_a == 0 or ha_b == 0:
            continue
        if ha_a <= ha_b:
            sub, core, ha_sub, ha_core = a, b, ha_a, ha_b
        else:
            sub, core, ha_sub, ha_core = b, a, ha_b, ha_a
        if ha_sub > max_sub_heavy or ha_sub > max_sub_core_ratio * ha_core:
            continue
        out.append((core, sub))
    return out


def mmp_analysis(df):
    """Finds all MMPs within one assay's molecule set and flags activity cliffs
    (matched pairs with differing response). Returns (summary_dict, detail_dict)."""
    smiles_list = df['smiles'].tolist()
    labels = df['response'].values
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    # core_smiles -> list of (mol_idx, substituent_smiles)
    core_to_entries = {}
    for idx, m in enumerate(mols):
        if m is None:
            continue
        for core, sub in get_core_substituent_pairs(m):
            core_to_entries.setdefault(core, []).append((idx, sub))

    seen_pairs, cliff_pairs = set(), set()
    for core, entries in core_to_entries.items():
        n = len(entries)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                idx1, sub1 = entries[i]
                idx2, sub2 = entries[j]
                if idx1 == idx2 or sub1 == sub2:
                    continue
                key = (min(idx1, idx2), max(idx1, idx2))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if labels[idx1] != labels[idx2]:
                    cliff_pairs.add(key)

    n_active = int(labels.sum())
    actives_in_cliff, actives_in_mmp = set(), set()
    for (i, j) in cliff_pairs:
        for k in (i, j):
            if labels[k] == 1:
                actives_in_cliff.add(k)
    for (i, j) in seen_pairs:
        for k in (i, j):
            if labels[k] == 1:
                actives_in_mmp.add(k)

    summary = {
        'n_molecules': len(df),
        'n_matched_pairs_total': len(seen_pairs),
        'n_activity_cliff_pairs': len(cliff_pairs),
        'frac_mmp_that_are_cliffs': len(cliff_pairs) / len(seen_pairs) if seen_pairs else np.nan,
        'n_active': n_active,
        'cliff_pairs_per_active': len(cliff_pairs) / n_active if n_active else np.nan,
        'frac_actives_in_mmp_series': len(actives_in_mmp) / n_active if n_active else np.nan,
        'frac_actives_involved_in_cliff': len(actives_in_cliff) / n_active if n_active else np.nan,
        # of actives that HAVE a close matched-pair analog in the dataset, how often is that analog's label flipped?
        'frac_of_analog_actives_that_are_cliffy': (len(actives_in_cliff) / len(actives_in_mmp)) if actives_in_mmp else np.nan,
    }
    detail = {'seen_pairs': seen_pairs, 'cliff_pairs': cliff_pairs,
              'mols': mols, 'labels': labels, 'smiles': smiles_list}
    return summary, detail


def plot_nearest_neighbours(df, i, topk, pdf_pages, labels):

    df.dropna(inplace=True)
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)

    group_mols = [df.loc[i, "mol"]] + [
        df.loc[j, "mol"] for j in topk
    ]

    legends = (
        [f"Query: {df.loc[i,'compound']}\nres: {labels[i]}"]
        + [f'{df.loc[j, "compound"]}\nres: {labels[j]}' for j in topk]
    )

    mcs = rdFMCS.FindMCS(
        group_mols,
        ringMatchesRingOnly=True,
        completeRingsOnly=True
    )

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

    highlight_atoms = [
        mol.GetSubstructMatch(mcs_mol)
        for mol in group_mols
    ]

    row_img = Draw.MolsToGridImage(
        group_mols,
        molsPerRow=len(group_mols),
        legends=legends,
        highlightAtomLists=highlight_atoms,
        subImgSize=(250, 250),
        returnPNG=False
    )

    pdf_pages.append(row_img.convert("RGB"))
    return pdf_pages


def nn_concordance(df, fps, k=K_NEIGHBORS, name=None, test_folds = None):
    assay_compound_map = {
        'NVS_NR_bER': ['DTXSID0020573', 'DTXSID0020814'],
        'OT_AR_ARELUC_AG_1440': ['DTXSID1024704', 'DTXSID1029124'],
        'TOX21_AR_LUC_MDAKB2_Antagonist_0.5nM_R1881': ['DTXSID0022220', 'DTXSID0025654'],
        'TOX21_ERb_BLA_Agonist_ratio': ['DTXSID2022127', 'DTXSID7020685']
    }
    labels = df['response'].values
    n = len(fps)
    nn_sim = np.zeros(n)
    nn_same_label = np.zeros(n, dtype=bool)
    knn_active_frac = np.zeros(n)
    all_rows = []
    pred_nn = {}
    for i, compound in enumerate(df['compound'].values):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        
        sims[i] = -1  # exclude self
        nn_idx = np.argmax(sims)
        nn_sim[i] = sims[nn_idx]
        nn_same_label[i] = (labels[nn_idx] == labels[i])
        pred_nn[compound] = labels[nn_idx]
        topk = np.argpartition(-sims, k)[:k]
        knn_active_frac[i] = labels[topk].mean()
        '''if compound in assay_compound_map[name]:

            all_rows = plot_nearest_neighbours(
                df=df, i=i, topk=topk, pdf_pages=all_rows, labels=labels)

    pages = []

    for start in range(0, len(all_rows), ROWS_PER_PAGE):
        chunk = all_rows[start:start + ROWS_PER_PAGE]
        page = combine_rows(chunk)
        pages.append(page)

    pages[0].save(
        f"../plotting_results/nn_mcs_report_show_only_selected_{name}.pdf",
        save_all=True,
        append_images=pages[1:]
    )'''
    
    if not test_folds is None:
        nn_pred = pd.DataFrame(pred_nn.items(), columns = ["compound", "prediction"])
        merged = nn_pred.merge(df, on = 'compound')
        merged.set_index('compound', inplace = True, drop = True)
        mccs= []
        for test_samples in test_folds:
            sub_df = merged.loc[test_samples, ['response', 'prediction']]
            mccs = matthews_corrcoef(merged['response'].values, merged['prediction'].values)
        mean_mcc = fisher_mean_mcc(mccs)
        with open('assays_baseline_mcc.csv', 'a') as output:
            output.write(f'{name}\t{mean_mcc}\n')
        
        nn_pred.to_csv(f'../model_outputs/baseline_predictions/{name}_nn_pred.csv', sep ='\t', index = False)
            
    return {
        'mean_NN_tanimoto_sim': nn_sim.mean(),
        'NN_label_concordance_overall': nn_same_label.mean(),
        'NN_label_concordance_actives': nn_same_label[labels == 1].mean(),
        'NN_label_concordance_inactives': nn_same_label[labels == 0].mean(),
        f'{k}NN_active_frac_among_active_query': knn_active_frac[labels == 1].mean(),
        f'{k}NN_active_frac_among_inactive_query': knn_active_frac[labels == 0].mean(),
    }


def combine_rows(rows):
    width = max(img.width for img in rows)
    height = sum(img.height for img in rows)

    page = Image.new("RGB", (width, height), "white")

    y = 0
    for img in rows:
        page.paste(img, (0, y))
        y += img.height

    return page


def calculate_correlation(df, outpath):
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

    corrs.to_csv(outpath, sep='\t')


def main():
    fingerprints = pd.read_csv(
        '../model_inputs/morgan.csv', sep='\t', index_col=0)
    nn_rows = []
    mmp_rows = []
    with open('assays_baseline_mcc.csv', 'w') as output:
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

            row = {'assay': name}
            mmp_row = {'assay': name}
            #row.update(nn_concordance(df, fps, name=name, test_folds = test_folds))
            summary, detail = mmp_analysis(df)
            mmp_row.update(summary)
            nn_rows.append(row)
            mmp_rows.append(mmp_row)
    nn_df = pd.DataFrame(nn_rows)
    mmp_df = pd.DataFrame(mmp_rows)
    nn_df.to_csv('nn_dataframe.csv', sep='\t', index=False)
    mmp_df.to_csv('mmp_dataframe.csv', sep='\t', index=False)
    calculate_correlation(nn_df, 'correlations_nn.csv')
    calculate_correlation(mmp_df, 'correlations_mmp.csv')


if __name__ == '__main__':
    main()
