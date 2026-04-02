import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, f
import math
import scikit_posthocs as sp
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import argparse

custom_palette = {
    "rf": "#c82254",
    "mlp": "#d7df23",
    "cat_boost": "#004877",
    "svm": "#000000"
}
model_order = ["rf", "cat_boost", 'mlp', 'svm']


# Fisher Z-transformation
def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))  # avoid ±1

# Inverse Fisher Z-transformation


def inverse_fisher_z(z):
    return np.tanh(z)


def cd_plot_for_nemenyi(final_results, dr_method, model_name_map, feature_type, feature_name_map, output_dir):

    means = (
        final_results.groupby(['assay', 'model'], as_index=False)['auroc']
        .mean()
    )

    wide = means.pivot(index='assay', columns='model', values='auroc')

    # Rank per assay (rank 1 = best; higher auroc is better)

    ranks = wide.rank(axis=1, ascending=False, method='average')

    # Average ranks

    avg_ranks = ranks.mean(axis=0)
    models = [model_name_map[m] for m in avg_ranks.index]
    wide = means.pivot(index='assay', columns='model', values='auroc')
    avg_auroc = wide.mean(axis=0)
    avg_auroc = avg_auroc.apply(inverse_fisher_z)

    pretty_names = {
        m: f"{model_name_map [m]}\navg. ROC-AUC={avg_auroc[m]:.3f}" for m in avg_ranks.index}
    avg_ranks.index = [pretty_names[m] for m in avg_ranks.index]

    # Nemenyi p-values (pairwise)

    pvals = sp.posthoc_nemenyi_friedman(ranks.values)
    pvals.index = models
    pvals.columns = models

    # generate p-value heat map
    plt.close('all')
    norm = TwoSlopeNorm(vmin=pvals.values.min(),
                        vmax=pvals.values.max(), vcenter=0.05)

    cmap = LinearSegmentedColormap.from_list(
        "p-values", ["#c82254", '#D3D3D3', "#004877"])
    mask = np.eye(len(pvals), dtype=bool)

    annot = np.empty(pvals.shape, dtype=object)
    for i in range(pvals.shape[0]):
        for j in range(pvals.shape[1]):
            if i == j:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{pvals.iat[i,j]:.1e}"

    sns.heatmap(
        pvals,
        cmap=cmap,
        mask=mask,
        norm=norm,
        vmin=0,
        fmt="",
        vmax=1,
        cbar_kws={"label": "p-value"},
        annot=annot,
    )

    plt.title(
        f"Model comparison across assays on \n{feature_name_map[feature_type]} using {dr_method}")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dr_method}_{feature_type}_nemenyi_pval_heatmap_AUROC.png',
                dpi=300, transparent=False)

    pvals.index = avg_ranks.index
    pvals.columns = avg_ranks.index

    # Generate CD diagram

    plt.close('all')
    color_palette = {pretty_names[m]: custom_palette[m]
                     for m in pretty_names.keys()}
    color_palette[f'{pretty_names["mlp"]}'] = '#a3a919'
    _ = sp.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=pvals,
        alpha=0.05,
        label_fmt_left="{label}\navg. rank: {rank:.2f}",
        label_fmt_right="{label}\navg. rank: {rank:.2f}",
        color_palette=color_palette
    )
    plt.title(
        f"Critical difference\nmodel comparison across assays on \n{feature_name_map[feature_type]} using {dr_method}")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dr_method}_{feature_type}_critical_difference_plot_nemenyi_AUROC.png',
                dpi=300, transparent=False)


def pairwise_friedman_nemenyi(data):
    # Pivot the data to have one column per model and one row per subject
    pivoted = data.pivot(index='fold', columns='model', values='auroc')
    # Drop rows with missing values (if any)
    pivoted = pivoted.dropna()

    # Run Friedman test with correction by Iman and Davenport (1980)
    friedman_stat, p = friedmanchisquare(*pivoted.values.T)
    N = len(np.unique(data['fold']))
    k = len(np.unique(data['model']))
    iman_davenport_correction = (
        (N - 1) * friedman_stat) / (N * (k - 1) - friedman_stat)

    # Compute p-value from F distribution
    p_id = 1 - f.cdf(iman_davenport_correction, k - 1, (k - 1) * (N - 1))

    # Nemenyi post-hoc test (only if Friedman is significant)
    if p_id < 0.05:
        nemenyi = sp.posthoc_nemenyi_friedman(pivoted.values)
        nemenyi.columns = pivoted.columns
        nemenyi.index = pivoted.columns

        results = []
        for i in range(len(nemenyi)):
            for j in range(i + 1, len(nemenyi)):
                model1 = nemenyi.index[i]
                model2 = nemenyi.columns[j]
                pval = nemenyi.iloc[i, j]
                results.append((model1, model2, pval))
        return results

    else:

        return []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation of 5 fold CV results comparing models across all assays with one feature-DR combination")
    parser.add_argument("-d", '--directory',
                        help="input_directory", default='../model_outputs/')
    parser.add_argument("--dr_method", help="DR method used", default='MI')
    parser.add_argument("--feature_type", '-f',
                        help="Feature type used", default='physchem')
    parser.add_argument("--output_dir", '-o',
                        help="Output_directory", default='../plotting_results/')
    parser.add_argument("--tabpfn_missing",
                        help="is TabPFN missing", default='False', type=str)
    return parser.parse_args()


def main(args):

    final_results = None
    dr_method = args.dr_method
    feature_type = args.feature_type
    out_dir = args.output_dir
    num_models = 5 if (not args.tabpfn_missing in [
                       'y', 'yes', 'true', 't', 'True']) else 4
    for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:
        directory = f'{args.directory}/{subfolder}/'

        for content in os.listdir(directory):

            if '.csv' in content:
                continue
            if '.png' in content:
                continue

            results_df = None
            assay_done = True
            for fold in range(5):
                try:
                    new_df = pd.read_csv(
                        f'{directory}/{content}/fold{fold}/final_models_{args.feature_type}_{dr_method}.txt', sep='\t', skiprows=1, names=['model', 'fold', 'mcc', 'auroc'])

                except:
                    assay_done = False
                    break

                if (args.tabpfn_missing in ['y', 'yes', 'true', 't', 'True']) and ('tabpfn' in new_df['model'].values):
                    # tabpfn is run for this setting, but we don't want to evaluate it
                    new_df = new_df.loc[new_df['model'] != 'tabpfn', :]
                if len(new_df) < num_models:
                    assay_done = False
                    break

                if results_df is None:
                    results_df = new_df
                else:
                    results_df = pd.concat([results_df, new_df], axis=0)

            if not assay_done:
                continue
            else:
                print(content)
                results_df.reset_index(inplace=True, drop=True)

                results_df['assay'] = [content for _ in range(len(results_df))]
                if final_results is None:
                    final_results = results_df
                else:
                    final_results = pd.concat(
                        [final_results, results_df.copy(deep=True)])
                    final_results.reset_index(inplace=True, drop=True)

        feature_name_map = {
            'physchem': 'physicochemical properties',
            'morgan': 'Morgan fingerprints',
            'maccs': 'MACCS fingerprints',
            'embeddings': 'Embeddings'
        }

        model_name_map = {
            "rf": "RF",
            "mlp": "MLP",
            "svm": "SVM",
            "cat_boost": "CatBoost",
        }

    if not (args.tabpfn_missing in ['y', 'yes', 'true', 't', 'True']):
        model_name_map["tabpfn"] = "TabPFN"
        custom_palette['tabpfn'] = "#6e6e6e"
        model_order.append('tabpfn')
    print(final_results)

    cd_plot_for_nemenyi(final_results, dr_method, model_name_map,
                        feature_type, feature_name_map, output_dir=out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
