import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon


def compare_wilcoxon(df, outdir, metric = 'mcc'):
    """
    If metric is mcc, aggregate MCC values across folds per assay using Fisher z
    transformation, then compare rf and chemeleon at the assay level
    using a paired Wilcoxon signed-rank test.

    Otherwise simply perform wilcoxon test for AUROC
    
    Parameters
    ----------
    df : pd.DataFrame
        Columns required:
        ['model', 'fold', 'mcc', 'auroc, 'assay']

    Returns
    -------
    statistic : float
    pvalue : float
    paired_df : pd.DataFrame
        Assay-level Fisher-averaged MCCs.
    """

    # Avoid infinities if MCC happens to be exactly +/-1
    eps = 1e-10

    df_tmp = df.copy()
    if metric == 'mcc':
        # Fisher transform
        df_tmp["mcc_clipped"] = df_tmp["mcc"].clip(
            -1 + eps,
            1 - eps
        )

        df_tmp["z"] = np.arctanh(df_tmp["mcc_clipped"])

        # Average z values across folds for each assay/model
        assay_avg = (
            df_tmp.groupby(["assay", "model"])["z"]
            .mean()
            .reset_index()
        )

        # Transform back to MCC scale
        assay_avg["mcc_fisher"] = np.tanh(assay_avg["z"])
        values_col = 'mcc_fisher'
    else:
        assay_avg = (
                    df_tmp.groupby(["assay", "model"])["auroc"]
                    .mean()
                    .reset_index()
                )
        values_col = 'auroc'
    # Create paired assay-level table
    paired_df = (
        assay_avg.pivot(
            index="assay",
            columns="model",
            values=values_col
        )
        .dropna(subset=["rf", "chemeleon"])
    )

    # Paired differences
    diff = paired_df["rf"] - paired_df["chemeleon"]

    # Wilcoxon test
    statistic, pvalue = wilcoxon(
        paired_df["rf"],
        paired_df["chemeleon"],
        alternative="two-sided"
    )

    print(f"Number of assays: {len(paired_df)}")
    print(f"Median RF: {paired_df['rf'].median():.4f}")
    print(f"Median Chemeleon: {paired_df['chemeleon'].median():.4f}")
    print(f"Median difference (RF - Chemeleon): {diff.median():.4f}")
    print(f"RF wins: {(diff > 0).sum()}")
    print(f"Chemeleon wins: {(diff < 0).sum()}")
    print(f"Ties: {(diff == 0).sum()}")
    print(f"Wilcoxon statistic: {statistic:.4f}")
    print(f"P-value: {pvalue:.6g}")

    # Plot
    plot_df = paired_df.reset_index()[["rf", "chemeleon"]].melt(
        var_name="model",
        value_name=metric
    )

    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(data=plot_df, x="model", y=metric, color = 'lightgrey')
    sns.stripplot(
        data=plot_df,
        x="model",
        y=metric,
        color="red",
        #alpha=0.5,
        jitter=True
    )

    plt.title(
        f"Assay-level Fisher-averaged MCC\nWilcoxon p={pvalue:.3g}"
    )
    if metric == 'mcc':
        plt.ylabel("MCC")
    else:
        plt.ylabel('AUROC')
    ax.set_xticklabels(["RF", "CheMeleon"])
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(f'{outdir}/chemeleon_vs_rf_{metric}.pdf', dpi = 600)

    return statistic, pvalue, paired_df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparing Chemeleon to RF")
    parser.add_argument("-d", '--directory',
                        help="input_directory", default='../model_outputs/')
    parser.add_argument("--output_dir", '-o',
                        help="Output_directory", default='../plotting_results/')
    parser.add_argument("--metric", '-m',
                        help="Metric", default='mcc')

    return parser.parse_args()

def main(arge):
    final_results = None
    out_dir = args.output_dir
    feature_type  = 'physchem'
    dr_method  = 'none'
    for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:
        directory = f'{args.directory}/{subfolder}/'

        for content in os.listdir(directory):

            if '.csv' in content:
                continue
            if '.png' in content:
                continue

            results_df = None
            for fold in range(5):

                new_df = pd.read_csv(
                    f'{directory}/{content}/fold{fold}/final_models_{feature_type}_{dr_method}.txt', sep='\t', names=['model', 'fold', 'mcc', 'auroc'])
                new_df.dropna(inplace = True)
                
                
                chemeleon_results = pd.read_csv(f'{directory}/{content}/fold{fold}/final_models_chemeleon.txt', sep = '\t', names=['model', 'fold', 'mcc', 'auroc'])
                chemeleon_results.dropna(inplace = True)
                
                new_df = pd.concat([new_df, chemeleon_results], axis = 0)
                new_df = new_df.loc[new_df['model'].isin(['rf', 'chemeleon']), :]
                
                if results_df is None:
                    results_df = new_df
                else:

                    results_df = pd.concat(
                        [results_df, new_df], axis=0)



            results_df.reset_index(inplace=True, drop=True)

            results_df['assay'] = [
                content for _ in range(len(results_df))]
            if final_results is None:
                final_results = results_df
            else:
                final_results = pd.concat(
                    [final_results, results_df.copy(deep=True)])
                final_results.reset_index(inplace=True, drop=True)
    compare_wilcoxon(final_results, out_dir, metric = args.metric)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)