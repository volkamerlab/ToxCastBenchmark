import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def check_missing_compounds():
    physchem = pd.read_csv('../model_inputs/physchem_properties.csv', sep = '\t', index_col = 0)



    cols_with_nan_or_inf = physchem.columns[
        physchem.isna().any() |
        np.isinf(physchem.select_dtypes(include=np.number)).reindex(columns=physchem.columns, fill_value=False).any()
    ]



    physchem = physchem.replace([np.inf, -np.inf], np.nan).dropna()
    dropped_rows = {}
    assay_names = []
    for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:

            directory = f'../model_inputs/{subfolder}/'

            for name in os.listdir(directory):
                if '.csv' in name:
                    continue
                assay_names.append(name)
                pattern = f'{name}-*_datasail_input.csv'

                matching_files = glob.glob(
                    f'../ToxCastDownloads/binary_responses_and_datasail_input_files/{subfolder}/{pattern}')
                if matching_files:
                    response = matching_files[0]
                res = pd.read_csv(response, sep =  '\t', index_col = 0)
                n_before = len(res)
                n_after = (~res.index.isin(physchem.index)).sum()
                dropped_rows[name] = (100 -(((n_before - n_after)/n_before ) * 100))

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(dropped_rows.values())), dropped_rows.values())

    

    with open("../model_inputs/sorted_assays.txt", "w") as f:
        for assay, _ in sorted(
            dropped_rows.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            f.write(f"{assay}\n")
    plt.ylabel("% Compounds dropped")
    plt.xlabel("Assay")
    #plt.xticks(range(len(assay_names)), assay_names, rotation=90)
    plt.title("Dropped compounds assay file")

    plt.tight_layout()

    plt.savefig('dropped_compounds.png')
    return dropped_rows
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def plot_physchem_drop_correlation(
    results_df,
    drop_percentages,
    metric_col="mcc",
    assay_col="assay",
    repr_col="representation",
    physchem_name="physchem",
):
    """
    Compare physchem performance advantage against
    percentage of dropped compounds.

    Returns
    -------
    assay_df : pd.DataFrame
    spearman_rho : float
    p_value : float
    """

    # Average over folds first
    perf = (
        results_df
        .groupby(
            [assay_col, "model", "dr_method", repr_col]
        )[metric_col]
        .mean()
        .reset_index()
    )

    rows = []

    for assay, tmp in perf.groupby(assay_col):

        physchem_mean = tmp.loc[
            tmp[repr_col] == physchem_name,
            metric_col
        ].mean()

        other_mean = tmp.loc[
            tmp[repr_col] != physchem_name,
            metric_col
        ].mean()

        rows.append({
            assay_col: assay,
            "physchem_mean": physchem_mean,
            "other_mean": other_mean,
            "advantage": physchem_mean - other_mean,
            "drop_percent": drop_percentages.get(assay)
        })

    assay_df = pd.DataFrame(rows).dropna()

    rho, ps = spearmanr(
        assay_df["drop_percent"],
        assay_df["advantage"]
    )
    pcc, pp = pearsonr(assay_df["drop_percent"],
        assay_df["advantage"])

    # -----------------------------------------------------
    # Plot 1: Main correlation plot
    # -----------------------------------------------------

    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=assay_df,
        x="drop_percent",
        y="advantage",
        ci = False
    )



    plt.title(
        f"Physchem Advantage vs Dropped Compounds\n"
        f"SCC={rho:.3f} (p-val= {ps:.3f}), PCC = {pcc:.3f} (p-val= {pp:.3f})"
    )

    plt.xlabel("% compounds dropped")
    plt.ylabel(
        "Physchem performance - Other representations"
    )

    plt.tight_layout()
    plt.savefig('../plotting_results/correlation_plots.pdf', dpi = 600)

    # -----------------------------------------------------
    # Plot 2: Ranked assays
    # -----------------------------------------------------

    ranked = assay_df.sort_values("advantage")

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=ranked,
        x=assay_col,
        y="advantage",
        color="steelblue"
    )


    plt.xticks(rotation=90)

    plt.ylabel(
        "Physchem advantage"
    )

    plt.title(
        "Physchem Advantage by Assay"
    )

    plt.tight_layout()
    plt.savefig('../plotting_results/physchem_advantage_by_assay.png')

    # -----------------------------------------------------
    # Plot 3: Drop percentage distribution
    # -----------------------------------------------------

    plt.figure(figsize=(7, 4))

    sns.histplot(
        assay_df["drop_percent"],
        bins=15,
        kde=True
    )

    plt.xlabel("% dropped compounds")
    plt.title("Distribution of Dropped Compounds")

    plt.tight_layout()
    plt.savefig('../plotting_results/dropped_compounds_versus_performance_diff.png', dpi = 600)

    return assay_df, rho

def main():

    final_results = None
    num_models = 5 

    for dr_method in ['MI', 'mrmr', 'pca', 'variance', 'none']:
        for feature_type in ['morgan', 'embeddings', 'maccs', 'physchem']:
            for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:
                directory = f'../model_outputs/{subfolder}/'

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
                                f'{directory}/{content}/fold{fold}/final_models_{feature_type}_{dr_method}.txt', sep='\t', names=['model', 'fold', 'mcc', 'auroc'])
                            new_df.dropna(inplace = True)
                            new_df['dr_method'] = [
                                f'{dr_method}' for _ in range(len(new_df.index))]
                            new_df['representation'] = [f'{feature_type}' for _ in range(len(new_df.index))]
                        except:
                            assay_done = False
                            break
                        if len(new_df) < num_models:
                            assay_done = False
                            break

                        if results_df is None:
                            results_df = new_df
                        else:

                            results_df = pd.concat(
                                [results_df, new_df], axis=0)

                    if not assay_done:
                        continue
                    else:

                        results_df.reset_index(inplace=True, drop=True)

                        results_df['assay'] = [
                            content for _ in range(len(results_df))]
                        if final_results is None:
                            final_results = results_df
                        else:
                            final_results = pd.concat(
                                [final_results, results_df.copy(deep=True)])
                            final_results.reset_index(inplace=True, drop=True)



    dropped_rows = check_missing_compounds()
    plot_physchem_drop_correlation(results_df = final_results, drop_percentages = dropped_rows)


if __name__ == "__main__":
    main()