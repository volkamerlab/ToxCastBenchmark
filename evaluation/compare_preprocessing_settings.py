import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, f
import scikit_posthocs as sp


# ============================================================
# Read assays
# ============================================================

with open("../model_inputs/assays.txt", "r") as file:
    assays = file.read().splitlines()


# ============================================================
# Settings
# ============================================================

settings = {
    "Morgan, all compounds":
        "final_models_morgan_none.txt",
    "Morgan, physchem samples":
        "final_models_morgan_none_physchem_samples.txt",
    "Morgan, MolPipeline cleaned":
        "final_models_morgan_none_cleaned.txt",

    "MACCS, all compounds":
        "final_models_maccs_none.txt",    
    "MACCS, physchem samples":
        "final_models_maccs_none_physchem_samples.txt",
    "MACCS, MolPipeline cleaned":
        "final_models_maccs_none_cleaned.txt",


    "Embeddings, all compounds":
        "final_models_embeddings_none.txt",
    "Embeddings, physchem samples":
        "final_models_embeddings_none_physchem_samples.txt",
    "Embeddings, MolPipeline cleaned":
        "final_models_embeddings_none_cleaned.txt",
    
    "Physchem, imputed":
        "final_models_physchem_none_physchem_imputed.txt",
    "Physchem, descriptors removed":
        "final_models_physchem_none_removed_problematic_descriptors.txt",
    "Physchem, physchem samples":
        "final_models_physchem_none.txt",
    "Physchem, MolPipeline cleaned":
        "final_models_physchem_none_cleaned.txt",
}


groups = {
    "Morgan": [
        "Morgan, all compounds",
        "Morgan, physchem samples",
        "Morgan, MolPipeline cleaned",
    ],

    "MACCS": [
        "MACCS, all compounds",
        "MACCS, physchem samples",
        "MACCS, MolPipeline cleaned",
    ],

    "Embeddings": [
        "Embeddings, all compounds",
        "Embeddings, physchem samples",
        "Embeddings, MolPipeline cleaned",
    ],

    "Physchem": [
        "Physchem, imputed",
        "Physchem, descriptors removed",
        "Physchem, physchem samples",
        "Physchem, MolPipeline cleaned",
    ]
}


# ============================================================
# Fisher averaging across folds
# ============================================================

assay_setting_results = {}

import os

def get_base_dir(assay):
    candidate_dirs = [
        f"../model_outputs/steroidal/{assay}",
        f"../model_outputs/androgens/{assay}",
        f"../model_outputs/estrogens/{assay}",
        f"../model_outputs/progestagens/{assay}",
        f"../model_outputs/glucocorticoids/{assay}",
    ]

    for path in candidate_dirs:
        if os.path.isdir(path):
            return path

    return None

for assay in assays:

    base_dir = get_base_dir(assay)
    assay_setting_results[assay] = {}

    for setting_name, filename in settings.items():

        mccs = []

        for fold in range(5):

            filepath = os.path.join(
                base_dir,
                f"fold{fold}",
                filename
            )

            
            try:
                df = pd.read_csv(
                filepath,
                sep="\t",
                header=None
                )
            except:
                df = pd.read_csv(
                filepath,
                sep="\t",
                header=None,
                skiprows = 1
                )
            df.dropna(inplace = True, how = 'all')
            df.drop_duplicates(inplace = True)
            rf_row = df[df[0] == "rf"]
            if len(rf_row) == 0:
                print(f"No rf row in {filepath}")
                continue
            if len(rf_row) > 1:
                rf_row.drop_duplicates(inplace = True, keep = 'first')
            mcc = float(rf_row.iloc[0, 2])

            # avoid atanh overflow
            mcc = np.clip(mcc, -0.999999, 0.999999)

            mccs.append(mcc)

        if len(mccs) == 0:
            assay_setting_results[assay][setting_name] = np.nan
            continue

        z = np.arctanh(mccs)
        mean_z = np.mean(z)

        fisher_mean_mcc = np.tanh(mean_z)

        assay_setting_results[assay][setting_name] = fisher_mean_mcc


# ============================================================
# Significance annotation helper
# ============================================================

def add_significance_bar(ax, x1, x2, y, h, text="*"):

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=1.3,
        c="black"
    )

    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=10
    )


# ============================================================
# One plot per group
# ============================================================

rng = np.random.default_rng(42)

for group_name, group_settings in groups.items():

    # --------------------------------------------------------
    # Build assay x setting matrix
    # --------------------------------------------------------

    rows = []

    for assay in assays:

        values = []

        for setting in group_settings:
            values.append(
                assay_setting_results[assay].get(
                    setting,
                    np.nan
                )
            )

        if np.any(np.isnan(values)):
            print(values)
            continue

        rows.append(values)

    X = np.array(rows)

    if X.shape[0] < 2:
        print(f"Skipping {group_name}: insufficient assays")
        print(X)
        continue

    # --------------------------------------------------------
    # Friedman
    # --------------------------------------------------------

    friedman_stat, friedman_p = friedmanchisquare(
        *[X[:, i] for i in range(X.shape[1])]
    )

    n = X.shape[0]
    k = X.shape[1]

    Ff = (
        (n - 1) * friedman_stat
        / (n * (k - 1) - friedman_stat)
    )

    iman_p = 1 - f.cdf(
        Ff,
        k - 1,
        (k - 1) * (n - 1)
    )

    print("\n")
    print("=" * 60)
    print(group_name)
    print(f"Friedman p = {friedman_p:.6g}")
    print(f"Iman-Davenport p = {iman_p:.6g}")

    # --------------------------------------------------------
    # Nemenyi
    # --------------------------------------------------------

    nemenyi_pvals = None

    if iman_p < 0.05:

        nemenyi_pvals = sp.posthoc_nemenyi_friedman(X)

        nemenyi_pvals.columns = group_settings
        nemenyi_pvals.index = group_settings

        print("\nPairwise Nemenyi:")
        print(nemenyi_pvals)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    data = [X[:, i] for i in range(k)]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=group_settings
    )

    for box in bp["boxes"]:
        box.set(
            facecolor="lightblue",
            alpha=0.7
        )

    # assay points

    for i, values in enumerate(data, start=1):

        x = rng.normal(
            i,
            0.05,
            size=len(values)
        )

        ax.scatter(
            x,
            values,
            color="black",
            s=25,
            alpha=0.8,
            zorder=3
        )

    ax.set_ylabel("Fisher-averaged MCC", fontsize = 14)
    ax.set_title(
        f"{group_name} preprocessing\n comparison across assays", fontsize = 18
    )

    plt.xticks(
        rotation=25,
        ha="right", fontsize = 14
    )
    plt.yticks( fontsize = 14
    )

    # --------------------------------------------------------
    # Significant pairwise annotations
    # --------------------------------------------------------

    ymax = max(np.max(d) for d in data)

    step = 0.04
    bar_height = 0.01

    pairs_to_annotate = []

    if nemenyi_pvals is not None:

        for i in range(k):
            for j in range(i + 1, k):

                pval = nemenyi_pvals.iloc[i, j]

                label = "*" if pval < 0.05 else "n.s."

                pairs_to_annotate.append(
                    (i, j, label)
                )

    else:
        # No significant Friedman/Iman-Davenport result:
        # label every pair as non-significant

        for i in range(k):
            for j in range(i + 1, k):

                pairs_to_annotate.append(
                    (i, j, "n.s.")
                )

    for n_bar, (i, j, label) in enumerate(pairs_to_annotate):

        y = ymax + 0.04 + n_bar * step

        add_significance_bar(
            ax,
            i + 1,
            j + 1,
            y,
            bar_height,
            label
        )

    ax.set_ylim(-0.1, 1)

    plt.tight_layout()

    plt.savefig(
        f"../plotting_results/{group_name}_preprocessing_setting_comparison_friedman_nemenyi.png",
        dpi=600
    )

    plt.close()

