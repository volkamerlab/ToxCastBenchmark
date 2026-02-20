import itertools
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from sklearn.metrics import roc_auc_score
custom_palette = {
    "rf": "#c82254",
    "mlp": "#d7df23",
    "cat_boost": "#004877",
    "svm": "#000000",
    "tabpfn": "#6e6e6e"
}
model_order = ["rf", "cat_boost", "tabpfn", 'mlp', 'svm']


def pairwise_tests(data):
    results = []
    models = data["model"].unique()
    for m1, m2 in itertools.combinations(models, 2):
        d1 = data.loc[data["model"] == m1, "auroc"].values
        d2 = data.loc[data["model"] == m2, "auroc"].values

        stat, pval = wilcoxon(d1, d2)
        results.append((m1, m2, pval))
    pvals = [p for (_, _, p) in results]
    reject, pvals_corrected, _, _ = multipletests(
        pvals, method="fdr_bh")
    results_corrected = [(m1, m2, pvals_corrected[i])
                         for i, (m1, m2, _) in enumerate(results)]
    return results_corrected


def facet_plot(data, **kwargs):
    ax = plt.gca()

    # ---- custom spacing ----
    gap = 0.5           # distance between categories (<1 makes them closer)
    width = 0.4         # box width (independent of gap!)
    positions = [i * gap for i in range(len(model_order))]

    medians = {}
    # Draw each model’s box manually
    for pos, m in zip(positions, model_order):
        y = data.loc[data["model"] == m, "auroc"].dropna().values
        if y.size == 0:
            continue

        bp = ax.boxplot(
            y,
            positions=[pos],
            widths=width,
            whis=(5, 95),
            patch_artist=True,
            manage_ticks=False
        )

        # Style
        for box in bp["boxes"]:
            box.set(facecolor=custom_palette[m], alpha=0.4, edgecolor="black")
        for med in bp["medians"]:
            med.set(color="black", linewidth=2)
        for wh in bp["whiskers"]:
            wh.set(color="black")
        for cap in bp["caps"]:
            cap.set(color="black")

        # Save median
        medians[m] = y.mean() if y.size == 0 else np.mean(y)

        # Overlay fold points
        ax.scatter(
            [pos] * len(y), y,
            s=50, color=custom_palette[m], edgecolors="black", linewidths=1,
            zorder=3
        )

    # Connect folds across models
    for fold, sub in data.groupby("fold"):
        xs, ys = [], []
        for pos, m in zip(positions, model_order):
            row = sub[sub["model"] == m]
            if row.empty:
                continue
            xs.append(pos)
            ys.append(row["auroc"].iloc[0])
        if len(xs) >= 2:
            ax.plot(xs, ys, color="lightgrey", alpha=0.6, zorder=0)

    # Reference line
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1)

    # ---- Highlight best median ----
    if medians:
        best_model = max(medians, key=medians.get)
        best_pos = positions[model_order.index(best_model)]
        best_median = medians[best_model]

        # Rectangle around the box
        rect = patches.Rectangle(
            # x,y lower-left
            (best_pos - width/2 - 0.05, min(data["auroc"]) - 0.02),
            width + 0.1, max(data["auroc"]) -
            min(data["auroc"]) + 0.04,  # width,height
            linewidth=2, edgecolor="black", facecolor="none", zorder=4
        )
        ax.add_patch(rect)

    ymax = data["auroc"].max()
    h = 0.05  # height of significance bars
    results = pairwise_tests(data)

    for i, (m1, m2, pval) in enumerate(results):
        d1 = data[data["model"] == m1]["auroc"].values
        d2 = data[data["model"] == m2]["auroc"].values

        # x positions from model_order
        x1, x2 = positions[model_order.index(
            m1)], positions[model_order.index(m2)]
        y = ymax + h*i

        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="black")
        if pval < 0.05:
            ax.text((x1+x2)/2, y+h, f"p={pval:.3f}",
                    ha="center", va="bottom", c='red', fontsize=14)
        else:
            ax.text((x1+x2)/2, y+h, f"p={pval:.3f}",
                    ha="center", va="bottom", c='black', fontsize=14)

    # Cosmetic cleanup
    ax.set_xticks([])
    ax.set_xlabel("")  # will set assay outside
    ax.set_xlim(positions[0] - width/2-0.1, positions[-1] + width/2+0.1)
    ax.tick_params(axis="y", labelsize=14)
    # ax.set_ylim(-0.5,1)
    return ax

def across_assay_test(final_results, dr_method, model_name_map):
    # Compute the mean auroc per assay and model (averaging over folds)
    assay_means = (
        final_results.groupby(["assay", "model"])["auroc"]
        .mean()
        .reset_index()
    )

    # Pivot to wide format: each column = model, each row = assay
    assay_matrix = assay_means.pivot(index="assay", columns="model", values="auroc")

    # Perform pairwise Wilcoxon signed-rank tests across assays
    model_names = assay_matrix.columns.tolist()
    results_global = []
    for m1, m2 in itertools.combinations(model_names, 2):
        # Drop NaNs (some assays might be missing)
        valid = assay_matrix[[m1, m2]].dropna()
        if valid.empty:
            continue
        stat, pval = wilcoxon(valid[m1], valid[m2])
        results_global.append((m1, m2, pval))

    # Correct for multiple comparisons (FDR)
    pvals = [p for (_, _, p) in results_global]
    reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

    results_global_corrected = []
    for i, (m1, m2, p) in enumerate(results_global):
        results_global_corrected.append({
            "Model 1": m1,
            "Model 2": m2,
            "p_uncorrected": p,
            "p_corrected": pvals_corr[i],
            "Significant (FDR<0.05)": reject[i]
        })

    global_results_df = pd.DataFrame(results_global_corrected)

    summary_df = assay_means.copy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    gap = 0.5
    width = 0.4
    positions = [i * gap for i in range(len(model_order))]
    medians = {}

    for pos, m in zip(positions, model_order):
        y = summary_df.loc[summary_df["model"] == m, "auroc"].dropna().values
        if y.size == 0:
            continue

        bp = ax.boxplot(
            y,
            positions=[pos],
            widths=width,
            whis=(5, 95),
            patch_artist=True,
            manage_ticks=False
        )

        # Style
        for box in bp["boxes"]:
            box.set(facecolor=custom_palette[m], alpha=0.4, edgecolor="black")
        for med in bp["medians"]:
            med.set(color="black", linewidth=2)
        for wh in bp["whiskers"]:
            wh.set(color="black")
        for cap in bp["caps"]:
            cap.set(color="black")

        ax.scatter(
            [pos] * len(y),
            y,
            s=70, color=custom_palette[m],
            edgecolors="black", linewidths=1, zorder=3
        )

        medians[m] = np.mean(y)

    # Highlight best mean
    if medians:
        best_model = max(medians, key=medians.get)
        best_pos = positions[model_order.index(best_model)]
        rect = patches.Rectangle(
            (best_pos - width/2 - 0.05, min(summary_df["auroc"]) - 0.02),
            width + 0.1,
            max(summary_df["auroc"]) - min(summary_df["auroc"]) + 0.04,
            linewidth=2, edgecolor="black", facecolor="none", zorder=4
        )
        ax.add_patch(rect)

    # Add global significance bars
    ymax = summary_df["auroc"].max()
    h = 0.05
    for i, (m1, m2, pval) in enumerate(zip(global_results_df["Model 1"],
                                           global_results_df["Model 2"],
                                           global_results_df["p_corrected"])):
        x1, x2 = positions[model_order.index(m1)], positions[model_order.index(m2)]
        y = ymax + h * i
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        color = 'red' if pval < 0.05 else 'black'
        ax.text((x1 + x2) / 2, y + h, f"p={pval:.3f}", ha="center", va="bottom", c=color, fontsize=12)

    ax.set_xticks(positions)
    ax.set_xticklabels([model_name_map[m]for m in model_order], fontsize=14, rotation = 45)
    ax.set_ylabel("Mean AUROC per assay", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(
        f"/home/lisa-marie-rolli//comptox_benchmark/ToxCast_Assays_Endpoint_Results/{dr_method}_global_summary_physchem.png",
        dpi=300
    )
    plt.close(fig)


def main():

    final_results = None
    dr_method = 'mrmr'
    for subfolder in ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']:

        directory = f'/home/lisa-marie-rolli//comptox_benchmark/ToxCast_Assays_Endpoint_Results/{subfolder}/'

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
                        f'{directory}/{content}/fold{fold}/final_models_physchem_{dr_method}.txt', sep='\t', skiprows=1, names=['model', 'fold', 'mcc', 'auroc'])
                except:
                    assay_done = False
                    break
                if not len(new_df) == 5:
                    assay_done = False
                    break
                
                if results_df is None:
                    results_df = new_df
                else:
                    results_df = pd.concat([results_df, new_df], axis=0)
            if not assay_done:
                continue
            else:
                results_df.reset_index(inplace=True, drop=True)

                results_df['assay'] = [content for _ in range(len(results_df))]
                if final_results is None:
                    final_results = results_df
                else:
                    final_results = pd.concat(
                        [final_results, results_df.copy(deep=True)])
                    final_results.reset_index(inplace=True, drop=True)
            
            # Fix the order of models on the x-axis (so folds line up correctly)

        model_name_map = {
            "rf": "Random Forest",
            "mlp": "Multi-layer Perceptron",
            "svm": "Support Vector Machine",
            "cat_boost": "CatBoost",
            "tabpfn": "TabPFN"
        }
    
    g = sns.FacetGrid(final_results, col="assay",
                      col_wrap=4, sharey=True, height=5)
    g.map_dataframe(facet_plot)

    # Global y-axis label
    g.set_ylabels("AUROC", fontsize=16)
    # g.set_ylabels("ROC-AUC", fontsize=16)

    g.set_titles("{col_name}", size=14)
    # Global title
    g.figure.subplots_adjust(top=0.8)
    g.figure.suptitle(
        f"Model performance on physicochemical properties with {dr_method.upper()}", fontsize=18, y=1)

    # One global legend
    legend_elements = [
        Patch(facecolor=custom_palette[m],
              edgecolor="black", label=model_name_map[m])
        for m in model_order
    ]


    for ax in g.axes.flat:
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 - (0.07 / (math.ceil(len(final_results['assay'].unique()) / 4))), box.width, box.height + (0.05 / (math.ceil(len(final_results['assay'].unique()) / 4)))])
    g.figure.legend(handles=legend_elements, title="Model", loc="lower center",
                    bbox_to_anchor=(0.5, 0), ncol=len(legend_elements), fontsize=13, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(
        f'/home/lisa-marie-rolli//comptox_benchmark/ToxCast_Assays_Endpoint_Results//{dr_method}_physchem_auroc.png')
    return
    across_assay_test(final_results, dr_method, model_name_map)


if __name__ == '__main__':
    main()
