import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pandas as pd
import seaborn as sns

def boxplot_best_performances(df_avg, outdir = '../plotting_results', metric='mcc_avg', ylab="Best MCC per assay (Fisher-averaged)"):
    best_per_assay = (
        df_avg.loc[
            df_avg.groupby("assay")[metric].idxmax()
        ]
    )
    # best_values = best_per_assay[metric]

    plt.figure(figsize=(6, 6))

    ax = sns.boxplot(data=best_per_assay,
                     y=metric,
                     color='lightgrey',
                     width=0.4
                     )
    print(best_per_assay)
    # overlay points (important!)
    sns.stripplot(
        data=best_per_assay,
        y=metric,
        color = 'red',
        size=6,
        alpha=0.7
    )
    plt.ylim(-0.1,0.6)
    plt.ylabel(ylab, fontsize=13)
    plt.title("Baseline performance across assays", fontsize=15)

    plt.xticks([])  # no x-axis needed

    plt.tight_layout()
    plt.savefig(
        f'{outdir}/boxplot_with_baseline_performance.pdf', dpi=600)


def compare_mcc_wilcoxon(
    baseline_df: pd.DataFrame,
    best_df: pd.DataFrame,
    assay_col: str = "assay",
    mcc_col: str = "mcc_avg",
    outdir = '../plotting_results/',
    figsize=(6, 5),
):
    """
    Compare paired MCC values using a two-sided Wilcoxon signed-rank test.

    Pairing is defined by the assay column.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline results.
    best_df : pd.DataFrame
        Results with best-performing models.
    assay_col : str
        Column defining assay pairs.
    mcc_col : str
        MCC column to compare.
    figsize : tuple
        Figure size.

    Returns
    -------
    statistic : float
        Wilcoxon statistic.
    p_value : float
        Two-sided p-value.
    merged_df : pd.DataFrame
        DataFrame containing the paired observations.
    """

    # Keep only required columns and rename MCC columns
    baseline = baseline_df[[assay_col, mcc_col]].rename(
        columns={mcc_col: "baseline_mcc"}
    )

    best = best_df[[assay_col, mcc_col]].rename(
        columns={mcc_col: "best_mcc"}
    )

    # Pair assays
    merged_df = baseline.merge(best, on=assay_col, how="inner")

    # Remove missing values
    merged_df = merged_df.dropna(subset=["baseline_mcc", "best_mcc"])

    x = merged_df["baseline_mcc"].to_numpy()
    y = merged_df["best_mcc"].to_numpy()

    if len(x) == 0:
        raise ValueError("No paired assays found after merging.")

    # Two-sided Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(
        x,
        y,
        alternative="two-sided",
        zero_method="wilcox",
    )

    # Visualization
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        [x, y],
        labels=["Baseline", "Best Performance"],
        patch_artist=True,
        widths=0.5,
    )

    # Light grey boxes
    for box in bp["boxes"]:
        box.set(facecolor="lightgrey", edgecolor="black")

    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    rng = np.random.default_rng(42)

    # Scatter points in red
    for xpos, values in zip([1, 2], [x, y]):
        jitter = rng.normal(0, 0.04, len(values))
        ax.scatter(
            xpos + jitter,
            values,
            color="red",
            alpha=0.7,
            s=30,
            zorder=3,
        )

    ax.set_ylabel("MCC")
    ax.set_title(
        f"Paired Wilcoxon signed-rank test\n"
        f"n = {len(merged_df)}, p = {p_value:.3g}"
    )

    plt.tight_layout()

    
    
    plt.savefig(f'{outdir}/baseline_versus_best_wilcoxon.png', dpi=600)


def main():
    df_baseline = pd.read_csv('assays_baseline_mcc.csv', sep='\t')
    df_best_performances = pd.read_csv('assays_with_best_performance.csv', sep = '\t')
    compare_mcc_wilcoxon(baseline_df = df_baseline,best_df= df_best_performances)
    boxplot_best_performances(df_baseline)

if __name__ == '__main__':
    main()
