import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def boxplot_best_performances(best_values, outdir, ylab="Baseline MCC per assay (Fisher-averaged)"):
    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(
        data=best_values, y='mcc_avg',
        color="lightgray",
        width=0.4
    )

    sns.stripplot(
        data=best_values, y='mcc_avg',
        color="red",
        size=6,
        alpha=0.7
    )
    print(np.median(best_values['mcc_avg']))
    plt.ylabel(ylab, fontsize=13)
    plt.title("Baseline performance across assays", fontsize=15)
    plt.ylim((-0.1, 0.6))
    plt.xticks([])  # no x-axis needed

    plt.tight_layout()
    plt.savefig(f'{outdir}/boxplot_with_baseline_performances.png', dpi=600)


def main():
    df_baseline = pd.read_csv('assays_baseline_mcc.csv', sep='\t')
    boxplot_best_performances(df_baseline, outdir='../plotting_results/')


if __name__ == '__main__':
    main()
