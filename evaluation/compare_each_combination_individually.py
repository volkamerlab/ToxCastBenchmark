import os
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt



# -----------------------------
# 2. Best combination per assay
# -----------------------------
def get_best_per_assay(df_avg):
    best = df_avg.loc[
        df_avg.groupby("assay")["mcc_avg"].idxmax()
    ].copy()
    
    return best


def compute_assay_size(df):
    size_dict = {}
    
    for assay_name in df["assay"].unique():
        
        subfolders = ['steroidal', 'androgens', 'estrogens', 'glucocorticoids', 'progestagens']

        for subfolder in subfolders:
            input_directory = f'../model_inputs/{subfolder}/'
            content = os.listdir(input_directory)
            if assay_name in content:

                pattern = f'{assay_name}-*_binary_response.csv'
                matching_files = glob.glob(
                    f'../ToxCastDownloads/binary_responses_and_datasail_input_files/{subfolder}/{pattern}')

        
        if len(matching_files) == 0:
            print(f"[Warning] No file found for {assay_name}")
            continue
        
        file_path = matching_files[0]
        data = pd.read_csv(file_path, sep="\t")
        
        y = data["response"]
        size_dict[assay_name] = len(y)
    
    return size_dict

# -----------------------------
# 3. Compute imbalance per assay
# -----------------------------
def compute_imbalance(df, base_path):
    imbalance_dict = {}
    
    for assay_name in df["assay"].unique():
        
        subfolders = ['steroidal', 'androgens', 'estrogens', 'glucocorticoids', 'progestagens']

        for subfolder in subfolders:
            input_directory = f'../model_inputs/{subfolder}/'
            content = os.listdir(input_directory)
            if assay_name in content:

                pattern = f'{assay_name}-*_binary_response.csv'
                matching_files = glob.glob(
                    f'../ToxCastDownloads/binary_responses_and_datasail_input_files/{subfolder}/{pattern}')

        
        if len(matching_files) == 0:
            print(f"[Warning] No file found for {assay_name}")
            continue
        
        file_path = matching_files[0]
        data = pd.read_csv(file_path, sep="\t")
        
        y = data["response"]
        
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        
        if n_pos == 0 or n_neg == 0:
            imbalance = 0
        else:
            imbalance = n_pos / (n_pos + n_neg)
        
        imbalance_dict[assay_name] = imbalance
    
    return imbalance_dict


# -----------------------------
# 4. Merge performance + imbalance
# -----------------------------
def create_plot_df(best_per_assay, imbalance_dict):
    plot_df = best_per_assay.copy()
    plot_df["imbalance"] = plot_df["assay"].map(imbalance_dict)
    
    # drop missing if any files weren't found
    plot_df = plot_df.dropna(subset=["imbalance"])
    
    return plot_df

def plot_performance_versus_assay_size(plot_df, outdir):
    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=plot_df,
        x="imbalance",
        y="mcc_avg",
        s=70,
        color="red",
        edgecolor="black"
    )
    
    # regression line
    sns.regplot(
        data=plot_df,
        x="imbalance",
        ci = None,
        y="mcc_avg",
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2}
    )
    
    # correlation
    corr = plot_df["imbalance"].corr(plot_df["mcc_avg"])
    plt.text(
        0.05, 0.95,
        f"r = {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top"
    )
    
    plt.xlabel("Number of tested samples", fontsize=13)
    plt.ylabel("Best MCC (Fisher-averaged)", fontsize=13)
    plt.title("Best performance vs assay size", fontsize=15)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{outdir}/performance_vs_assay_size.png')
# -----------------------------
# 5. Plot
# -----------------------------
def plot_performance_vs_imbalance(plot_df, outdir):
    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=plot_df,
        x="imbalance",
        y="mcc_avg",
        s=70,
        color="red",
        edgecolor="black"
    )
    
    # regression line
    sns.regplot(
        data=plot_df,
        x="imbalance",
        ci = None,
        y="mcc_avg",
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2}
    )
    
    # correlation
    corr = plot_df["imbalance"].corr(plot_df["mcc_avg"])
    plt.text(
        0.05, 0.95,
        f"r = {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top"
    )
    
    plt.xlabel("Class balance (#active / all samples)", fontsize=13)
    plt.ylabel("Best MCC (Fisher-averaged)", fontsize=13)
    plt.title("Best performance vs class imbalance", fontsize=15)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{outdir}/performance_vs_imbalance.png')



def boxplot_best_performances(df_avg, outdir):
    best_per_assay = (
    df_avg.loc[
        df_avg.groupby("assay")["mcc_avg"].idxmax()
    ]
    )
    best_values = best_per_assay["mcc_avg"]
    print(best_per_assay.loc[:, ['mcc_avg', 'assay']])
    #print(best_per_assay['mcc_avg'].max())
    plt.figure(figsize=(6, 6))

    ax = sns.boxplot(
        y=best_values,
        color="lightgray",
        width=0.4
    )

    # overlay points (important!)
    sns.stripplot(
        y=best_values,
        color="red",
        size=6,
        alpha=0.7
    )

    plt.ylabel("Best MCC per assay (Fisher-averaged)", fontsize=13)
    plt.title("Best achievable performance across assays", fontsize=15)

    plt.xticks([])  # no x-axis needed

    plt.tight_layout()
    plt.savefig(f'{outdir}/boxplot_with_best_performances.png', dpi = 600)
    best_per_assay = best_per_assay.sort_values("mcc_avg")


def count_top_k(k, df_avg):
    

    
    
    return (
        df_avg[df_avg["rank"] <= k]
        .groupby(["model", "dr_method", "representation"])
        .size()
        .rename(f"top{k}")
    )

def plot_heatmap(df, outpath):
    
    
    models = df["model"].unique()
    dr_methods = df["dr_method"].unique()
    representations = df["representation"].unique()

    full_combinations = pd.MultiIndex.from_product(
        [models, dr_methods, representations],
        names=["model", "dr_method", "representation"]
    )

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
        'tabpfn': 'TabPFN'

    }

    dr_name_map = {
        "pca": "PCA",
        "mrmr": "MRMR",
        "variance": "Highest variance",
        "MI": "Mutual Information",
        "none": "No DR"
    }
    # ---- 1. Fisher transform ----
    # clip to avoid inf
    eps = 1e-6
    df["mcc_clipped"] = df["mcc"].clip(-1 + eps, 1 - eps)
    df["z"] = np.arctanh(df["mcc_clipped"])

    # ---- 2. Average over folds ----
    group_cols = ["model", "dr_method", "representation", "assay"]
    df_avg = (
        df.groupby(group_cols)["z"]
        .mean()
        .reset_index()
    )

    # ---- 3. Inverse transform ----
    df_avg["mcc_avg"] = np.tanh(df_avg["z"])

    # ---- 4. Rank per assay ----
    df_avg["rank"] = df_avg.groupby("assay")["mcc_avg"] \
                        .rank(method="min", ascending=False)

    # ---- 5. Count top-k appearances ----


    top1 = count_top_k(1, df_avg)
    top3 = count_top_k(3, df_avg)
    top5 = count_top_k(5, df_avg)
    top10 = count_top_k(10, df_avg)

    # ---- 6. Combine ----
    result = pd.concat([top1,top3, top5], axis=1).fillna(0)

    # sort by top1
    result = result.sort_values(["top1", "top3", "top5"], ascending=False)
    result = result[result["top1"] >= 2]
    
    
    mapped_index = []
    for m, d, r in result.index:
        mapped_index.append((
            model_name_map.get(m, m),
            dr_name_map.get(d, d),
            feature_name_map.get(r, r)
        ))

    result.index = mapped_index

    
    result.index = [
    f"{r}\n{d}\n{m}"
    for m, d, r in result.index
    ]


    
    # ---- 7. Heatmap ----
    plt.figure(figsize=(14, max(8, len(result) * 0.7)))
    
    ax = sns.heatmap(
        result,
        annot=True,
        fmt=".0f",
        cmap="coolwarm",   # blue -> white -> red (high = red)
        cbar_kws={"label": "# assays"}
    )

    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=10)

    # annotations inside cells
    for text in ax.texts:
        text.set_size(11)

    # colorbar label + ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("# assays", fontsize=12)

    # title and axis labels
    plt.title("Top-k assay wins per representation-DR method-model combination", fontsize=16, pad=30)
    plt.xlabel("")
    plt.ylabel("combination", fontsize=13)

    # improve multiline spacing
    for label in ax.get_yticklabels():
        label.set_linespacing(1.4)
    plt.tight_layout()
    plt.savefig(f'{outpath}/combination_heatmap.svg', dpi = 600)
    
    boxplot_best_performances(df_avg, outpath)
    

    best_per_assay = get_best_per_assay(df_avg)

    imbalance_dict = compute_imbalance(
        df,
        base_path="../ToxCastDownloads/binary_responses_and_datasail_input_files"
    )

    plot_df = create_plot_df(best_per_assay, imbalance_dict)

    plot_performance_vs_imbalance(plot_df, outpath)
    
    
    size_dict = compute_assay_size(df)
    
    plot_df = create_plot_df(best_per_assay, size_dict)

    plot_performance_versus_assay_size(plot_df, outpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation of 5 fold CV results comparing the combinations")
    parser.add_argument("-d", '--directory',
                        help="input_directory", default='../model_outputs/')
    parser.add_argument("--output_dir", '-o',
                    help="Output_directory", default='../plotting_results/')
    return parser.parse_args()

def main(args):

    final_results = None
    out_dir = args.output_dir
    for feature_type in ['maccs', 'morgan', 'embeddings', 'physchem']:
        for dr_method in ['MI', 'mrmr', 'pca', 'variance', 'none']:
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

                        new_df = pd.read_csv(
                            f'{directory}/{content}/fold{fold}/final_models_{feature_type}_{dr_method}.txt', sep='\t', skiprows=1, names=['model', 'fold', 'mcc', 'auroc'])

                        new_df['dr_method'] = [
                            f'{dr_method}' for _ in range(len(new_df.index))]
                        new_df['representation'] = [f'{feature_type}' for _ in range(len(new_df.index))]
                        if results_df is None:
                            results_df = new_df
                        else:

                            results_df = pd.concat([results_df, new_df], axis=0)

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
    print(final_results)

    plot_heatmap(final_results, out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
