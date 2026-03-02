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
    "svm": "#000000",
    "tabpfn": "#6e6e6e"
}
model_order = ["rf", "cat_boost", "tabpfn", 'mlp', 'svm']



# Fisher Z-transformation
def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))  # avoid ±1

# Inverse Fisher Z-transformation
def inverse_fisher_z(z):
    return np.tanh(z)



def cd_plot_for_nemenyi(final_results, dr_method, model_name_map, feature_type, feature_name_map, output_dir):
    final_results['mcc_z'] = fisher_z(final_results['mcc'])
    
    means = (
        final_results.groupby(['assay', 'model'], as_index=False)['mcc_z']
        .mean()
    )
    means['mcc'] = means['mcc_z'].apply(inverse_fisher_z)
    wide = means.pivot(index='assay', columns='model', values='mcc')

    # Rank per assay (rank 1 = best; higher mcc_z is better)

    ranks = wide.rank(axis=1, ascending=False, method='average')

    # Average ranks 
    
    avg_ranks = ranks.mean(axis=0)
    models = [model_name_map[m] for m in avg_ranks.index]
    wide = means.pivot(index='assay', columns='model', values='mcc_z')
    avg_mcc = wide.mean(axis=0)
    avg_mcc = avg_mcc.apply(inverse_fisher_z)

    pretty_names = {m: f"{model_name_map [m]}\navg. MCC={avg_mcc[m]:.3f}" for m in avg_ranks.index}
    avg_ranks.index = [pretty_names[m] for m in avg_ranks.index]


    # Nemenyi p-values (pairwise)

    pvals = sp.posthoc_nemenyi_friedman(ranks.values)
    pvals.index = models
    pvals.columns = models

    #generate p-value heat map
    plt.close('all')
    norm = TwoSlopeNorm(vmin=pvals.values.min(), vmax=pvals.values.max(), vcenter = 0.05)


    cmap = LinearSegmentedColormap.from_list("p-values", ["#c82254", '#D3D3D3',"#004877"])
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
    

    plt.title(f"Model comparison across assays on \n{feature_name_map[feature_type]} using {dr_method}")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dr_method}_{feature_type}_nemenyi_pval_heatmap.png', dpi = 300, transparent = False)

    pvals.index = avg_ranks.index
    pvals.columns = avg_ranks.index
    
    # Generate CD diagram
    
    plt.close('all')
    color_palette = {pretty_names[m]: custom_palette[m] for m in pretty_names.keys()}
    color_palette[f'{pretty_names["mlp"]}'] = '#a3a919'
    _ = sp.critical_difference_diagram(
        ranks=avg_ranks,            
        sig_matrix=pvals,  
        alpha=0.05,         
        label_fmt_left="{label}\navg. rank: {rank:.2f}",
        label_fmt_right="{label}\navg. rank: {rank:.2f}",
        color_palette=color_palette
    )
    plt.title(f"Critical difference\nmodel comparison across assays on \n{feature_name_map[feature_type]} using {dr_method}")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dr_method}_{feature_type}_critical_difference_plot_nemenyi.png', dpi = 300, transparent = False)
    


def pairwise_friedman_nemenyi(data):
    # Pivot the data to have one column per model and one row per subject
    pivoted = data.pivot(index='fold', columns='model', values='mcc')
    # Drop rows with missing values (if any)
    pivoted = pivoted.dropna()

    # Run Friedman test with correction by Iman and Davenport (1980)
    friedman_stat, p = friedmanchisquare(*pivoted.values.T)
    N = len(np.unique(data['fold']))
    k = len(np.unique(data['model']))
    iman_davenport_correction = ((N - 1) * friedman_stat) / (N * (k - 1) - friedman_stat)

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


def facet_plot(data, **kwargs):
    ax = plt.gca()

    # ---- custom spacing ----
    gap = 0.5           # distance between categories (<1 makes them closer)
    width = 0.4         # box width (independent of gap!)
    positions = [i * gap for i in range(len(model_order))]

    means = {}
    medians= {}
    # Draw each model’s box manually

    for pos, m in zip(positions, model_order):
        y = data.loc[data["model"] == m, "mcc"].dropna().values
        if y.size == 0:
            continue
        
        z_vals = fisher_z(y)
        z_mean = np.mean(z_vals)
        z_median = np.median(z_vals)
        means[m] = inverse_fisher_z(z_mean)
        medians[m] = inverse_fisher_z(z_median)
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
            med.set(color="black", linewidth=0)
        for wh in bp["whiskers"]:
            wh.set(color="black")
        for cap in bp["caps"]:
            cap.set(color="black")

        # Overlay fold points
        ax.scatter(
            [pos] * len(y), y,
            s=50, color=custom_palette[m], edgecolors="black", linewidths=1,
            zorder=3
        )

        ax.hlines(means[m], pos-0.5*width, pos + 0.5* width, linestyle='--', color='black', linewidth=1.5)
        ax.hlines(medians[m], pos-0.5*width, pos + 0.5* width, linestyle='-', color='black', linewidth=1.5)
    # Connect folds across models
    for fold, sub in data.groupby("fold"):
        xs, ys = [], []
        for pos, m in zip(positions, model_order):
            row = sub[sub["model"] == m]
            if row.empty:
                continue
            xs.append(pos)
            ys.append(row["mcc"].iloc[0])
        if len(xs) >= 2:
            ax.plot(xs, ys, color="lightgrey", alpha=0.6, zorder=0)

    # Reference line
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # ---- Highlight best mean ----
    if means:
        best_model = max(means, key=means.get)
        best_pos = positions[model_order.index(best_model)]

        # Rectangle around the box
        rect = patches.Rectangle(
            # x,y lower-left
            (best_pos - width/2 - 0.05, min(data["mcc"]) - 0.02),
            width + 0.1, max(data["mcc"]) -
            min(data["mcc"]) + 0.04,  # width,height
            linewidth=2, edgecolor="black", facecolor="none", zorder=4
        )
        ax.add_patch(rect)

    ymax = data["mcc"].max()
    h = 0.05  # height of significance bars
    results = pairwise_friedman_nemenyi(data)


    j = 0
    for i, (m1, m2, pval) in enumerate(results):
        d1 = data[data["model"] == m1]["mcc"].values
        d2 = data[data["model"] == m2]["mcc"].values

        # x positions from model_order
        x1, x2 = positions[model_order.index(
            m1)], positions[model_order.index(m2)]
        y = ymax + h*j


        if pval < 0.05:
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="black")
            ax.text((x1+x2)/2, y+h, f"p={pval:.3f}",
                    ha="center", va="bottom", c='black', fontsize=14)
            j += 1
        else:
            pass
    # Cosmetic cleanup
    ax.set_xticks([])
    ax.set_xlabel("")  # will set assay outside
    ax.set_xlim(positions[0] - width/2-0.1, positions[-1] + width/2+0.1)
    ax.tick_params(axis="y", labelsize=14)

    return ax


def create_boxplots_per_assay(final_results, dr_method, feature_type, model_name_map, feature_name_map, output_dir):
    g = sns.FacetGrid(final_results, col="assay",
                      col_wrap=4, sharey=True, height=5)
    g.map_dataframe(facet_plot)

    # Global y-axis label
    g.set_ylabels("MCC", fontsize=16)
    # g.set_ylabels("ROC-AUC", fontsize=16)

    g.set_titles("{col_name}", size=14)
    # Global title
    g.figure.subplots_adjust(top=0.8)
    g.figure.suptitle(
        f"Model performance on {feature_name_map[feature_type]} with {dr_method.upper()}", fontsize=18, y=1)

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
                    bbox_to_anchor=(0.5, 0), ncol=len(legend_elements), fontsize=18, title_fontsize=20)


    mean_line = ax.hlines(0, 0, 0, color='black', linestyle='--', linewidth=1.5)
    median_line = ax.hlines(0,0, 0, color='black', linestyle='-', linewidth=1.5)

    custom_handles = [mean_line, median_line]
    custom_labels = ['Mean', 'Median']

    legend2 = g.figure.legend(custom_handles, custom_labels, loc='upper left', title="Statistics", fontsize=16, title_fontsize=18)
    ax.add_artist(legend2)

    plt.tight_layout()
    plt.savefig(
        f'{output_dir}/{dr_method}_{feature_type}_friedman_nemenyi.png', dpi = 300, transparent = False)




def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of 5 fold CV results comparing models across all assays with one feature-DR combination")
    parser.add_argument("-d", '--directory', help="input_directory", default='/home/lisa-marie-rolli/ToxCastBenchmark/model_outputs/')
    parser.add_argument("--dr_method", help="DR method used", default='MI')
    parser.add_argument("--feature_type", '-f', help="Feature type used", default='physchem')
    parser.add_argument("--output_dir", '-o', help="Output_directory", default='/home/lisa-marie-rolli/ToxCastBenchmark/plotting_results/')
    parser.add_argument("--tabpfn_missing", help="is TabPFN missing", default=False, type=bool)
    return parser.parse_args()



def main(args):

    final_results = None
    dr_method = args.dr_method
    feature_type = args.feature_type
    out_dir = args.output_dir
    num_models = 5 if not bool(args.tabpfn_missing) else 4
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
                if not len(new_df) == num_models:
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
        
            
        
        feature_name_map = {
            'physchem': 'physicochemical properties',
            'morgan': 'Morgan fingerprints',
            'maccs': 'MACCS fingerprints',
        }

        model_name_map = {
            "rf": "RF",
            "mlp": "MLP",
            "svm": "SVM",
            "cat_boost": "CatBoost",
            "tabpfn": "TabPFN"
        }

    print(final_results)
    create_boxplots_per_assay(final_results=final_results, dr_method=dr_method, feature_name_map=feature_name_map, feature_type=feature_type, model_name_map=model_name_map, output_dir=out_dir)
    cd_plot_for_nemenyi(final_results, dr_method, model_name_map, feature_type, feature_name_map, output_dir=out_dir)



if __name__ == "__main__":
    args = parse_args()
    main(args)

