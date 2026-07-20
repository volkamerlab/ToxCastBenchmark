import pandas as pd
import os
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.spatial.distance import rogerstanimoto
import warnings
warnings.filterwarnings(action='ignore')

title_font_size = 16
label_font_size = 14


def sim_to_nearest_neighbour(train_df, test_df):
    X_train = train_df.to_numpy(dtype=np.uint8)
    X_test = test_df.to_numpy(dtype=np.uint8)

    # Number of bits set in each training fingerprint
    train_counts = X_train.sum(axis=1)

    nearest_similarity = np.empty(X_test.shape[0])

    for i, test_fp in enumerate(X_test):
        # Intersection counts
        intersection = X_train @ test_fp

        # Union counts
        union = train_counts + test_fp.sum() - intersection

        # Tanimoto similarities
        similarity = intersection / union

        nearest_similarity[i] = similarity.max()
    return np.mean(nearest_similarity)


def main():
    assays_with_best_performances = pd.read_csv('assays_with_best_performance.csv', sep = '\t')
    print(assays_with_best_performances)
    morgan_fingerprints = pd.read_csv('../model_inputs/morgan.csv', sep = '\t', index_col = 0)
    assays_with_best_performances['average nn similarity'] = [None for _ in range(len(assays_with_best_performances))]
    for assay_name in assays_with_best_performances["assay"].unique():
        
        subfolders = ['steroidal', 'androgens', 'estrogens', 'glucocorticoids', 'progestagens']
        
        for subfolder in subfolders:
            input_directory = f'../model_inputs/{subfolder}/'
            content = os.listdir(input_directory)
            if assay_name in content:
                sim = []
                for fold in range(5):
                    with open(f'../model_inputs/{subfolder}/{assay_name}/fold{fold}/train.txt', 'r') as train_file:
                        train_compounds = train_file.read().splitlines()
                    with open(f'../model_inputs/{subfolder}/{assay_name}/fold{fold}/test.txt', 'r') as test_file:
                        test_compounds = test_file.read().splitlines()
                    X = morgan_fingerprints.loc[np.concatenate([train_compounds, test_compounds]), :]
                    
                    
                    similarity = sim_to_nearest_neighbour(train_df=X.loc[train_compounds, :], test_df=X.loc[test_compounds,:])
                    sim.append(similarity)
                average_sims = np.average(sim)

                print(f'{assay_name}: {average_sims}')
                assays_with_best_performances.loc[assays_with_best_performances['assay'] == assay_name, 'average nn similarity'] = average_sims
    assays_with_best_performances.to_csv('assays_average_nn_distance.csv', sep = '\t', index = False)

    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=assays_with_best_performances,
        x="average nn similarity",
        y="mcc_avg",
        s=70,
        color="red",
        edgecolor="black"
    )
    
    # regression line
    sns.regplot(
        data=assays_with_best_performances,
        x="average nn similarity",
        ci = None,
        y="mcc_avg",
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2}
    )
    
    # correlation
    corr = assays_with_best_performances["average nn similarity"].corr(assays_with_best_performances["mcc_avg"])
    plt.text(
        0.05, 0.95,
        f"r = {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=label_font_size,
        verticalalignment="top"
    )
    
    plt.xlabel("Average Similarity to Closest Train Fold Compound", fontsize=title_font_size)
    plt.ylabel("Best MCC (Fisher-averaged)", fontsize=label_font_size)
    plt.title("Best performance vs average similarity", fontsize=label_font_size)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../plotting_results/performance_vs_train_similarity.png')
    
    
if __name__=='__main__':
    main()