import pandas as pd
import os
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.spatial.distance import rogerstanimoto
import warnings
warnings.filterwarnings(action='ignore')

def main():
    #assays_with_best_performances = pd.read_csv('assays_with_best_performance.csv', sep = '\t')
    assays_with_best_performances = pd.read_csv('assays_average_silhouette_score.csv', sep = '\t')
    ''' morgan_fingerprints = pd.read_csv('../model_inputs/morgan.csv', sep = '\t', index_col = 0)
    assays_with_best_performances['average silhouette score'] = [None for _ in range(len(assays_with_best_performances))]
    assays_with_best_performances['average train-test distance'] = [None for _ in range(len(assays_with_best_performances))]
    for assay_name in assays_with_best_performances["assay"].unique():
        
        subfolders = ['steroidal', 'androgens', 'estrogens', 'glucocorticoids', 'progestagens']

        for subfolder in subfolders:
            input_directory = f'../model_inputs/{subfolder}/'
            content = os.listdir(input_directory)
            if assay_name in content:
                silhouette_scores = []
                dist = []
                for fold in range(5):
                    with open(f'../model_inputs/{subfolder}/{assay_name}/fold{fold}/train.txt', 'r') as train_file:
                        train_compounds = train_file.read().splitlines()
                    with open(f'../model_inputs/{subfolder}/{assay_name}/fold{fold}/test.txt', 'r') as test_file:
                        test_compounds = test_file.read().splitlines()
                    X = morgan_fingerprints.loc[np.concatenate([train_compounds, test_compounds]), :]
                    labels = ['train' for _ in range(len(train_compounds))]
                    labels = np.concatenate([labels, ['test' for _ in range(len(test_compounds))]])
                    score = silhouette_score(X = X, labels=labels, metric='jaccard', sample_size=None, random_state=42)
                    distances = pairwise_distances(X = morgan_fingerprints.loc[train_compounds, :].values, Y =morgan_fingerprints.loc[test_compounds, :].values, metric='jaccard', n_jobs=-1 )
                    d = np.average(distances)
                    dist.append(d)
                    silhouette_scores.append(score)
                average_silhouette_score = np.average(silhouette_scores)
                average_distance = np.average(dist)
                print(f'{assay_name}: {average_silhouette_score}, {average_distance}')
                assays_with_best_performances.loc[assays_with_best_performances['assay'] == assay_name, 'average silhouette score'] = average_silhouette_score
                assays_with_best_performances.loc[assays_with_best_performances['assay'] == assay_name, 'average train-test distance'] = average_distance
    assays_with_best_performances.to_csv('assays_average_silhouette_score.csv', sep = '\t', index = False)
    '''
    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=assays_with_best_performances,
        x="average silhouette score",
        y="mcc_avg",
        s=70,
        color="red",
        edgecolor="black"
    )
    
    # regression line
    sns.regplot(
        data=assays_with_best_performances,
        x="average silhouette score",
        ci = None,
        y="mcc_avg",
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2}
    )
    
    # correlation
    corr = assays_with_best_performances["average silhouette score"].corr(assays_with_best_performances["mcc_avg"])
    plt.text(
        0.05, 0.95,
        f"r = {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top"
    )
    
    plt.xlabel("Average silhouette score across CV folds", fontsize=13)
    plt.ylabel("Best MCC (Fisher-averaged)", fontsize=13)
    plt.title("Best performance vs average silhouette score", fontsize=15)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../plotting_results/performance_vs_silhouette.png')
    plt.clf()
    
    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=assays_with_best_performances,
        x="average train-test distance",
        y="mcc_avg",
        s=70,
        color="red",
        edgecolor="black"
    )
    
    # regression line
    sns.regplot(
        data=assays_with_best_performances,
        x="average train-test distance",
        ci = None,
        y="mcc_avg",
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2}
    )
    
    # correlation
    corr = assays_with_best_performances["average train-test distance"].corr(assays_with_best_performances["mcc_avg"])
    plt.text(
        0.05, 0.95,
        f"r = {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top"
    )
    
    plt.xlabel("Average train-test distance across CV folds", fontsize=13)
    plt.ylabel("Best MCC (Fisher-averaged)", fontsize=13)
    plt.title("Best performance vs average distance", fontsize=15)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../plotting_results/performance_vs_distance.png')
    
    
if __name__=='__main__':
    main()