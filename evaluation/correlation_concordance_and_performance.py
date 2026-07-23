import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
title_font_size = 16
label_font_size = 14

def plot_performance_vs_concordance(plot_df, outdir, col_name, title):
    plt.figure(figsize=(7, 6))
    
    sns.scatterplot(
        data=plot_df,
        x=col_name,
        y="mcc_avg",
        s=70,
        color="red",
        edgecolor="black"
    )
    
    # regression line
    sns.regplot(
        data=plot_df,
        x=col_name,
        ci = None,
        y="mcc_avg",
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2}
    )
    
    # correlation
    corr = plot_df[col_name].corr(plot_df["mcc_avg"])
    plt.text(
        0.05, 0.95,
        f"r = {corr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=label_font_size,
        verticalalignment="top"
    )
    name_map = {
        '5NN_active_frac_among_active_query': '5NN active fraction',
        'NN_label_concordance_actives': 'NN label concordance actives',
        'frac_of_analog_actives_that_are_cliffy': 'fraction of active compounds in activity cliffs'
    }
    plt.xlabel(name_map[col_name], fontsize=label_font_size)
    plt.ylabel("Best MCC (Fisher-averaged)", fontsize=label_font_size)
    plt.title(f"Best performance vs\n {title}", fontsize=title_font_size)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{outdir}/performance_vs_{col_name}.png')
    
def main():
    nn_dataframe = pd.read_csv('nn_dataframe.csv', sep ='\t')
    mmp_dataframe = pd.read_csv('mmp_dataframe.csv', sep ='\t')
    performances = pd.read_csv('assays_with_best_performance.csv', sep = '\t')
    performances = performances.loc[:, ['assay', 'mcc_avg']]
    merged = performances.merge(nn_dataframe, on='assay', how='inner')
    plot_performance_vs_concordance(merged, '../plotting_results/', col_name = '5NN_active_frac_among_active_query', title ='fraction of active compounds among\n 5 nearest neighbors')
    plot_performance_vs_concordance(merged, '../plotting_results/', col_name = 'NN_label_concordance_actives',title =  'label concordance for nearest neighbor')
    merged = performances.merge(mmp_dataframe, on='assay', how='inner')
    plot_performance_vs_concordance(merged, '../plotting_results/', col_name = 'frac_of_analog_actives_that_are_cliffy', title ='fraction of active compounds\n that are part of an activity cliff')

if __name__ == '__main__':
    main()