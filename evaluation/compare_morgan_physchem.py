import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


with open('../model_inputs/sorted_assays.txt', 'r') as sorted_assay_file:
    assays = sorted_assay_file.read().splitlines()
# Base directory
for assay in assays:
    base_dir = f"../model_outputs/steroidal/{assay}/"

    # Settings files
    settings = {
        "Morgan + None":
            "final_models_morgan_none.txt",
        "Morgan + None + cleaned":
            'final_models_morgan_none_cleaned.txt',
        "Morgan + None + Physchem Samples":
            "final_models_morgan_none_physchem_samples.txt",
        "MACCS + None":
            "final_models_maccs_none.txt",
        "MACCS + None + cleaned":
            'final_models_maccs_none_cleaned.txt',
        "MACCS + None + Physchem Samples":
            "final_models_maccs_none_physchem_samples.txt",
        "Embeddings + None":
            "final_models_embeddings_none.txt",
        "Embeddings + None + cleaned":
            'final_models_embeddings_none_cleaned.txt',
        "Embeddings + None + Physchem Samples":
            "final_models_embeddings_none_physchem_samples.txt",
        "Physchem + None":
            "final_models_physchem_none.txt",
        "Physchem + None + cleaned":
            'final_models_physchem_none_cleaned.txt',
        "Physchem Imputed":
            "final_models_physchem_none_physchem_imputed.txt",
        "Physchem Problematic Removed":
            "final_models_physchem_none_removed_problematic_descriptors.txt",
        
    }

    # Collect MCC values
    results = {name: [] for name in settings}

    for fold in range(5):
        fold_dir = os.path.join(base_dir, f"fold{fold}")

        for setting_name, filename in settings.items():
            filepath = os.path.join(fold_dir, filename)

            if not os.path.exists(filepath):
                print(f"Warning: missing file {filepath}")
                continue

            df = pd.read_csv(filepath, sep="\t", header=None)

            # Expected columns:
            # 0 = model name
            # 1 = fold name
            # 2 = MCC
            rf_row = df[df[0] == "rf"]

            if len(rf_row) == 0:
                print(f"Warning: no rf entry in {filepath}")
                continue

            mcc = float(rf_row.iloc[0, 2])
            results[setting_name].append(mcc)

    # Prepare data for plotting
    labels = list(results.keys())
    data = [results[label] for label in labels]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels
    )

    # Color boxes
    for box in bp["boxes"]:
        box.set(facecolor="lightblue", alpha=0.7)

    # Overlay individual fold points
    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.05, size=len(values))
        ax.scatter(
            x,
            values,
            color="black",
            s=50,
            alpha=0.8,
            zorder=3
        )

    ax.set_ylabel("MCC")
    ax.set_title("RF MCC across 5 folds for different settings")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(f"../plotting_results/{assay}_morgan_versus_rdkit.png", dpi=600)

