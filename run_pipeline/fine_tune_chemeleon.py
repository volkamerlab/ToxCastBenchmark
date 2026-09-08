import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir / "models"))

from CheMeleon import CheMeleon

def main():
    checkpoint_path = base_dir / "pretrained_models/chemeleon_mp.pt"
    smiles_file = base_dir / "model_inputs/compounds_with_smiles.csv"

    subfolders = ["androgens", "estrogens", "glucocorticoids", "progestagens", "steroidal"]

    smiles = pd.read_csv(smiles_file, sep="\t")
    smiles.set_index("compound", inplace=True)

    all_assays = []

    for subfolder in subfolders:
        directory = base_dir / "model_outputs" / subfolder

        for content in sorted(os.listdir(directory)):
            assay_path = directory / content
            if assay_path.is_dir():
                all_assays.append((subfolder, content))

    for subfolder, content in all_assays:
        print(content, flush=True)
        path_to_CV_folds = base_dir / "model_inputs" / subfolder / content

        response_dir = (
            base_dir
            / "ToxCastDownloads"
            / "binary_responses_and_datasail_input_files"
            / subfolder
        )

        pattern = f"{content}-*_binary_response.csv"
        matching_files = glob.glob(str(response_dir / pattern))

        if not matching_files:
            continue

        response = pd.read_csv(matching_files[0], sep="\t")
        response.set_index("compound", inplace=True)

        assay_data = response[["response"]].join(smiles[["smiles"]], how="left")

        for fold in range(5):
            fold_path = path_to_CV_folds / f"fold{fold}"

            training_samples = fold_path / "train.txt"
            test_samples = fold_path / "test.txt"

            with open(training_samples) as f:
                train_ids = f.read().splitlines()

            with open(test_samples) as f:
                test_ids = f.read().splitlines()

            train_df = assay_data.loc[train_ids]
            test_df = assay_data.loc[test_ids]

            output_dir = (base_dir / "model_outputs" / subfolder / content / f"fold{fold}")
            output_dir.mkdir(parents=True, exist_ok=True)

            prediction_file = (output_dir / "chemeleon_best_combi_test_predictions.csv")
            results_file = output_dir / "final_models_chemeleon.txt"

            if prediction_file.exists() and results_file.exists():
                continue

            model = CheMeleon(checkpoint_path=checkpoint_path, seed=42, device="auto")
            model.fit(train_df["smiles"].tolist(), train_df["response"].to_numpy())

            probabilities = model.predict_proba(test_df["smiles"].tolist())

            predicted = (probabilities[:, 1] >= 0.5).astype(int)
            actual = test_df["response"].to_numpy(dtype=int)

            mcc = matthews_corrcoef(actual, predicted)

            if len(np.unique(actual)) == 2:
                auroc = roc_auc_score(actual, probabilities[:, 1])
            else:
                auroc = np.nan

            res = pd.DataFrame({
                "compound": test_df.index,
                "predicted": predicted,
                "actual": actual,
                "p(0)": probabilities[:, 0],
                "p(1)": probabilities[:, 1],
            })

            res.to_csv(prediction_file, sep="\t", index=False)

            with open(results_file, "w") as output:
                output.write(
                    f"chemeleon\tfold{fold}\t{mcc}\t{auroc}\n"
                )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
