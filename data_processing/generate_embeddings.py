import argparse
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.utils import gen_batches
from transformers import AutoTokenizer
from rdkit import Chem
from utils import adapt_model_with_mtr, compute_targets

os.environ["WANDB_DISABLED"] = "true"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("train", help="train.txt: one compound ID per line")
    p.add_argument("test", help="test.txt: one compound ID per line")
    p.add_argument("smiles", help="tab-separated csv with columns: compound, smiles")
    p.add_argument("output", help="output CSV for embeddings")
    return p.parse_args()


def first_token(outputs):
    token_embeddings = outputs.last_hidden_state
    summary = token_embeddings[:, 0, :]
    return summary


def filter_bad_smiles(df, smiles_col="smiles"):
    # dropping empty/missing SMILES
    s = df[smiles_col].astype(str).str.strip()
    s = s.replace({"nan": "", "None": ""})
    df = df.loc[s != ""].copy()
    df[smiles_col] = s.loc[s != ""]

    # keep only SMILES RDKit can parse
    valid_mask = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"Excluding {n_invalid} compounds from DA (invalid SMILES)")

    df = df[valid_mask].copy()

    # RDKit descriptor check for rows with nan/infinite values
    targets = compute_targets(df[smiles_col].tolist())
    finite_mask = np.isfinite(targets).all(axis=1)
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        print(f"Excluding {n_bad} compounds from DA (nan/inf RDKit descriptors)")

    df = df[finite_mask].copy()

    print(f"Using {len(df)} compounds for domain adaptation")

    return df


def embed_df(adapted, tokenizer, df, smiles_col="smiles", device="cpu"):
    tokens = tokenizer(
        df[smiles_col].tolist(),
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )

    embs = []
    with torch.no_grad():
        for batch in tqdm(gen_batches(len(df), batch_size=256), desc="Computing embeddings"):
            inputs = {k: v[batch].to(device) for k, v in tokens.items()}
            embs.append(adapted.embeddings(**inputs))

    return torch.vstack(embs).cpu().numpy()


def main():
    args = parse_args()

    train_ids = pd.read_csv(args.train, header=None, names=["compound"])
    test_ids = pd.read_csv(args.test, header=None, names=["compound"])
    smiles_df = pd.read_csv(args.smiles, sep="\t")[["compound", "smiles"]].dropna()

    # merge ids with smiles
    train_all = train_ids.merge(smiles_df, on="compound", how="inner")
    test_all = test_ids.merge(smiles_df, on="compound", how="inner")

    # filter for domain adaptation
    train_da = filter_bad_smiles(train_all, smiles_col="smiles")
    if train_da.empty:
        raise ValueError("No valid compounds left for domain adaptation after filtering.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig, adapted, tokens = adapt_model_with_mtr(
        model_name="UdS-LSV/da4mt-mlm-30",
        smiles=train_da["smiles"].tolist(),
        extract_embeddings_fn=first_token,
    )
    adapted = adapted.to(device)

    # tokenizer for embedding everything
    tokenizer = AutoTokenizer.from_pretrained("UdS-LSV/da4mt-mlm-30")

    train_embs = embed_df(adapted, tokenizer, train_all, smiles_col="smiles", device=device)
    test_embs = embed_df(adapted, tokenizer, test_all, smiles_col="smiles", device=device)

    # one combined embeddings file
    all_embs = np.vstack([train_embs, test_embs])
    all_compounds = pd.concat([train_all["compound"], test_all["compound"]], ignore_index=True)

    df_out = pd.DataFrame(all_embs)
    df_out.insert(0, "compound", all_compounds.values)

    df_out.to_csv(args.output, sep="\t", index=False)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()