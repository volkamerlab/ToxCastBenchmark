from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

class CheMeleon:
    """
    Binary classification using pretrained CheMeleon.
    """
    def __init__(self, checkpoint_path, seed=42, device="auto"):
        self.checkpoint_path = Path(checkpoint_path)
        self.seed = seed
        self.device = device

        self.max_epochs = 50
        self.batch_size = 64
        self.validation_fraction = 0.2
        self.patience = 5
        self.ffn_hidden_dim = 256

        self.model = None
        self.trainer = None
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def _make_dataset(self, smiles, labels=None):
        smiles = list(smiles)

        if labels is None:
            labels = np.zeros(len(smiles), dtype=float)

        datapoints = [
            data.MoleculeDatapoint.from_smi(
                smi,
                np.array([y], dtype=float),
            )
            for smi, y in zip(smiles, labels)
        ]

        return data.MoleculeDataset(datapoints, self.featurizer)

    def _build_model(self):
        """
        Load a fresh pretrained encoder and attach a classification head.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)

        mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
        mp.load_state_dict(checkpoint["state_dict"])

        agg = nn.MeanAggregation()

        ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim, hidden_dim=self.ffn_hidden_dim)

        model = models.MPNN(mp, agg, ffn)

        return model

    def fit(self, train_smiles, train_labels):
        """
        Fine-tune on the training pool.
        """
        pl.seed_everything(self.seed, workers=True)

        train_smiles = np.asarray(train_smiles)
        train_labels = np.asarray(train_labels, dtype=float)

        train_idx, val_idx = train_test_split(
            np.arange(len(train_smiles)),
            test_size=self.validation_fraction,
            random_state=self.seed,
            stratify=train_labels,
        )

        train_dset = self._make_dataset(train_smiles[train_idx], train_labels[train_idx])
        val_dset = self._make_dataset(train_smiles[val_idx], train_labels[val_idx])

        train_loader = data.build_dataloader(
            train_dset,
            batch_size=self.batch_size,
            num_workers=0,
        )

        val_loader = data.build_dataloader(
            val_dset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        self.model = self._build_model()

        with TemporaryDirectory() as checkpoint_dir:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.patience,
            )

            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )

            self.trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.device,
                devices=1,
                logger=False,
                enable_checkpointing=True,
                check_val_every_n_epoch=1,
                callbacks=[early_stopping, checkpoint_callback],
            )

            self.trainer.fit(self.model, train_loader, val_loader)

            best_path = checkpoint_callback.best_model_path

            if not best_path:
                raise RuntimeError("No best-model checkpoint was saved.")

            self.model = models.MPNN.load_from_checkpoint(best_path, map_location="cpu")

        self.trainer = pl.Trainer(
            accelerator=self.device,
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        return self

    def predict_proba(self, smiles):
        """
        Return an array with columns p(0) and p(1).
        """
        if self.model is None or self.trainer is None:
            raise RuntimeError("The model must be fitted before prediction.")

        dataset = self._make_dataset(smiles)
        loader = data.build_dataloader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        probabilities = self.trainer.predict(self.model, loader)
        p1 = torch.cat(probabilities, dim=0).cpu().numpy().reshape(-1)

        return np.column_stack([1.0 - p1, p1])

    def predict(self, smiles):
        probabilities = self.predict_proba(smiles)
        return (probabilities[:, 1] >= 0.5).astype(int)
