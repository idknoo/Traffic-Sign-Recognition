#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import datetime

import git
import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def dvc_pull():
    """Подтягивает данные через DVC из remote, если доступны."""
    try:
        subprocess.check_call(["dvc", "pull"], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        pass


@hydra.main(version_base=None, config_path="../configs/train", config_name="train")
def main(cfg: DictConfig):
    dvc_pull()

    from scripts.dvc_utils import download_data

    download_data()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    run_name = f"train_{datetime.now():%Y%m%d_%H%M%S}"
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("batch_size", cfg.train.batch_size)
    mlflow.log_param("lr", cfg.train.lr)
    mlflow.log_param("weight_decay", cfg.train.weight_decay)
    mlflow.log_param("num_workers", cfg.train.num_workers)
    mlflow.log_param("num_classes", cfg.train.num_classes)
    repo = git.Repo(search_parent_directories=True)
    mlflow.log_param("git_commit", repo.head.object.hexsha)

    raw_dir = os.path.join(PROJECT_ROOT, cfg.data_dir, "raw", "GTSRB")
    train_folder = os.path.join(raw_dir, "Train")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    full_train = ImageFolder(train_folder, transform=transform)
    train_len = int(0.8 * len(full_train))
    val_len = len(full_train) - train_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    class TrafficSignModel(pl.LightningModule):
        def __init__(self, num_classes, lr, weight_decay):
            super().__init__()
            self.save_hyperparameters()
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=False
            )
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            self.log("train_loss", loss, prog_bar=True)
            mlflow.log_metric("train_loss", loss.item(), step=self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
            mlflow.log_metric("val_loss", loss.item(), step=self.current_epoch)
            mlflow.log_metric("val_acc", acc.item(), step=self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

    model = TrafficSignModel(
        num_classes=cfg.train.num_classes,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    if cfg.train.gpus and cfg.train.gpus > 0:
        accelerator = "gpu"
        devices = cfg.train.gpus
    else:
        accelerator = "cpu"
        devices = 1

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(PROJECT_ROOT, "models", "checkpoints"),
        filename="final_model",
        save_top_k=1,
        mode="min",
        monitor="val_loss",
    )

    mlf_logger = MLFlowLogger(
        experiment_name="traffic_sign_experiment",
        tracking_uri="http://127.0.0.1:8080",
        run_name=run_name,
    )

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=mlf_logger,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, train_loader, val_loader)
    mlflow.end_run()


if __name__ == "__main__":
    main()
