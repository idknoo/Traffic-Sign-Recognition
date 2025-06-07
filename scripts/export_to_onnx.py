#!/usr/bin/env python3
import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


@hydra.main(version_base=None, config_path="../configs/infer", config_name="infer")
def export(cfg: DictConfig):
    """
    Экспорт чекпоинта Lightning в ONNX.
    """
    model = TrafficSignModel.load_from_checkpoint(checkpoint_path=cfg.ckpt_path)
    model.eval()

    export_dir = os.path.join(PROJECT_ROOT, "models", "onnx")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "traffic_sign.onnx")

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    export()
