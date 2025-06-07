#!/usr/bin/env python3
import csv
import os
import subprocess
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def dvc_pull():
    try:
        subprocess.check_call(["dvc", "pull"], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        pass


class TrafficSignModel(pl.LightningModule):
    def __init__(self, num_classes, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=False
        )
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


@hydra.main(version_base=None, config_path="../configs/infer", config_name="infer")
def main(cfg: DictConfig):
    dvc_pull()
    from scripts.dvc_utils import download_data

    download_data()

    model: pl.LightningModule = TrafficSignModel.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path
    )
    model.eval()

    img_dir = os.path.join(PROJECT_ROOT, cfg.images_dir)
    if not os.path.isdir(img_dir):
        print(f"Images directory not found: {img_dir}")
        return

    valid_exts = (".ppm", ".png", ".jpg", ".jpeg")
    filenames = sorted(
        [
            f
            for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
            and f.lower().endswith(valid_exts)
        ]
    )
    if not filenames:
        print(f"No image files found in {img_dir}")
        return

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    images = []
    for fname in filenames:
        path = os.path.join(img_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Warning: skipping file {fname}: {e}")
            continue
        images.append(transform(img))

    batch = torch.stack(images)

    with torch.no_grad():
        logits = model(batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    output_path = os.path.join(PROJECT_ROOT, cfg.output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class"])
        for fname, pred in zip(filenames, preds):
            writer.writerow([fname, int(pred)])
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
