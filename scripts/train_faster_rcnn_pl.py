import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
import yaml
from dataset import ensure_dataset
from dotenv import load_dotenv
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class YOLODataset(Dataset):
    def __init__(self, img_dir: Path, label_dir: Path):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(img_dir))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(self.img_dir / img_name).convert("RGB")
        w, h = img.size

        label_path = self.label_dir / (Path(img_name).stem + ".txt")
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls_id) + 1)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32).clamp(min=0)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(max=w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(max=h)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if boxes.numel() > 0
            else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.uint8),
        }

        return self.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


class MCDetDataModule(LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 4, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        with open(self.data_dir / "data.yaml") as f:
            cfg = yaml.safe_load(f)

        self.class_names = cfg["names"]
        self.num_classes = len(self.class_names) + 1

        self.train_dataset = YOLODataset(
            self.data_dir / "train" / "images", self.data_dir / "train" / "labels"
        )
        self.val_dataset = YOLODataset(
            self.data_dir / "valid" / "images", self.data_dir / "valid" / "labels"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class MCDetModule(LightningModule):
    def __init__(self, num_classes: int, lr: float = 0.005):
        super().__init__()
        self.save_hyperparameters()

        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def _compute_loss(self, images, targets):
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values(), torch.tensor(0.0, device=self.device))
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss, loss_dict = self._compute_loss(images, targets)
        batch_size = len(images)
        self.log_dict(
            {f"train/{k}": v.item() for k, v in loss_dict.items()},
            on_step=True,
            batch_size=batch_size,
        )
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        self.map_metric.update(preds, targets)

    def on_validation_epoch_end(self):

        metrics = self.map_metric.compute()
        self.log("val/mAP50", metrics["map_50"], prog_bar=True)
        self.log("val/mAP75", metrics["map_75"])
        self.log("val/mAP50_95", metrics["map"], prog_bar=True)
        self.log("val/mar_100", metrics["mar_100"])
        if metrics.get("map_per_class") is not None and len(metrics["map_per_class"]) > 0:
            for cls_idx, ap in zip(metrics["classes"], metrics["map_per_class"]):
                cls_name = self.trainer.datamodule.class_names[int(cls_idx) - 1]  # type: ignore
                self.log(f"val/AP/{cls_name}", ap)

        self.map_metric.reset()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=self.hparams["lr"], momentum=0.9, weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
        return [optimizer], [scheduler]


def _warn_if_low_gpu_memory(min_free_gib: float = 6.0) -> None:
    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    total_gib = total_bytes / (1024**3)
    if free_gib < min_free_gib:
        print(
            f"WARNING: only {free_gib:.1f} GiB GPU memory free "
            f"(of {total_gib:.1f} GiB). Faster R-CNN needs several GiB free — "
            "stop other GPU jobs (`nvidia-smi`) before training."
        )


def main():
    load_dotenv()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")
        _warn_if_low_gpu_memory()

    data_dir = ensure_dataset()
    batch_size = 2
    accumulate_grad_batches = 2

    dm = MCDetDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=4)
    dm.setup()

    module = MCDetModule(num_classes=dm.num_classes, lr=0.005)

    precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"

    wandb_logger = WandbLogger(project="mcdetect", name="faster-rcnn-pl", log_model=False)
    wandb_logger.experiment.config.update(
        {
            "epochs": 50,
            "batch_size": dm.batch_size,
            "accumulate_grad_batches": accumulate_grad_batches,
            "precision": precision,
            "backbone": "resnet50-fpn",
            "dataset": "minecraft-mob-detection-v10",
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("weights"),
        filename="faster_rcnn_pl_{epoch:02d}",
        every_n_epochs=10,
        save_top_k=-1,
    )

    trainer = Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
    )

    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
