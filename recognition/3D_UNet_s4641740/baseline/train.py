"""Train ``modules.PaperUNet2D`` on ``dataset.PaperUNetMRI2DSliceDataset``.

The HipMRI data root must contain ``semantic_MRs_anon`` and
``semantic_labels_anon`` as expected by dataset.py.  Splitting is performed
by patient, not by individual slice, so slices from one patient cannot leak
between train/validation/test partitions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import (
    PaperUNetMRI2DSliceDataset,
    get_image_names,
    get_label_names,
    get_patient_ids,
    split_patients,
)
from modules import PaperUNet2D, crop_mask_to_logits

NUM_CLASSES = 6

DATA_FOLDER = Path(
        "/home/groups/comp3710/HipMRI_Study_open"
        )

def patient_id(image_path: Path) -> str:
    """Extract patient ID from the MRI filename."""
    return image_path.name.split("_")[0]

def select_partition(image_names, label_names, wanted_ids):
    return zip(*[(image, label) for image, label in zip(image_names, label_names)
        if patient_id(image) in wanted_ids])

def make_dataset(image_names, label_names, patient_ids):
    pairs = [
        (image, label)
        for image, label in zip(image_names, label_names)
        if patient_id(image) in patient_ids
    ]

    if not pairs:
        raise ValueError("The selected patient partition is empty.")

    images, labels = zip(*pairs)

    return PaperUNetMRI2DSliceDataset(
        images,
        labels,
        normalise=True,
        random_crop=False,
    )

def mean_foreground_dice(logits: Tensor, target: Tensor) -> float:
    """Mean Dice across HipMRI foreground classes 1--5 for one batch."""
    prediction = logits.argmax(dim=1)
    scores = []
    for class_id in range(1, NUM_CLASSES):
        predicted_class, target_class = prediction == class_id, target == class_id
        denominator = predicted_class.sum().item() + target_class.sum().item()
        if denominator:
            scores.append(2.0 * (predicted_class & target_class).sum().item() / denominator)
    return float(np.mean(scores)) if scores else 1.0

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Convert one-hot masks [N, 6, H, W] to class IDs [N, H, W].
        target_ids = targets.argmax(dim=1).long()

        with torch.set_grad_enabled(training):
            logits = model(images)

            # The U-Net uses valid convolutions, so align the target
            # with the smaller output dimensions.
            target_ids = crop_mask_to_logits(target_ids, logits)

            loss = criterion(logits, target_ids)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_dice += mean_foreground_dice(
            logits.detach(), target_ids
        )

    return total_loss / len(loader), total_dice / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_FOLDER)
    parser.add_argument("--output", type=Path,
            default=Path("checkpoints/unet2d_hipmri.pt"))
    parser.add_argument("--epochs", type=int, default=1) #was 100
    parser.add_argument("--batch-size", type=int,
            default=1, help="All slices in a batch must have equal dimensions.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
            default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    image_names = get_image_names(args.data_root)
    label_names = get_label_names(image_names, args.data_root)

    patient_ids = get_patient_ids(image_names)

    train_ids, validation_ids, test_ids = split_patients(patient_ids)

    train_dataset = make_dataset(
        image_names, label_names, train_ids
    )
    validation_dataset = make_dataset(
        image_names, label_names, validation_ids
    )
    test_dataset = make_dataset(
        image_names, label_names, test_ids
    )

    images, targets = next(iter(DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory)

    model = PaperUNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    start_epoch, best_dice = 1, -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_validation_dice", best_dice)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_dice = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer)
        val_loss, val_dice = run_epoch(model, validation_loader, criterion, device)
        print(f"Epoch {epoch:03d}/{args.epochs}: train loss={train_loss:.4f}, dice={train_dice:.4f} | val loss={val_loss:.4f}, dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_validation_dice": best_dice,
                "in_channels": 1,
                "num_classes": NUM_CLASSES,
                "architecture": "modules.PaperUNet2D",
            }, args.output)
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()

