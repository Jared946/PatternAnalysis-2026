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

import csv
import time
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler

from dataset import (
    PaperUNetMRI2DSliceDataset,
    get_image_names,
    get_label_names,
    get_patient_ids,
    split_patients,
)
from modules import PaperUNet2D, crop_mask_to_logits

NUM_CLASSES = 6

DATA_FOLDER = Path("C:/Users/User/SEMESTER 2/COMP3710/project/HipMRI_Study_open")

class PerVolumeSliceSampler(Sampler):
    """Sample a fixed number of slices from each MRI volume per epoch."""

    def __init__(
        self,
        dataset,
        slices_per_volume=16,
        min_slice=0,
        max_slice=None,
    ):
        self.dataset = dataset
        self.slices_per_volume = slices_per_volume
        self.min_slice = min_slice
        self.max_slice = max_slice

        # Group eligible dataset indices by their source MRI volume
        self.volume_indices = {}

        for dataset_idx, (volume_idx, slice_idx) in enumerate(
            dataset.slice_index
        ):
            if slice_idx < min_slice:
                continue

            if max_slice is not None and slice_idx > max_slice:
                continue

            self.volume_indices.setdefault(volume_idx, []).append(
                dataset_idx
            )

        if not self.volume_indices:
            raise ValueError("No slices available in the selected range.")

    def __iter__(self):
        selected = []

        # Sample the same maximum number of slices from each volume
        for indices in self.volume_indices.values():
            count = min(self.slices_per_volume, len(indices))
            selected.extend(random.sample(indices, count))

        # Shuffle the selected slices across volumes
        random.shuffle(selected)
        return iter(selected)

    def __len__(self):
        return sum(
            min(self.slices_per_volume, len(indices))
            for indices in self.volume_indices.values()
        )

def patient_id(image_path: Path) -> str:
    """Extract patient ID from the MRI filename."""
    return image_path.name.split("_")[0]

def make_dataset(image_names, label_names, patient_ids):
    # Select the MRI volumes assigned to this patient partition
    pairs = [
        (image, label)
        for image, label in zip(image_names, label_names)
        if patient_id(image) in patient_ids
    ]

    if not pairs:
        raise ValueError("The selected patient partition is empty.")

    images, labels = zip(*pairs)

    # Load slices lazily and apply the dataset's normalisation
    return PaperUNetMRI2DSliceDataset(
        images,
        labels,
        normalise=True,
        random_crop=False,
    )

def run_epoch(model, loader, criterion, device, optimizer=None):
    # Supplying an optimizer enables training, otherwise run validation
    training = optimizer is not None
    model.train(training)

    # Accumulate epoch metrics on the device to reduce synchronisation
    total_loss = torch.zeros((), device=device)
    total_dice = torch.zeros((), device=device)
    class_intersections = torch.zeros(NUM_CLASSES, device=device)
    class_denominators = torch.zeros(NUM_CLASSES, device=device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Convert one-hot segmentation masks into class-index targets
        target_ids = targets.argmax(dim=1).long()

        with torch.set_grad_enabled(training):
            logits = model(images)

            # The U-Net uses valid convolutions, so align the target
            # with the smaller output dimensions
            target_ids = crop_mask_to_logits(target_ids, logits)
            loss = criterion(logits, target_ids)

            # Update model weights only during training
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        # Accumulate loss on the device, synchronise only after the epoch
        total_loss += loss.detach()

        predictions = logits.detach().argmax(dim=1)

        # Calculate foreground Dice scores for this batch and accumulate
        # intersections and denominators for the epoch-level class scores
        batch_scores = []
        for class_id in range(1, NUM_CLASSES):
            predicted_class = predictions == class_id
            target_class = target_ids == class_id

            intersection = (predicted_class & target_class).sum()
            denominator = predicted_class.sum() + target_class.sum()

            class_intersections[class_id] += intersection
            class_denominators[class_id] += denominator

            # Preserve the original batch-level Dice averaging behaviour
            if denominator.item() > 0:
                batch_scores.append(
                    2.0 * intersection.float() / denominator.float()
                )

        if batch_scores:
            total_dice += torch.stack(batch_scores).mean()
        else:
            total_dice += 1.0

    if len(loader) == 0:
        raise ValueError("DataLoader contains no batches.")

    # Calculate mean loss and Dice across all batches in the epoch
    mean_loss = (total_loss / len(loader)).item()
    mean_dice = (total_dice / len(loader)).item()

    # Per-class Dice aggregated across all pixels in this epoch
    per_class_dice = {}
    for class_id in range(1, NUM_CLASSES):
        denominator = class_denominators[class_id].item()

        if denominator > 0:
            per_class_dice[class_id] = (
                2.0 * class_intersections[class_id].item() / denominator
            )
        else:
            per_class_dice[class_id] = None

    return mean_loss, mean_dice, per_class_dice

def main():
    # Configure training options and paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_FOLDER)
    parser.add_argument("--output", type=Path,
            default=Path("checkpoints/unet2d_hipmri.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int,
            default=1, help="All slices in a batch must have equal dimensions.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
            default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    # Seed the random number generators for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Discover MRI volumes and their corresponding segmentation labels
    image_names = get_image_names(args.data_root)
    label_names = get_label_names(image_names, args.data_root)

    # Split by patient IDs
    patient_ids = get_patient_ids(image_names)

    train_ids, validation_ids, test_ids = split_patients(patient_ids)

    train_dataset = make_dataset(
        image_names, label_names, train_ids
    )
    validation_dataset = make_dataset(
        image_names, label_names, validation_ids
    )

    # Configure data loading, using pinned memory when training on CUDA
    pin_memory = device.type == "cuda"
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.workers > 0,
    }

    # Sample a fixed number of slices from the central region of each
    # training volume during every epoch
    train_sampler = PerVolumeSliceSampler(
        train_dataset,
        slices_per_volume=16,
        min_slice=32,
        max_slice=96,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    # Iterate through the complete validation dataset without shuffling
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    # Initialise the model, optimiser and multi-class classification loss
    model = PaperUNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Restore model and optimiser state when continuing a previous run
    start_epoch, best_dice, best_epoch = 1, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_validation_dice", best_dice)
        best_epoch = checkpoint.get("epoch", 0)

    # One CSV file alongside the checkpoint
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # If resuming, append to an existing log, otherwise start a fresh log
    csv_mode = "a" if args.resume and csv_path.exists() else "w"

    # Record aggregate and per-class metrics, runtime and GPU memory use
    csv_fields = [
        "epoch",
        "train_loss",
        "train_dice",
        "val_loss",
        "val_dice",
        *[f"train_dice_class_{c}" for c in range(1, NUM_CLASSES)],
        *[f"val_dice_class_{c}" for c in range(1, NUM_CLASSES)],
        "epoch_runtime_seconds",
        "cumulative_runtime_seconds",
        "peak_gpu_allocated_gib",
        "peak_gpu_reserved_gib",
    ]

    overall_peak_allocated = 0.0
    overall_peak_reserved = 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create a new CSV header if starting a fresh run
    if csv_mode == "w" or not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # Track elapsed time for this training run
    run_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        # Reset CUDA peak-memory measurements at the start of each epoch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        epoch_start = time.perf_counter()

        # Train the model and then evaluate it on the validation set
        train_loss, train_dice, train_class_dice = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_dice, val_class_dice = run_epoch(
            model, validation_loader, criterion, device
        )

        # Synchronise CUDA before recording runtime and memory statistics
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_runtime = time.perf_counter() - epoch_start
        cumulative_runtime = time.perf_counter() - run_start

        # Record peak GPU memory usage for the current epoch
        if device.type == "cuda":
            peak_allocated = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(device) / 1024**3
            )
        else:
            peak_allocated = None
            peak_reserved = None

        if peak_allocated is not None:
            overall_peak_allocated = max(
                overall_peak_allocated, peak_allocated
            )
            overall_peak_reserved = max(
                overall_peak_reserved, peak_reserved
            )

        # Save a checkpoint whenever validation Dice improves
        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_validation_dice": best_dice,
                "in_channels": 1,
                "num_classes": NUM_CLASSES,
                "architecture": "modules.PaperUNet2D",
            }, args.output)

        # Collect epoch metrics and performance measurements for the CSV
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "epoch_runtime_seconds": epoch_runtime,
            "cumulative_runtime_seconds": cumulative_runtime,
            "peak_gpu_allocated_gib": peak_allocated,
            "peak_gpu_reserved_gib": peak_reserved,
        }

        # Add the individual foreground-class Dice scores
        for class_id in range(1, NUM_CLASSES):
            row[f"train_dice_class_{class_id}"] = (
                train_class_dice[class_id]
            )
            row[f"val_dice_class_{class_id}"] = (
                val_class_dice[class_id]
            )

        # Persist results immediately after each epoch
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        print(f"\nEpoch {epoch:03d}/{args.epochs}")

        if improved:
            print(f"  Saved best checkpoint to {args.output}")

    # Print the best validation result and the locations of saved outputs
    total_runtime = time.perf_counter() - run_start

    print("\n=== Training Summary ===")
    print(f"Best validation Dice:   {best_dice:.4f}")
    print(f"Best epoch:             {best_epoch}")
    print(f"Total runtime:          {total_runtime / 60:.2f}")
    print(f"Checkpoint path: {args.output}")
    print(f"CSV log path:    {csv_path}")


if __name__ == "__main__":
    main()

