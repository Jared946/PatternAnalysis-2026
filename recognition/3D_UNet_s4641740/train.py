"""Train UNet3D on patient-separated HipMRI patches."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import csv
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import (
    MRI3DVolumeDataset,
    get_image_names,
    get_label_names,
    get_patient_ids,
    select_pairs,
    split_patients,
)
from modules import UNet3D

NUM_CLASSES = 6

class CrossEntropyDiceLoss(nn.Module):
    """Cross-entropy plus multiclass soft Dice loss.

    Input targets are one-hot masks shaped [N,C,D,H,W].
    Background class 0 is excluded from the Dice component.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        ce_weight=1.0,
        dice_weight=1.0,
        smooth=1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets_one_hot):
        # Check that predictions and one-hot targets have matching 3D shapes
        # logits:          [N,C,D,H,W]
        # targets_one_hot: [N,C,D,H,W]
        if logits.ndim != 5:
            raise ValueError(
                f"Expected 5-D logits, got {tuple(logits.shape)}"
            )

        if targets_one_hot.shape != logits.shape:
            raise ValueError(
                f"Expected one-hot targets shaped like logits; "
                f"got targets={tuple(targets_one_hot.shape)}, "
                f"logits={tuple(logits.shape)}"
            )

        # Cross-entropy expects integer class IDs, not one-hot masks
        targets = targets_one_hot.argmax(dim=1).long()
        ce = self.ce_loss(logits, targets)

        # Calculate soft Dice using the one-hot targets and predicted probabilities
        target_probs = targets_one_hot.to(dtype=logits.dtype)
        probabilities = torch.softmax(logits, dim=1)

        # Sum intersections and class volumes across the batch and spatial dimensions
        dims = (0, 2, 3, 4)

        intersection = (
            probabilities * target_probs
        ).sum(dim=dims)

        denominator = (
            probabilities + target_probs
        ).sum(dim=dims)

        dice_per_class = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        # Exclude background class 0 from the Dice loss
        dice_loss = 1.0 - dice_per_class[1:].mean()

        # Combine the weighted cross-entropy and Dice losses
        return (
            self.ce_weight * ce
            + self.dice_weight * dice_loss
        )

def epoch(model, loader, loss_fn, device, optimizer = None):
    """Run one training or validation epoch.

    Loss is averaged over batches. Dice is calculated from accumulated
    class-wise voxel counts across the entire loader, rather than
    averaging the Dice score of each batch equally.
    """
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    batch_count = 0
    samples_seen = 0

    # Accumulate foreground intersections and denominators across all batches
    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    denominators = np.zeros(NUM_CLASSES, dtype=np.float64)

    # Enable gradient calculation only during training
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            samples_seen += images.shape[0]

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Convert one-hot masks to integer IDs for hard Dice metrics
            target_ids = targets.argmax(dim=1)

            logits = model(images)
            loss = loss_fn(logits, targets)

            # Update model weights only during training
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
            batch_count += 1

            # Accumulate hard Dice statistics for each foreground class
            predictions = logits.detach().argmax(dim=1)

            for c in range(1, NUM_CLASSES):
                pred_mask = predictions == c
                target_mask = target_ids == c

                intersections[c] += (
                    pred_mask & target_mask
                ).sum().item()

                denominators[c] += (
                    pred_mask.sum().item()
                    + target_mask.sum().item()
                )

    # Ensure the loader contained data and every sample was processed
    if batch_count == 0:
        raise ValueError(
            "The DataLoader contains no batches. Check the patient split "
            "and dataset construction."
        )

    expected_samples = len(loader.dataset)

    if samples_seen != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} samples, "
            f"but processed {samples_seen}."
        )

    # Calculate mean foreground Dice from the accumulated voxel counts
    class_scores = []

    for c in range(1, NUM_CLASSES):
        if denominators[c] > 0:
            class_scores.append(
                2.0 * intersections[c] / denominators[c]
            )

    mean_dice = (
        float(np.mean(class_scores)) if class_scores else 1.0
    )

    return loss_sum / batch_count, mean_dice

def evaluate_test(model, test_loader, device):
    """Evaluate complete test volumes and report patient-level Dice.

    Accumulates class intersections and denominators across all volumes
    belonging to each patient. Background class 0 is excluded.
    """
    model.eval()

    # Store accumulated Dice statistics separately for each patient
    # patient_id -> per-class intersection and denominator
    patient_stats = {}
    volume_index = 0

    with torch.inference_mode():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Convert the one-hot ground truth for class-wise Dice metrics.
            target_ids = targets.argmax(dim=1)

            # Batch size must be 1 because full volumes can differ in size.
            if images.shape[0] != 1:
                raise ValueError(
                    "Full-volume test evaluation requires batch_size=1."
                )

            # Identify the patient associated with the current volume
            image_path = test_loader.dataset.pairs[volume_index][0]
            volume_index += 1
            pid = image_path.name.split("_")[0]

            # Preserve the original spatial dimensions before padding
            _, _, d, h, w = images.shape

            # Pad at the end of each dimension to multiples of 16
            pd = (-d) % 16
            ph = (-h) % 16
            pw = (-w) % 16

            if pd or ph or pw:
                images = F.pad(
                    images,
                    (0, pw, 0, ph, 0, pd),
                    mode="replicate",
                )

            logits = model(images)

            # Remove the padding from the predictions
            logits = logits[:, :, :d, :h, :w]
            predictions = logits.argmax(dim=1)

            # Initialise metric accumulators for a patient when first encountered
            if pid not in patient_stats:
                patient_stats[pid] = {
                    "intersection": np.zeros(NUM_CLASSES, dtype=np.float64),
                    "denominator": np.zeros(NUM_CLASSES, dtype=np.float64),
                }

            stats = patient_stats[pid]

            # Accumulate voxel-level Dice statistics across the patient's volumes
            for c in range(1, NUM_CLASSES):
                pred_mask = predictions == c
                target_mask = target_ids == c

                stats["intersection"][c] += (
                    pred_mask & target_mask
                ).sum().item()

                stats["denominator"][c] += (
                    pred_mask.sum().item()
                    + target_mask.sum().item()
                )

    # Calculate Dice for each patient, excluding classes absent from
    # both prediction and target for that patient
    patient_scores = {}
    class_scores = {c: [] for c in range(1, NUM_CLASSES)}

    for pid, stats in patient_stats.items():
        scores = []

        for c in range(1, NUM_CLASSES):
            denom = stats["denominator"][c]

            if denom > 0:
                score = 2.0 * stats["intersection"][c] / denom
                scores.append(score)
                class_scores[c].append(score)

        patient_scores[pid] = float(np.mean(scores)) if scores else 1.0

    return patient_scores

def evaluate_validation(model, loader, loss_fn, device):
    """Evaluate full validation volumes and calculate patient-level Dice."""
    model.eval()

    # Accumulate validation metrics by patient across all their volumes
    patient_stats = {}
    loss_sum = 0.0
    volume_count = 0

    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Integer IDs are used for hard Dice evaluation
            target_ids = targets.argmax(dim=1)

            # Batch size must be 1 because full volumes can differ in size
            if images.shape[0] != 1:
                raise ValueError(
                    "Full-volume validation requires batch_size=1."
                )

            # Identify the patient associated with the current volume
            image_path = loader.dataset.pairs[index][0]
            pid = image_path.name.split("_")[0]

            # Preserve the original spatial dimensions before padding
            _, _, d, h, w = images.shape

            # Pad dimensions to multiples of 16 for the 3D U-Net
            pd = (-d) % 16
            ph = (-h) % 16
            pw = (-w) % 16

            if pd or ph or pw:
                images = F.pad(
                    images,
                    (0, pw, 0, ph, 0, pd),
                    mode="replicate",
                )

            logits = model(images)

            # Remove padding so predictions match the original target
            logits = logits[:, :, :d, :h, :w]

            # Calculate validation loss on the unpadded volume
            loss = loss_fn(logits, targets)
            loss_sum += loss.item()
            volume_count += 1

            predictions = logits.argmax(dim=1)

            # Initialise metric accumulators for each patient
            if pid not in patient_stats:
                patient_stats[pid] = {
                    "intersection": np.zeros(NUM_CLASSES),
                    "denominator": np.zeros(NUM_CLASSES),
                }

            stats = patient_stats[pid]

            # Accumulate voxel-level Dice statistics for each foreground class
            for c in range(1, NUM_CLASSES):
                pred_mask = predictions == c
                target_mask = target_ids == c

                stats["intersection"][c] += (
                    pred_mask & target_mask
                ).sum().item()

                stats["denominator"][c] += (
                    pred_mask.sum().item()
                    + target_mask.sum().item()
                )

    if volume_count == 0:
        raise ValueError("Validation loader contains no volumes.")

    # Calculate each patient's mean foreground Dice, then average across patients
    patient_scores = []
    per_class_patient_scores = {
        c: [] for c in range(1, NUM_CLASSES)
    }

    for stats in patient_stats.values():
        class_scores = []

        for c in range(1, NUM_CLASSES):
            denom = stats["denominator"][c]

            if denom > 0:
                class_dice = (
                    2.0 * stats["intersection"][c] / denom
                )

                class_scores.append(class_dice)
                per_class_patient_scores[c].append(class_dice)

        if class_scores:
            patient_scores.append(float(np.mean(class_scores)))

    if not patient_scores:
        raise ValueError("No foreground Dice scores could be calculated.")

    # Calculate the mean Dice for each foreground class across patients
    per_class_dice = {
        c: (
            float(np.mean(scores))
            if scores
            else 0.0
        )
        for c, scores in per_class_patient_scores.items()
    }

    return (
        loss_sum / volume_count,
        float(np.mean(patient_scores)),
        per_class_dice,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Configure dataset, checkpoint, model, training and device options
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("C:/Users/User/SEMESTER 2/COMP3710/project/HipMRI_Study_open"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/unet3d_hipmri.pt"),
    )
    parser.add_argument(
        "--patch-size",
        nargs=3,
        type=int,
        default=(64, 128, 128),
        metavar=("D", "H", "W"),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
        help="Base channel width",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Discover MRI volumes, their corresponding labels and patient IDs
    images = get_image_names(args.data_root)

    if not images:
        raise FileNotFoundError(
            f"No volumes found under {args.data_root / 'semantic_MRs'}"
        )

    labels = get_label_names(images, args.data_root)
    patient_ids = get_patient_ids(images)

    # Assign patients to separate training, validation and test partitions
    train_ids, val_ids, test_ids = split_patients(patient_ids)

    train_pairs = select_pairs(images, labels, train_ids)
    val_pairs = select_pairs(images, labels, val_ids)
    test_pairs = select_pairs(images, labels, test_ids)

    # Ensure that all three partitions contain matching MRI and label pairs
    if not train_pairs:
        raise ValueError("No training volumes matched the training IDs.")
    if not val_pairs:
        raise ValueError("No validation volumes matched the validation IDs.")
    if not test_pairs:
        raise ValueError("No test volumes matched the test IDs.")

    # Verify that every discovered MRI volume appears in exactly one split
    all_pairs = train_pairs + val_pairs + test_pairs

    all_image_paths = [str(pair[0]) for pair in all_pairs]
    expected_image_paths = {str(image) for image in images}

    assert len(all_image_paths) == len(set(all_image_paths)), (
        "Duplicate MRI volumes found across splits!"
    )

    assert set(all_image_paths) == expected_image_paths, (
        "Some MRI volumes are missing from the splits, "
        "or unexpected volumes have been included!"
    )

    # Create datasets: random patches for training and complete volumes
    # for validation and test evaluation
    train_ds = MRI3DVolumeDataset(
        *zip(*train_pairs),
        patch_size=tuple(args.patch_size),
        random_patch=True,
    )
    val_ds = MRI3DVolumeDataset(
        *zip(*val_pairs),
        patch_size=None,
    )
    test_ds = MRI3DVolumeDataset(
        *zip(*test_pairs),
        patch_size=None,
    )

    device = torch.device(args.device)

    # Shared settings for the training DataLoader
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }

    # Shuffle training patches, but preserve volume order for evaluation
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Initialise the 3D U-Net, optimiser and combined loss function
    model = UNet3D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_fn = CrossEntropyDiceLoss(
        num_classes=NUM_CLASSES,
        ce_weight=1.0,
        dice_weight=1.0,
    ).to(device)

    # Create the checkpoint directory and CSV log path
    args.output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = args.output.with_suffix(".csv")

    best_val_dice = -float("inf")
    checkpoint_saved = False

    # Track the highest GPU memory usage observed across epochs
    overall_peak_allocated = 0.0
    overall_peak_reserved = 0.0

    training_start = time.perf_counter()

    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint: {args.output}")
    print(f"CSV log: {csv_path}\n")

    # Define the metrics recorded after every epoch
    csv_fields = [
        "epoch",
        "train_loss",
        "train_dice",
        "val_loss",
        "val_dice",
        "best_val_dice",
        "epoch_seconds",
        "peak_gpu_allocated_gib",
        "peak_gpu_reserved_gib",
        "learning_rate",
    ]

    # Start a new log for this run and record each epoch's results
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        csv_file.flush()

        for e in range(1, args.epochs + 1):
            # Reset peak statistics so memory is measured separately
            # for each epoch, including training and validation
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start = time.perf_counter()

            # Train the model on randomly sampled patches
            train_loss, train_dice = epoch(
                model,
                train_loader,
                loss_fn,
                device,
                optimizer,
            )

            # Evaluate on complete validation volumes
            val_loss, val_dice, val_class_dice = evaluate_validation(
                model,
                val_loader,
                loss_fn,
                device,
            )

            # Save the model whenever it achieves a new best validation Dice
            if val_dice > best_val_dice:
                best_val_dice = val_dice

                torch.save(
                    {
                        "epoch": e,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_validation_dice": best_val_dice,
                        "in_channels": 1,
                        "num_classes": NUM_CLASSES,
                        "base_channels": args.base_channels,
                        "architecture": "UNet3D",
                    },
                    args.output,
                )

                checkpoint_saved = args.output.is_file()
                print(f"Saved best checkpoint to {args.output}")

            # Synchronise before stopping the timer so CUDA operations
            # are included in the epoch duration
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            epoch_seconds = time.perf_counter() - epoch_start

            # Record peak GPU memory usage for this epoch
            if device.type == "cuda":
                peak_allocated = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                )
                peak_reserved = (
                    torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                )

                overall_peak_allocated = max(
                    overall_peak_allocated, peak_allocated
                )
                overall_peak_reserved = max(
                    overall_peak_reserved, peak_reserved
                )
            else:
                peak_allocated = float("nan")
                peak_reserved = float("nan")

            learning_rate = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {e:03d}/{args.epochs}")

            # Write the epoch metrics to CSV
            writer.writerow(
                {
                    "epoch": e,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "best_val_dice": best_val_dice,
                    "epoch_seconds": epoch_seconds,
                    "peak_gpu_allocated_gib": peak_allocated,
                    "peak_gpu_reserved_gib": peak_reserved,
                    "learning_rate": learning_rate,
                }
            )
            csv_file.flush()

    # Calculate total runtime after all training epochs have completed
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_seconds = time.perf_counter() - training_start

    print("\n=== TRAINING SUMMARY ===")
    print(f"Total training runtime: {total_seconds / 60:.2f} minutes")
    print(f"Best validation Dice:   {best_val_dice:.4f}")
    print(f"Checkpoint saved:       {checkpoint_saved}")
    print(f"Checkpoint path:        {args.output}")
    print(f"CSV log path:           {csv_path}")

    # Load the best checkpoint and evaluate it on the held-out test set
    if not checkpoint_saved:
        raise RuntimeError("No best checkpoint was saved; cannot evaluate test set.")

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']}")
    test_scores = evaluate_test(model, test_loader, device)

    print("\n=== TEST RESULTS ===")
    for pid, score in test_scores.items():
        print(f"{pid}: {score:.4f}")

    print(f"Mean patient Dice: {np.mean(list(test_scores.values())):.4f}")

if __name__ == '__main__':
    main()
