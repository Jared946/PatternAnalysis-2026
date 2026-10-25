from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


INFERENCE_DIR = Path("C:/Users/User/SEMESTER 2/COMP3710/project/PatternAnalysis-2026/recognition/3D_UNet_s4641740/inference")
DATASET_DIR = Path("C:/Users/User/SEMESTER 2/COMP3710/project/HipMRI_Study_open")

MRI_DIR = DATASET_DIR / "semantic_MRs"
GT_DIR = DATASET_DIR / "semantic_labels_only"

NUM_CLASSES = 6  # Background = 0, foreground classes = 1 to 5

# Based on current patient split. Verify against dataset.py.
TEST_PATIENT_IDS = {"B040", "W041", "K042"}


def find_matching_file(folder, key, suffix):
    """Find exactly one file matching the selected case and suffix."""
    matches = list(folder.glob(f"*{key}{suffix}"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No matching file found in:\n{folder}\n"
            f"Search key: {key}\nExpected suffix: {suffix}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files matched '{key}' in {folder}:\n"
            + "\n".join(str(p) for p in matches)
            + "\nUse a more specific search key."
        )

    return matches[0]

def load_volume(path):
    """Load a NIfTI volume and its spatial metadata."""
    nii = nib.load(str(path))
    volume = np.asarray(nii.dataobj)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {volume.shape} for {path}"
        )

    return volume, nii.affine

def dice_score(prediction_mask, ground_truth_mask):
    """Calculate binary Dice for two masks."""
    pred_count = np.count_nonzero(prediction_mask)
    gt_count = np.count_nonzero(ground_truth_mask)

    denominator = pred_count + gt_count

    # Return None if both masks are empty.
    if denominator == 0:
        return None

    intersection = np.count_nonzero(
        prediction_mask & ground_truth_mask
    )

    return 2.0 * intersection / denominator

def calculate_dice(prediction, ground_truth):
    """Calculate per-class Dice and overall foreground Dice."""
    results = {}

    for class_id in range(1, NUM_CLASSES):
        results[class_id] = dice_score(
            prediction == class_id,
            ground_truth == class_id,
        )

    # All foreground voxels combined, regardless of class.
    results["foreground"] = dice_score(
        prediction > 0,
        ground_truth > 0,
    )

    # Macro average across classes present in either mask.
    # Classes absent from both masks are excluded.
    present_scores = [
        score
        for class_id, score in results.items()
        if isinstance(class_id, int) and score is not None
    ]

    results["macro_foreground"] = (
        float(np.mean(present_scores)) if present_scores else None
    )

    return results

def print_dice(results):
    print("\n" + "=" * 48)
    print("DICE RESULTS")
    print("=" * 48)

    for class_id in range(1, NUM_CLASSES):
        score = results[class_id]

        if score is None:
            print(f"Class {class_id}: N/A (empty in both masks)")
        else:
            print(f"Class {class_id}: {score:.4f}")

    print("-" * 48)

    for name, label in [
        ("foreground", "Overall foreground Dice"),
        ("macro_foreground", "Mean foreground class Dice"),
    ]:
        score = results[name]
        print(f"{label}: {score:.4f}" if score is not None
              else f"{label}: N/A")

    print("=" * 48)

def display_case(mri, ground_truth, prediction, slice_index, key):
    """Display MRI, ground truth, prediction, and boundary overlay."""

    # Show one slice perpendicular to the third array axis.
    image_slice = mri[:, :, slice_index]
    gt_slice = ground_truth[:, :, slice_index]
    pred_slice = prediction[:, :, slice_index]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("Original MRI")

    label_cmap = ListedColormap([
        "black", "red", "lime", "blue", "yellow", "magenta"
    ])

    axes[1].imshow(
        gt_slice.T,
        cmap=label_cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        interpolation="nearest",
        origin="lower",
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(
        pred_slice.T,
        cmap=label_cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        interpolation="nearest",
        origin="lower",
    )
    axes[2].set_title("Prediction")

    # Display MRI with ground-truth and prediction boundaries.
    axes[3].imshow(image_slice.T, cmap="gray", origin="lower")

    gt_foreground = (gt_slice > 0).T
    pred_foreground = (pred_slice > 0).T

    if gt_foreground.any():
        axes[3].contour(
            gt_foreground.astype(float),
            levels=[0.5],
            colors=["lime"],
            linewidths=1.2,
        )

    if pred_foreground.any():
        axes[3].contour(
            pred_foreground.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=1.2,
        )

    axes[3].set_title("Overlay: GT green, prediction red")

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(
        f"{key} | Slice {slice_index}/{mri.shape[2] - 1}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

def main():
    print("HipMRI 3D U-Net evaluation")
    print("Enter a substring from the filenames, e.g. 006_Week0")
    key = input("Case key: ").strip()

    if not key:
        print("No case key entered.")
        return

    mri_path = find_matching_file(
        MRI_DIR, key, "_LFOV.nii.gz"
    )
    gt_path = find_matching_file(
        GT_DIR, key, "_SEMANTIC.nii.gz"
    )
    pred_path = find_matching_file(
        INFERENCE_DIR, key, "_LFOV_segmentation_3d.nii.gz"
    )

    print("\nMatched files:")
    print(f"MRI:        {mri_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Prediction: {pred_path}")

    patient_id = mri_path.name.split("_")[0]

    if patient_id in TEST_PATIENT_IDS:
        print(f"\n{patient_id}: listed as a held-out test patient.")
    else:
        print(
            f"\nWARNING: {patient_id} is not in the configured test "
            "patient list."
        )

    mri, mri_affine = load_volume(mri_path)
    gt, gt_affine = load_volume(gt_path)
    pred, pred_affine = load_volume(pred_path)

    if mri.shape != gt.shape or gt.shape != pred.shape:
        raise ValueError(
            "Volume shape mismatch:\n"
            f"MRI: {mri.shape}\n"
            f"Ground truth: {gt.shape}\n"
            f"Prediction: {pred.shape}"
        )

    if not (
        np.allclose(mri_affine, gt_affine)
        and np.allclose(gt_affine, pred_affine)
    ):
        raise ValueError(
            "The NIfTI affine matrices differ. Check that the volumes "
            "are spatially aligned before calculating Dice."
        )

    for name, volume in [
        ("Ground truth", gt),
        ("Prediction", pred),
    ]:
        if not np.all(np.isfinite(volume)):
            raise ValueError(f"{name} contains non-finite values.")

        if not np.allclose(volume, np.round(volume)):
            raise ValueError(
                f"{name} contains non-integer labels. "
                "Check how the segmentation was saved."
            )

        labels = set(np.unique(volume).astype(int))
        if not labels.issubset(set(range(NUM_CLASSES))):
            raise ValueError(
                f"{name} contains unexpected labels: {sorted(labels)}"
            )

    gt = np.round(gt).astype(np.int16)
    pred = np.round(pred).astype(np.int16)

    print(f"\nVolume shape: {mri.shape}")
    print(f"Ground-truth labels: {np.unique(gt)}")
    print(f"Prediction labels:   {np.unique(pred)}")

    results = calculate_dice(pred, gt)
    print_dice(results)

    slice_text = input(
        f"\nSlice index (0-{mri.shape[2] - 1}, "
        "press Enter for middle slice): "
    ).strip()

    if slice_text:
        slice_index = int(slice_text)
        if not 0 <= slice_index < mri.shape[2]:
            raise ValueError("Slice index is outside the volume.")
    else:
        slice_index = mri.shape[2] // 2

    display_case(mri, gt, pred, slice_index, key)


if __name__ == "__main__":
    main()
