"""3-D HipMRI volume/patch dataset based on the baseline 2-D data convention."""
from __future__ import annotations
import random
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 6

def to_channels(arr: np.ndarray) -> np.ndarray:
    """Convert integer labels [D,H,W] into one-hot [C,D,H,W].

    Returns uint8 to keep the stored masks memory-efficient.
    """
    # Validate labels before creating one-hot channels
    if not np.all(np.isin(arr, np.arange(NUM_CLASSES))):
        raise ValueError("Segmentation contains labels outside 0-5.")

    # Create one binary channel for each segmentation class
    return np.stack(
        [(arr == c) for c in range(NUM_CLASSES)],
        axis=0,
    ).astype(np.uint8)

def get_image_names(root: Path):
    """Return sorted paths to MRI NIfTI volumes."""
    # Find all MRI volumes in the dataset directory
    mri_folder = Path(root) / "semantic_MRs"
    images = sorted(mri_folder.glob("*.nii.gz"))

    if not images:
        raise FileNotFoundError(
            f"No MRI volumes found in {mri_folder}"
        )

    return images

def get_label_names(image_names, data_root):
    """Find the segmentation volume matching each MRI."""
    # Match each MRI filename to its corresponding segmentation filename
    label_folder = Path(data_root) / "semantic_labels_only"

    label_names = []

    for image_path in image_names:
        label_name = image_path.name.replace(
            "_LFOV.nii.gz",
            "_SEMANTIC.nii.gz",
        )
        label_path = label_folder / label_name

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing segmentation for {image_path.name}: "
                f"{label_path}"
            )

        label_names.append(label_path)

    return label_names

def patient_id(path: Path):
    """Extract patient ID from the MRI filename."""
    return Path(path).name.split("_")[0]

def get_patient_ids(images):
    """Return sorted unique patient IDs from MRI paths."""
    return sorted({patient_id(path) for path in images})

def split_patients(patient_ids):
    """Split unique patient IDs into train, validation and test sets."""
    # Assign patients to fixed partitions to prevent patient-level leakage
    train_ids = {
        pid for pid in patient_ids
        if 4 <= int(pid[1:4]) <= 35
    }
    val_ids = {
        pid for pid in patient_ids
        if 36 <= int(pid[1:4]) <= 39
    }
    test_ids = {
        pid for pid in patient_ids
        if 40 <= int(pid[1:4]) <= 42
    }

    return train_ids, val_ids, test_ids

def select_pairs(images, labels, ids):
    # Keep only MRI/segmentation pairs belonging to the requested patients
    result = [(i, l) for i, l in zip(images, labels) if patient_id(i) in ids]
    if not result:
        raise ValueError("The requested patient split contains no volumes.")
    return result

def _pad(image, target, shape):
    # Calculate symmetric padding needed to reach the requested patch size
    padding = []
    for current, needed in zip(image.shape, shape):
        extra = max(0, needed - current)
        padding.append((extra // 2, extra - extra // 2))
    
    # Pad MRI data using edge values and segmentation masks using background
    if any(a or b for a, b in padding):
        image = np.pad(image, padding, mode="edge")
        target = np.pad(target, padding, mode="constant")
    return image, target

class MRI3DVolumeDataset(Dataset):
    """Return image ``float32 [1,D,H,W]`` and label ``uint8 [C,D,H,W]``.

    Supplying ``patch_size=(D,H,W)`` returns one random train patch per
    volume. Patch dimensions must be multiples of 16 for four pooling levels.
    """
    def __init__(
        self,
        image_names,
        label_names,
        patch_size=None,
        random_patch=False,
        normalise=True,
    ):
        if len(image_names) != len(label_names):
            raise ValueError("Image and label counts must match.")

        self.pairs = [
            (Path(i), Path(l))
            for i, l in zip(image_names, label_names)
        ]

        self.patch_size = tuple(patch_size) if patch_size else None
        self.random_patch = random_patch
        self.normalise = normalise

        # Ensure patch dimensions are compatible with the four pooling levels
        if self.patch_size and (
            len(self.patch_size) != 3
            or any(n < 16 or n % 16 for n in self.patch_size)
        ):
            raise ValueError(
                "patch_size must be three multiples of 16: D H W."
            )

        # Validate each image/mask pair before training
        for image_path, label_path in self.pairs:
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing MRI: {image_path}"
                )

            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing segmentation: {label_path}"
                )

            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))

            # Check that image and segmentation volumes have matching geometry
            if (
                image_nii.shape != label_nii.shape
                or len(image_nii.shape) != 3
            ):
                raise ValueError(
                    f"Expected matching 3-D volumes: {image_path.name}; "
                    f"image={image_nii.shape}, "
                    f"mask={label_nii.shape}"
                )

            if not np.allclose(image_nii.affine, label_nii.affine):
                raise ValueError(
                    f"Affine mismatch: {image_path.name}"
                )

            # Load the data to validate voxel values and segmentation labels
            image = np.asarray(image_nii.dataobj, dtype=np.float32)
            mask = np.asarray(label_nii.dataobj)

            if not np.all(np.isfinite(image)):
                raise ValueError(
                    f"MRI contains non-finite values: {image_path.name}"
                )

            if not np.all(np.isfinite(mask)):
                raise ValueError(
                    f"Mask contains non-finite values: {label_path.name}"
                )

            if not np.allclose(mask, np.round(mask)):
                raise ValueError(
                    f"Mask contains non-integer labels: {label_path.name}"
                )

            labels = np.unique(np.round(mask).astype(np.int64))
            if not np.all(np.isin(labels, np.arange(6))):
                raise ValueError(
                    f"Mask contains labels outside 0-5: "
                    f"{label_path.name}; found {labels.tolist()}"
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]

        # Load NIfTI volumes and transpose [X,Y,Z] into [D,H,W]
        # for PyTorch 3D convolutions
        image = np.asarray(nib.load(str(image_path)).dataobj,
                dtype=np.float32).transpose(2, 0, 1)
        target = np.asarray(nib.load(str(label_path)).dataobj,
                dtype=np.int64).transpose(2, 0, 1)
        
        # Check segmentation labels before processing the volume
        if not np.isin(target, np.arange(NUM_CLASSES)).all():
            raise ValueError(f"Invalid label in {label_path.name}")
        
        # Normalise MRI intensities independently for each volume
        if (self.normalise):
            image = (image - image.mean()) / (image.std() + 1e-8)
        
        if self.patch_size:
            # Pad smaller volumes, then select either a random or central patch
            image, target = _pad(image, target, self.patch_size)
            starts = [random.randint(0, n - p) if self.random_patch else
                    (n - p) // 2 for n, p in zip(image.shape, self.patch_size)]
            d, h, w = starts
            pd, ph, pw = self.patch_size
            image, target = (
                image[d:d+pd, h:h+ph, w:w+pw],
                target[d:d+pd, h:h+ph, w:w+pw],
            )
        
        # Convert the MRI and segmentation arrays into PyTorch tensors
        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0)

        # One-hot encode after any random patch crop has been applied
        target_one_hot = torch.from_numpy(to_channels(target.copy()))
        return image_tensor, target_one_hot
