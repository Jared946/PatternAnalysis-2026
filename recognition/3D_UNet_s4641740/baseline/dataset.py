"""
loads 2D Medical Nifti Images for 2D baseline U-Net"
"""

import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

# Geometry of the U-Net
POOLING_LEVELS = 4
MIN_INPUT_SIZE = 188
UNET_REMAINDER = 12
UNET_STRIDE = 2 ** POOLING_LEVELS

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    valid_labels = np.isin(arr, np.arange(6))
    if not np.all(valid_labels):
        raise ValueError("Segmentation contains labels outside 0-5.")

    res = np.zeros(arr.shape + (6,), dtype=dtype)
    for c in range(6):
        res[..., c] = (arr == c)

    return res

def get_image_names(data_folder):
    """Return sorted paths to original MRI NIfTI volumes."""
    mri_folder = Path(data_folder) / "semantic_MRs"
    images = sorted(mri_folder.glob("*.nii.gz"))

    if not images:
        raise FileNotFoundError(
            f"No MRI volumes found in {mri_folder}"
        )

    return images

def get_label_names(image_names, data_folder):
    """Find the segmentation volume matching each MRI."""
    label_folder = Path(data_folder) / "semantic_labels_only"

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

def get_patient_ids(image_names):
    """Extract unique patient IDs from MRI filenames."""
    return sorted({
        image_path.name.split("_")[0]
        for image_path in image_names
    })

def split_patients(patient_ids):
    """Split unique patient IDs into train, validation and test sets."""
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


class MRI2DSliceDataset(Dataset):
    """Lazy-loading 2D MRI slices with volume-level normalisation."""

    def __init__(self, image_paths, label_paths, normalise=True):
        if len(image_paths) != len(label_paths):
            raise ValueError("Image and label counts do not match.")

        self.image_paths = [Path(p) for p in image_paths]
        self.label_paths = [Path(p) for p in label_paths]
        self.normalise = normalise

        # Keep lazy NIfTI proxies, not full image/mask arrays.
        self.image_niis = []
        self.label_niis = []
        self.volume_stats = []
        self.slice_index = []

        for volume_idx, (image_path, label_path) in enumerate(
            zip(self.image_paths, self.label_paths)
        ):
            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))

            if image_nii.shape != label_nii.shape:
                raise ValueError(
                    f"Shape mismatch for {image_path.name}: "
                    f"{image_nii.shape} vs {label_nii.shape}"
                )

            if not np.allclose(image_nii.affine, label_nii.affine):
                raise ValueError(
                    f"Affine mismatch for {image_path.name}"
                )

            # Calculate the original volume-level normalisation statistics.
            # Only one full image volume is temporarily materialised here.
            if self.normalise:
                volume = np.asarray(
                    image_nii.dataobj, dtype=np.float32
                )
                mean = float(volume.mean())
                std = float(volume.std())
                del volume
            else:
                mean, std = 0.0, 1.0

            self.image_niis.append(image_nii)
            self.label_niis.append(label_nii)
            self.volume_stats.append((mean, std))

            self.slice_index.extend(
                (volume_idx, slice_idx)
                for slice_idx in range(image_nii.shape[2])
            )

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_index[index]

        # Read only the requested 2D slices from disk.
        image = np.asarray(
            self.image_niis[volume_idx].dataobj[:, :, slice_idx],
            dtype=np.float32,
        )
        mask_raw = np.asarray(
            self.label_niis[volume_idx].dataobj[:, :, slice_idx]
        )

        # Preserve volume-level normalisation, not per-slice normalisation.
        if self.normalise:
            mean, std = self.volume_stats[volume_idx]
            image = (image - mean) / (std + 1e-8)

        # Validate labels before converting them to compact uint8 IDs.
        if not np.all(np.isfinite(mask_raw)):
            raise ValueError("Mask contains non-finite values.")

        mask_rounded = np.rint(mask_raw)

        if not np.allclose(mask_raw, mask_rounded):
            raise ValueError("Mask contains non-integer class labels.")

        if not np.all(np.isin(mask_rounded, np.arange(6))):
            raise ValueError("Segmentation contains labels outside 0-5.")

        mask_ids = mask_rounded.astype(np.uint8)

        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(
            to_channels(mask_ids).copy()
        ).permute(2, 0, 1)

        return image_tensor, mask_tensor

def paper_unet_size(size: int) -> int:
    """Largest original-paper U-Net input size not greater than size."""
    legal_size = size - (
        (size - UNET_REMAINDER) % UNET_STRIDE
    )

    if legal_size < MIN_INPUT_SIZE:
        raise ValueError(
            f"Dimension {size} is too small for the paper U-Net; "
            f"both dimensions must be at least {MIN_INPUT_SIZE}."
        )

    return legal_size

class PaperUNetMRI2DSliceDataset(MRI2DSliceDataset):
    """Dataset adapter for the 2D U-Net

    Returns:
        image: float32 tensor [1, H, W]
        target: uint8 one-hot tensor [6, H, W]
    """

    def __init__(self, *args, random_crop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_crop = random_crop

    def __getitem__(self, index):
        image, one_hot_mask = super().__getitem__(index)

        target = one_hot_mask

        height, width = image.shape[-2:]
        crop_height = paper_unet_size(height)
        crop_width = paper_unet_size(width)

        max_top = height - crop_height
        max_left = width - crop_width

        if self.random_crop:
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
        else:
            top = max_top // 2
            left = max_left // 2

        image = image[
            ..., top:top + crop_height,
            left:left + crop_width
        ]

        target = target[
            :, top:top + crop_height,
            left:left + crop_width
        ]

        return image.contiguous(), target.contiguous()