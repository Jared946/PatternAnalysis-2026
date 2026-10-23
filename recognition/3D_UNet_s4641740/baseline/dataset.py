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
    """
    Load 2D MRI slices and segmentation masks on demand.

    Each item returns:
        image: float32 tensor of shape (1, H, W)
        mask: float32 one-hot tensor of shape (6, H, W)
    """

    def __init__(
        self,
        image_names,
        label_names,
        normalise=True,
    ):
        if len(image_names) != len(label_names):
            raise ValueError(
                "Number of MRI and segmentation files must match."
            )

        self.image_names = list(image_names)
        self.label_names = list(label_names)
        self.normalise = normalise

        # Index every 2D slice
        self.slice_index = []
        self.images = []
        self.masks = []

        for volume_idx, (image_path, label_path) in enumerate(
            zip(self.image_names, self.label_names)
        ):
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing segmentation: {label_path}"
                )

            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))

            if image_nii.shape != label_nii.shape:
                raise ValueError(
                    f"Shape mismatch: {image_path.name}"
                )

            if len(image_nii.shape) != 3:
                raise ValueError(
                    f"Expected a 3D MRI: {image_path.name}"
                )

            if not np.allclose(image_nii.affine, label_nii.affine):
                raise ValueError(
                    f"Affine mismatch: {image_path.name}"
                )

            # Load the entire volume into RAM
            image_volume = np.asarray(
                image_nii.dataobj, dtype=np.float32
            )
            mask_volume = np.asarray(
                label_nii.dataobj, dtype=np.int8
            )

            self.images.append(image_volume)
            self.masks.append(mask_volume)

            self.slice_index.extend(
                (volume_idx, slice_idx)
                for slice_idx in range(image_nii.shape[2])
            )

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_index[index]

        image = self.images[volume_idx][:, :, slice_idx]
        mask = self.masks[volume_idx][:, :, slice_idx]

        if self.normalise:
            image = (image - image.mean()) / (
                image.std() + 1e-8
            )

        # PyTorch Conv2d expects (channels, height, width).
        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0)
        mask_tensor = (
            torch.from_numpy(to_channels(mask).copy())
            .permute(2, 0, 1)
            .float()
        )

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
        target: long tensor [H, W] containing class IDs
    """

    def __init__(self, *args, random_crop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_crop = random_crop

    def __getitem__(self, index):
        image, one_hot_mask = super().__getitem__(index)

        # CrossEntropyLoss expects integer class IDs, not one-hot masks.
        target = one_hot_mask.argmax(dim=0).to(dtype=torch.long)

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

        return (
            image[
                ..., top:top + crop_height,
                left:left + crop_width
            ].contiguous(),
            target[
                top:top + crop_height,
                left:left + crop_width
            ].contiguous(),
        )

class PaperUNetPreSlicedDataset(Dataset):
    """Load pre-sliced 2D MRI images and their segmentation masks."""

    def __init__(self, image_dir, label_dir, normalise=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.normalise = normalise

        self.image_paths = sorted(self.image_dir.glob("*.nii.gz"))

        if not self.image_paths:
            raise ValueError(f"No NIfTI images found in {self.image_dir}")

        self.label_paths = [
            self.label_dir / image_path.name.replace("case_", "seg_", 1)
            for image_path in self.image_paths
        ]

        for label_path in self.label_paths:
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing segmentation: {label_path}"
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.asarray(
            nib.load(str(self.image_paths[index])).dataobj,
            dtype=np.float32,
        ).squeeze()

        mask = np.asarray(
            nib.load(str(self.label_paths[index])).dataobj,
        ).squeeze()

        print('===================================================', flush=True)
        if image.shape != (256, 256):
            print(
                f"Unexpected dimensions: {self.image_paths[index].name}, "
                f"image={image.shape}, mask={mask.shape}",
                flush=True,
            )

        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected 2D slices, got image {image.shape}, "
                f"mask {mask.shape}"
            )

        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for "
                f"{self.image_paths[index].name}"
            )

        if not np.all(np.isfinite(image)):
            raise ValueError("MRI contains non-finite values.")

        if not np.all(np.isfinite(mask)):
            raise ValueError("Mask contains non-finite values.")

        if not np.allclose(mask, np.round(mask)):
            raise ValueError("Mask contains non-integer class values.")

        mask = np.round(mask).astype(np.int64)

        if not np.all(np.isin(mask, np.arange(6))):
            raise ValueError("Mask contains labels outside 0-5.")

        if self.normalise:
            image = (image - image.mean()) / (image.std() + 1e-8)

        # Apply the same paper U-Net input geometry as training before.
        height, width = image.shape
        crop_height = paper_unet_size(height)
        crop_width = paper_unet_size(width)

        top = (height - crop_height) // 2
        left = (width - crop_width) // 2

        image = image[
            top:top + crop_height,
            left:left + crop_width,
        ].copy()

        mask = mask[
            top:top + crop_height,
            left:left + crop_width,
        ].copy()

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        target_tensor = torch.from_numpy(mask).long()

        return image_tensor, target_tensor

