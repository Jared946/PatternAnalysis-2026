"""3-D HipMRI volume/patch dataset based on the baseline 2-D data convention."""
from __future__ import annotations
import random
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 6

def get_image_names(root: Path):
    """Return sorted paths to MRI NIfTI volumes."""
    mri_folder = Path(root) / "semantic_MRs"
    images = sorted(mri_folder.glob("*.nii.gz"))

    if not images:
        raise FileNotFoundError(
            f"No MRI volumes found in {mri_folder}"
        )

    return images

def get_label_names(image_names, data_root):
    """Find the segmentation volume matching each MRI."""
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
    train_ids = {f"case_{i:03d}" for i in range(4, 36)}
    val_ids = {f"case_{i:03d}" for i in range(36, 40)}
    test_ids = {f"case_{i:03d}" for i in range(40, 43)}

    available_ids = set(patient_ids)

    # Check that every expected patient exists.
    assert train_ids <= available_ids, (
        f"Missing training IDs: {train_ids - available_ids}"
    )
    assert val_ids <= available_ids, (
        f"Missing validation IDs: {val_ids - available_ids}"
    )
    assert test_ids <= available_ids, (
        f"Missing test IDs: {test_ids - available_ids}"
    )

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return train_ids, val_ids, test_ids

def select_pairs(images, labels, ids):
    result = [(i, l) for i, l in zip(images, labels) if patient_id(i) in ids]
    if not result:
        raise ValueError("The requested patient split contains no volumes.")
    return result

def _pad(image, target, shape):
    padding = []
    for current, needed in zip(image.shape, shape):
        extra = max(0, needed - current)
        padding.append((extra // 2, extra - extra // 2))
    if any(a or b for a, b in padding):
        image = np.pad(image, padding, mode="edge")
        target = np.pad(target, padding, mode="constant")
    return image, target

class MRI3DVolumeDataset(Dataset):
    """Return image ``float32 [1,D,H,W]`` and label ``long [D,H,W]``.

    Supplying ``patch_size=(D,H,W)`` returns one random train patch per
    volume.  Patch dimensions must be multiples of 16 for four pooling levels.
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

        if self.patch_size and (
            len(self.patch_size) != 3
            or any(n < 16 or n % 16 for n in self.patch_size)
        ):
            raise ValueError(
                "patch_size must be three multiples of 16: D H W."
            )

        # Validate each image/mask pair before training.
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

            # Validate the actual data, not just the NIfTI metadata.
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
        # NIfTI [X,Y,Z] -> PyTorch Conv3d [D,H,W].
        image = np.asarray(nib.load(str(image_path)).dataobj,
                dtype=np.float32).transpose(2, 0, 1)
        target = np.asarray(nib.load(str(label_path)).dataobj,
                dtype=np.int64).transpose(2, 0, 1)
        if not np.isin(target, np.arange(NUM_CLASSES)).all():
            raise ValueError(f"Invalid label in {label_path.name}")
        image = (image - image.mean()) / (image.std() + 1e-8)
        if self.patch_size:
            image, target = _pad(image, target, self.patch_size)
            starts = [random.randint(0, n - p) if self.random_patch else
                    (n - p) // 2 for n, p in zip(image.shape, self.patch_size)]
            d, h, w = starts
            pd, ph, pw = self.patch_size
            image, target = (
                image[d:d+pd, h:h+ph, w:w+pw],
                target[d:d+pd, h:h+ph, w:w+pw],
            )
        return torch.from_numpy(image.copy()).unsqueeze(0),torch.from_numpy(target.copy()).long()

