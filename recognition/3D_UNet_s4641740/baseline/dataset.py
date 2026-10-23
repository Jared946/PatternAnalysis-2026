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

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    valid_labels = np.isin(arr, np.arange(6))
    if not np.all(valid_labels):
        raise ValueError("Segmentation contains labels outside 0-5.")

    res = np.zeros(arr.shape + (6,), dtype=dtype)
    for c in range(6):
        res[..., c] = (arr == c)

    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.
        float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for
        each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive
        memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for
        quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def get_image_names(data_folder):
    """Return sorted paths to all MRI NIfTI volumes."""
    mri_folder = Path(data_folder) / "semantic_MRs_anon"
    return sorted(mri_folder.glob("*.nii.gz"))

def get_label_names(image_names, data_folder):
    """Find the segmentation volume matching each MRI."""
    label_folder = Path(data_folder) / "semantic_labels_anon"

    label_names = []
    for image_path in image_names:
        label_name = image_path.name.replace(
            "_LFOV.nii.gz",
            "_SEMANTIC_LFOV.nii.gz",
        )
        label_names.append(label_folder / label_name)

    return label_names

def get_patient_ids(image_names):
    """Extract unique patient IDs from MRI filenames."""
    return sorted({
        image_path.name.split("_")[1]
        for image_path in image_names
    })

#ensure number of test cases are equal
#e.g. test case could have 6 patient ids, but only 6 folders
#whereas val case could have 6 patient ids, but 30 folders
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
            
            # Load the entire volume into RAM instead of reopening it per slice.
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

# Geometry of the U-Net
POOLING_LEVELS = 4
MIN_INPUT_SIZE = 188
UNET_REMAINDER = 12
UNET_STRIDE = 2 ** POOLING_LEVELS

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
            self.label_dir / image_path.name
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
