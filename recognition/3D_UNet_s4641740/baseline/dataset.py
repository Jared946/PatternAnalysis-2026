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
def split_patients(patient_ids, seed=42):
    """Split unique patient IDs into train, validation and test sets."""
    patient_ids = sorted(set(patient_ids))

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train:n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val:])

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

        # Index every 2D slice without loading all voxel data into RAM.
        self.slice_index = []

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

            depth = image_nii.shape[2]

            for slice_idx in range(depth):
                self.slice_index.append((volume_idx, slice_idx))

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_index[index]

        # Load the corresponding MRI and segmentation volumes.
        image_nii = nib.load(str(self.image_names[volume_idx]))
        label_nii = nib.load(str(self.label_names[volume_idx]))

        # Extract the same axial slice from each.
        image = np.asarray(
            image_nii.dataobj[:, :, slice_idx],
            dtype=np.float32,
        )
        mask = np.asarray(
            label_nii.dataobj[:, :, slice_idx],
            dtype=np.int64,
        )

        if self.normalise:
            image = (image - image.mean()) / (
                image.std() + 1e-8
            )


        # PyTorch Conv2d expects (channels, height, width).
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        mask = to_channels(mask)
        mask = torch.from_numpy(mask.copy()).permute(2, 0, 1).float()

        return image, mask



