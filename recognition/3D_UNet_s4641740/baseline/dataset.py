"""
loads 2D Medical Nifti Images for 2D baseline U-Net"
"""

import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path

#make sure to have 6 consistent labels
#even when label does not appear
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

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
