"""Predict HipMRI segmentation volumes using ``modules.PaperUNet2D``.

Outputs retain each source volume's shape and affine.  Since the original
U-Net uses valid convolutions, unpredicted border voxels are set to class 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

#from dataset import get_image_names
#from modules import UNet2D
from dataset import get_image_names, paper_unet_size
from modules import PaperUNet2D

def find_images(data_root: Path | None, images: Path | None) -> list[Path]:
    if data_root:
        return get_image_names(data_root)
    if images is None:
        raise ValueError("Provide --data-root or --images.")
    if images.is_file():
        return [images]
    nested = images / "semantic_MRs_anon"
    return sorted((nested if nested.is_dir() else images).glob("*.nii.gz"))


def normalise_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    return (image - image.mean()) / (image.std() + 1e-8)

def predict_volume(
    model: PaperUNet2D,
    volume: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D MRI volume, got {volume.shape}")

    # Normalise the entire volume before extracting slices.
    volume = volume.astype(np.float32, copy=False)
    volume = (
        volume - volume.mean()
    ) / (volume.std() + 1e-8)

    height, width, depth = volume.shape

    crop_height = paper_unet_size(height)
    crop_width = paper_unet_size(width)

    crop_top = (height - crop_height) // 2
    crop_left = (width - crop_width) // 2

    result = np.zeros(volume.shape, dtype=np.uint8)

    with torch.inference_mode():
        for slice_idx in range(depth):
            # Normalise the complete slice, matching the training dataset.
            image = volume[:, :, slice_idx]

            # Apply the same central input crop as the training dataset.
            image = image[
                crop_top:crop_top + crop_height,
                crop_left:crop_left + crop_width,
            ]

            logits = model(
                torch.from_numpy(image[None, None]).to(device)
            )

            label_slice = (
                logits.argmax(dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Place the predicted region back in the original volume.
            output_height, output_width = label_slice.shape

            top = (height - output_height) // 2
            left = (width - output_width) // 2

            result[
                top:top + output_height,
                left:left + output_width,
                slice_idx,
            ] = label_slice

    return result

def output_name(image_path: Path) -> str:
    name = image_path.name
    stem = name[:-7] if name.endswith(".nii.gz") else image_path.stem
    return f"{stem}_segmentation.nii.gz"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path,
            help="Directory containing semantic_MRs_anon.")
    source.add_argument("--images", type=Path,
            help="An MRI .nii.gz file or directory of MRI volumes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device",
            default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = find_images(args.data_root, args.images)
    if not paths:
        raise FileNotFoundError("No MRI .nii.gz volumes found.")
    device = torch.device(args.device)
    checkpoint = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=False)
    model = PaperUNet2D(
            checkpoint.get("in_channels", 1),
            checkpoint.get("num_classes", 6)
            )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in paths:
        nifti = nib.load(str(image_path))
        prediction = predict_volume(model, np.asarray(nifti.dataobj), device)
        output_path = args.output_dir / output_name(image_path)

        header = nifti.header.copy()
        header.set_data_dtype(np.uint8)

        nib.save(
            nib.Nifti1Image(prediction, nifti.affine, header),
            str(output_path),
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

