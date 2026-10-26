"""Sliding-window prediction of full HipMRI volumes with UNet3D."""
from __future__ import annotations
import argparse
import time
import csv
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from dataset import get_image_names
from modules import UNet3D

def _starts(length, patch, stride):
    if length <= patch:
        return [0]

    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)

    return values

def predict(model, xyz, patch_size, device):
    volume = np.asarray(xyz, dtype = np.float32).transpose(2, 0, 1)
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    padding = []
    for current, patch in zip(volume.shape, patch_size):
        extra = max(0, patch - current)
        padding.append((extra // 2, extra - extra // 2))
    volume = np.pad(volume, padding, mode = 'edge')
    summed = np.zeros((model.num_classes, *volume.shape), np.float32)
    count = np.zeros(volume.shape, np.float32)
    stride = tuple(n//2 for n in patch_size)
    with torch.inference_mode():
        for d in _starts(volume.shape[0], patch_size[0], stride[0]):
            for h in _starts(volume.shape[1], patch_size[1], stride[1]):
                for w in _starts(volume.shape[2], patch_size[2], stride[2]):
                    pd, ph, pw = patch_size
                    patch = volume[d:d + pd, h:h + ph, w:w + pw]
                    logits = model(torch.from_numpy(patch[None, None]).
                            to(device)).squeeze(0).cpu().numpy()
                    summed[:, d:d+pd, h:h+ph, w:w+pw] += logits
                    count[d:d+pd, h:h+ph, w:w+pw] += 1
    labels = (summed / count[None]).argmax(0).astype(np.uint8)
    crop = tuple(slice(a, labels.shape[i] - b) for i, (a ,b) in enumerate(padding))
    return labels[crop].transpose(1, 2, 0)

def main():
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--data-root',type=Path,
            default=Path('C:/Users/User/SEMESTER 2/COMP3710/project/HipMRI_Study_open'))
    p.add_argument('--checkpoint',type=Path,
            default=Path('checkpoints/unet3d_hipmri.pt'))
    p.add_argument('--output-dir',type=Path,default=Path('inference/'))
    p.add_argument('--patch-size',nargs=3,type=int,
            default=(64,128,128),metavar=('D','H','W'))
    p.add_argument('--device',default='cuda'
            if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if any(n < 16 or n % 16 != 0 for n in a.patch_size):
        p.error("--patch-size dimensions must be multiples of 16 and >= 16")
    paths = get_image_names(a.data_root)
    if not paths:
        raise FileNotFoundError('No MRI volumes found.')
    device = torch.device(a.device)
    ckpt = torch.load(a.checkpoint, map_location = device, weights_only = False)
    model = UNet3D(ckpt.get('in_channels', 1), ckpt.get('num_classes', 6),
            ckpt.get('base_channels', 16))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    a.output_dir.mkdir(parents = True, exist_ok = True)

    metrics = []
    for path in paths:
        nii = nib.load(str(path))

        # Make sure previous CUDA work has finished before timing.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        # Time prediction only, excluding model loading and file saving.
        start_time = time.perf_counter()

        labels = predict(
            model,
            nii.dataobj,
            tuple(a.patch_size),
            device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        inference_seconds = time.perf_counter() - start_time

        # Read peak memory usage for this MRI volume.
        if device.type == "cuda":
            peak_allocated_gib = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
            peak_reserved_gib = (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            )
        else:
            peak_allocated_gib = None
            peak_reserved_gib = None

        # Save prediction as before.
        stem = (
            path.name[:-7]
            if path.name.endswith(".nii.gz")
            else path.stem
        )
        output = a.output_dir / f"{stem}_segmentation_3d.nii.gz"

        nib.save(
            nib.Nifti1Image(labels, nii.affine, nii.header),
            str(output),
        )

        metrics.append({
            "case": stem,
            "inference_seconds": inference_seconds,
            "peak_gpu_allocated_GiB": peak_allocated_gib,
            "peak_gpu_reserved_GiB": peak_reserved_gib,
            "prediction_file": str(output),
        })

        print(f"Saved {output}")
        print(f"  Inference time: {inference_seconds:.2f} seconds")

    # Write one row per MRI volume to a CSV file.
    metrics_path = a.output_dir / "inference_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nInference metrics saved to: {metrics_path.resolve()}")

if __name__=='__main__':
    main()
