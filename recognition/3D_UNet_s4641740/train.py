"""Train UNet3D on patient-separated HipMRI patches."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import (
    MRI3DVolumeDataset,
    get_image_names,
    get_label_names,
    get_patient_ids,
    select_pairs,
    split_patients,
)
from modules import UNet3D

def dice(logits, target):
    prediction = logits.argmax(1)
    scores = []
    for c in range(1, 6):
        p, t = prediction == c, target == c
        denom = p.sum().item() + t.sum().item()
        if denom:
            scores.append(2 * (p & t).sum().item() / denom)

    return float(np.mean(scores)) if scores else 1.0

def epoch(model, loader, loss_fn, device, optimizer = None):
    train = optimizer is not None
    model.train(train)
    loss = dice_sum = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            value = loss_fn(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none = True)
                value.backward()
                optimizer.step()
        loss += value.item()
        dice_sum += dice(logits.detach(), targets)

    return loss / len(loader), dice_sum / len(loader)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,
            default=Path('/home/groups/comp3710/HipMRI_Study_open'))
    p.add_argument('--output',type=Path,
            default=Path('checkpoints/unet3d_hipmri.pt'))
    p.add_argument('--patch-size',nargs=3,type=int,
            default=(64,128,128),metavar=('D','H','W'))
    p.add_argument('--base-channels',type=int,default=16,
            help='Set 64 for the 2-D channel widths; 16 is practical for 3-D memory.')
    p.add_argument('--epochs',type=int,default=20) #was 100
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available()
            else 'cpu')
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    images = get_image_names(a.data_root)
    if not images:
        raise FileNotFoundError(
            f'No volumes found under {a.data_root / "semantic_MRs"}'
        )
    labels = get_label_names(images, a.data_root)
    patient_ids = get_patient_ids(images)
    train_ids, val_ids, test_ids = split_patients(patient_ids)
    train_pairs = select_pairs(images, labels, train_ids)
    val_pairs = select_pairs(images, labels, val_ids)
    train_ds = MRI3DVolumeDataset(*zip(*train_pairs),
            patch_size = a.patch_size, random_patch = True)
    val_ds = MRI3DVolumeDataset(*zip(*val_pairs), patch_size = a.patch_size)
    device = torch.device(a.device)
    common = dict(batch_size = a.batch_size, num_workers = a.workers,
            pin_memory = device.type == 'cuda')
    train_loader = DataLoader(train_ds, shuffle = True, **common)
    val_loader = DataLoader(val_ds, shuffle = False, **common)
    model = UNet3D(base_channels = a.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = a.lr)
    loss_fn = nn.CrossEntropyLoss()
    a.output.parent.mkdir(parents = True, exist_ok = True)
    best =- float('inf')

    # Reset GPU memory statistics before measuring training.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    training_start = time.perf_counter()
    checkpoint_saved = False

    for e in range(1, a.epochs + 1):
        epoch_start = time.perf_counter()

        # Training and validation
        tl, td = epoch(model, train_loader, loss_fn, device, optimizer)
        vl, vd = epoch(model, val_loader, loss_fn, device)
        print(f'Epoch {e:03d}/{a.epochs}: train loss={tl:.4f}, dice={td:.4f} | val loss={vl:.4f}, dice={vd:.4f}')
        # Save the best checkpoint
        if vd > best:
            best = vd
            torch.save({'epoch': e,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_validation_dice': best,
                'in_channels': 1,
                'num_classes': 6,
                'base_channels': a.base_channels,
                'architecture': 'UNet3D'}, a.output)

            checkpoint_saved = a.output.is_file()

            print(f'Saved best checkpoint to {a.output}')

        # Synchronise CUDA so GPU work is included in the timing
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        epoch_seconds = time.perf_counter() - epoch_start

        print(f'\nEpoch {e:03d}/{a.epochs}')
        print(f'  Training loss:   {tl:.4f}')
        print(f'  Training Dice:   {td:.4f}')
        print(f'  Validation loss: {vl:.4f}')
        print(f'  Validation Dice: {vd:.4f}')
        print(f'  Epoch runtime:   {epoch_seconds:.2f} seconds')

    # Summary
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    total_seconds = time.perf_counter() - training_start

    print('\n========== SUMMARY ==========')
    print(f'Total training runtime: {total_seconds:.2f} seconds')
    print(f'Total training runtime: {total_seconds / 60:.2f} minutes')

    if device.type == 'cuda':
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        )

        print(f'Peak GPU memory allocated: {peak_allocated:.3f} GiB')
        print(f'Peak GPU memory reserved:  {peak_reserved:.3f} GiB')
    else:
        print('GPU memory: Not applicable (CPU execution)')

    print(f'Checkpoint saved successfully: {checkpoint_saved}')
    print(f'Checkpoint path: {a.output}')
    print('=========================================')

if __name__=='__main__':
    main()

