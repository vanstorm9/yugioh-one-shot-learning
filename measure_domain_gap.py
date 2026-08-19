#!/usr/bin/env python3
"""
Measure how close the synthetic detection dataset is to real webcam images.

Three complementary metrics:
  1. Low-level stats  — blur, brightness, noise, color per-channel
  2. DINOv2-FID       — Fréchet distance between DINOv2 embedding distributions
  3. Classifier acc   — logistic regression "real vs synthetic" (50% = indistinguishable)

Usage:
  python measure_domain_gap.py
  python measure_domain_gap.py --n 200 --real-dir eval_real_world/images
  python measure_domain_gap.py --synthetic-dir detection_dataset/images/train --n 200
"""

import os
import random
import argparse
import numpy as np
import cv2
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from yugioh_utils import DINOv2Extractor, imgArrayToRGBTensor
import torch
import faiss  # noqa — imported after yugioh_utils to avoid BLAS crash

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--real-dir",       default="eval_real_world/images",
                help="Directory of real webcam card images")
ap.add_argument("--synthetic-dir",  default="detection_dataset/images/train",
                help="Directory of synthetic training images")
ap.add_argument("--n",              type=int, default=86,
                help="Images to sample from each set (default: all real images)")
ap.add_argument("--no-dino",        action="store_true",
                help="Skip DINOv2-FID and classifier (faster, stats only)")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------
def _load_images(directory, n):
    exts = {'.jpg', '.jpeg', '.png'}
    paths = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.splitext(f)[1].lower() in exts]
    random.shuffle(paths)
    paths = paths[:n]
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img)
    return imgs

print(f"Loading images...")
real_imgs = _load_images(args.real_dir, args.n)
syn_imgs  = _load_images(args.synthetic_dir, args.n)
n_real, n_syn = len(real_imgs), len(syn_imgs)
print(f"  Real:      {n_real} images from {args.real_dir}")
print(f"  Synthetic: {n_syn} images from {args.synthetic_dir}")

# ---------------------------------------------------------------------------
# 1. Low-level stats
# ---------------------------------------------------------------------------
def _blur_score(img):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _noise_estimate(img):
    """Estimate sensor noise via high-frequency residual."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return np.std(gray - blur)

def _brightness(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()

def _channel_means(img):
    return img.mean(axis=(0, 1))  # BGR

def _compute_stats(imgs, label):
    blurs      = [_blur_score(i) for i in imgs]
    noises     = [_noise_estimate(i) for i in imgs]
    brightness = [_brightness(i) for i in imgs]
    b_means    = np.mean([_channel_means(i) for i in imgs], axis=0)
    return {
        "label":      label,
        "blur_mean":  np.mean(blurs),
        "blur_std":   np.std(blurs),
        "noise_mean": np.mean(noises),
        "brightness": np.mean(brightness),
        "B_mean":     b_means[0],
        "G_mean":     b_means[1],
        "R_mean":     b_means[2],
    }

print("\nComputing low-level stats...")
real_stats = _compute_stats(real_imgs, "Real webcam")
syn_stats  = _compute_stats(syn_imgs,  "Synthetic  ")

W = 14
print(f"\n{'Metric':<22}  {'Real webcam':>{W}}  {'Synthetic':>{W}}  {'Ratio (real/syn)':>{W}}")
print("-" * 72)
for key in ["blur_mean", "blur_std", "noise_mean", "brightness", "B_mean", "G_mean", "R_mean"]:
    r, s = real_stats[key], syn_stats[key]
    ratio = r / s if s != 0 else float('inf')
    flag = "  ⚠" if abs(ratio - 1.0) > 0.30 else ""
    print(f"  {key:<20}  {r:>{W}.2f}  {s:>{W}.2f}  {ratio:>{W}.3f}{flag}")

print()
print("  blur_mean   : higher = sharper  (webcam blur < 100 is typical)")
print("  noise_mean  : higher = noisier")
print("  brightness  : 0-255 mean pixel value")
print("  B/G/R_mean  : per-channel color balance")
print("  ⚠ = >30% difference between real and synthetic")

# ---------------------------------------------------------------------------
# 2. DINOv2-FID + 3. Classifier
# ---------------------------------------------------------------------------
if args.no_dino:
    print("\nSkipping DINOv2-FID and classifier (--no-dino).")
else:
    print("\nLoading DINOv2 for embedding-based metrics...")
    extractor = DINOv2Extractor()

    def _embed_set(imgs):
        embs = []
        for img in imgs:
            try:
                emb = extractor.extract_array(img)
                embs.append(emb)
            except Exception:
                pass
        return np.stack(embs, axis=0).astype(np.float32)

    print(f"  Embedding {n_real} real images...")
    real_embs = _embed_set(real_imgs)
    print(f"  Embedding {n_syn} synthetic images...")
    syn_embs  = _embed_set(syn_imgs)

    # --- FID ---
    def _fid(embs_a, embs_b):
        mu_a, sigma_a = embs_a.mean(0), np.cov(embs_a, rowvar=False)
        mu_b, sigma_b = embs_b.mean(0), np.cov(embs_b, rowvar=False)
        diff = mu_a - mu_b
        # Sqrt of matrix product — use scipy
        covmean, _ = linalg.sqrtm(sigma_a @ sigma_b, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))

    fid_score = _fid(real_embs, syn_embs)

    # --- Logistic regression classifier ---
    X = np.concatenate([real_embs, syn_embs], axis=0)
    y = np.array([1] * len(real_embs) + [0] * len(syn_embs))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    clf_acc = cv_scores.mean()

    print(f"\n{'─'*50}")
    print(f"  DINOv2-FID score   : {fid_score:.1f}")
    print(f"    (lower = distributions more similar; <50 is good, >200 is large gap)")
    print()
    print(f"  Classifier accuracy: {clf_acc*100:.1f}%  (±{cv_scores.std()*100:.1f}%)")
    print(f"    (50% = indistinguishable; 90%+ = large domain gap)")
    print(f"{'─'*50}")

    # Interpretation
    print()
    if fid_score < 50 and clf_acc < 0.65:
        print("  ✓ Distributions are close — augmentation is working well.")
    elif fid_score < 150 and clf_acc < 0.80:
        print("  ~ Moderate gap — augmentation helps but room to improve.")
    else:
        print("  ✗ Large domain gap — consider more aggressive augmentation")
        print("    or collecting more real webcam images for LoRA fine-tuning.")
