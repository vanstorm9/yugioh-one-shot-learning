#!/usr/bin/env python3
"""
Evaluate DINOv2 + FAISS on the scraped real-world image dataset.

Expects:
  eval_real_world/manifest.json   — entries with "label" field populated
  savedModels/dino_index.faiss
  savedModels/dino_meta.json

Usage:
  python eval_real_world.py
  python eval_real_world.py --no-yolo    # skip YOLO, feed whole image directly
  python eval_real_world.py --top-k 5
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from time import time

from yugioh_utils import (
    DINOv2Extractor, imgPathToRGBTensor, imgArtCropper, DIM,
)
import faiss

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--manifest",   default="eval_real_world/manifest.json")
ap.add_argument("--index",      default="./savedModels/dino_index.faiss")
ap.add_argument("--meta",       default="./savedModels/dino_meta.json")
ap.add_argument("--images-dir", default="eval_real_world/images")
ap.add_argument("--no-yolo",    action="store_true",
                help="Skip YOLO detection; embed the whole image as the card")
ap.add_argument("--yolo",       default="./runs/detect/card_yolov8n_120ep/weights/best.pt")
ap.add_argument("--top-k",      type=int, default=5)
ap.add_argument("--lora-checkpoint", default=None)
ap.add_argument("--save-errors",     default="eval_real_world/errors.json",
                help="Write mismatches to this JSON file")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------
with open(args.manifest) as f:
    manifest = json.load(f)

labeled = [e for e in manifest if e.get("label")]
print(f"Manifest: {len(manifest)} total, {len(labeled)} labeled")
if not labeled:
    print("No labeled entries. Run the labeling step first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading FAISS index...")
index = faiss.read_index(args.index)
with open(args.meta) as f:
    meta = json.load(f)
print(f"  {index.ntotal} vectors indexed")

print("Loading DINOv2...")
extractor = DINOv2Extractor(lora_checkpoint=args.lora_checkpoint)
print(f"  Device: {extractor.device}")

yolo_model = None
if not args.no_yolo:
    try:
        from ultralytics import YOLO
        print(f"Loading YOLO from {args.yolo}...")
        yolo_model = YOLO(args.yolo)
    except Exception as e:
        print(f"  YOLO load failed ({e}), falling back to no-YOLO mode")

# ---------------------------------------------------------------------------
# Normalize label for matching
# Manifest labels are card names like "Blue-Eyes White Dragon".
# meta[i]["card_name"] is the folder name like "BlueEyes-White-Dragon-1-89631139"
# ---------------------------------------------------------------------------

def _norm(s):
    """Lowercase, keep only letters/digits, strip separators."""
    import re
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _folder_to_display(folder_name):
    """BlueEyes-White-Dragon-1-89631139 → 'BlueEyes White Dragon'."""
    parts = folder_name.split("-")
    # Last two parts are variant_idx and passcode — strip them
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = parts[:-2]
    elif len(parts) >= 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)

def label_matches(ground_truth_label, predicted_folder):
    """
    Check if a human-written label matches a predicted folder name.
    Supports MULTI labels: "MULTI:Card A|Card B|Card C"
    """
    if ground_truth_label.startswith("MULTI:"):
        candidates = ground_truth_label[6:].split("|")
        return any(label_matches(c, predicted_folder) for c in candidates)

    gt_norm   = _norm(ground_truth_label)
    pred_norm = _norm(_folder_to_display(predicted_folder))
    if gt_norm == pred_norm:
        return True
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    return False


def get_best_crop(img_bgr):
    """Run YOLO and return the highest-confidence card crop, else full image."""
    if yolo_model is None:
        return img_bgr, None
    results = yolo_model(img_bgr, conf=0.20, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return img_bgr, None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs  = results[0].boxes.conf.cpu().numpy()
    best   = int(np.argmax(confs))
    x1, y1, x2, y2 = map(int, boxes[best])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return img_bgr, None
    return img_bgr[y1:y2, x1:x2], float(confs[best])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
correct   = 0
total     = 0
top_k_hit = 0
errors    = []

print(f"\nEvaluating {len(labeled)} images  (top-k={args.top_k})\n")
print(f"{'#':>4}  {'result':8s}  {'ground truth':<35s}  {'top-1 prediction':<35s}  score")
print("-" * 100)

t_start = time()
for entry in labeled:
    img_path = os.path.join(args.images_dir, entry["file"])
    label    = entry["label"].strip()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  Warning: cannot read {img_path}, skipping")
        continue

    is_multi = label.startswith("MULTI:")
    multi_labels = label[6:].split("|") if is_multi else [label]

    # Get card crop(s)
    if is_multi and yolo_model is not None:
        # For multi-card images, evaluate each detected crop separately
        results_yolo = yolo_model(img_bgr, conf=0.20, verbose=False)
        if results_yolo and results_yolo[0].boxes is not None and len(results_yolo[0].boxes) > 0:
            boxes = results_yolo[0].boxes.xyxy.cpu().numpy()
            confs  = results_yolo[0].boxes.conf.cpu().numpy()
            crops_to_eval = []
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                x1=max(0,x1); y1=max(0,y1)
                x2=min(img_bgr.shape[1],x2); y2=min(img_bgr.shape[0],y2)
                if x2-x1 > 20 and y2-y1 > 20:
                    crops_to_eval.append((img_bgr[y1:y2, x1:x2], float(conf)))
        else:
            crops_to_eval = [(img_bgr, None)]
    else:
        crop_bgr, yolo_conf = get_best_crop(img_bgr)
        crops_to_eval = [(crop_bgr, yolo_conf)]

    for crop_bgr, yolo_conf in crops_to_eval:
        # Embed
        try:
            emb = extractor.extract_array(crop_bgr)
        except Exception as e:
            print(f"  Warning: embed failed for {img_path}: {e}")
            continue

        query_vec = emb.reshape(1, -1).astype("float32")
        scores, indices = index.search(query_vec, k=args.top_k)
        scores  = scores[0]
        indices = indices[0]

        predictions = [
            (meta[i]["card_name"], float(scores[j]))
            for j, i in enumerate(indices)
        ]

        top1_name, top1_score = predictions[0]
        rank1_correct = label_matches(label, top1_name)
        topk_correct  = any(label_matches(label, name) for name, _ in predictions)

        total += 1
        if rank1_correct:
            correct   += 1
            top_k_hit += 1
            status = "OK"
        elif topk_correct:
            top_k_hit += 1
            status = f"top{args.top_k}"
        else:
            status = "WRONG"
            errors.append({
                "file":           entry["file"],
                "source":         entry.get("source"),
                "ground_truth":   label,
                "predictions":    [(n, f"{s:.4f}") for n, s in predictions],
                "yolo_conf":      yolo_conf,
            })

        gt_disp   = (multi_labels[0] if is_multi else label)[:33]
        top1_disp = _folder_to_display(top1_name)[:33]
        print(f"{total:4d}  {status:8s}  {gt_disp:<35s}  {top1_disp:<35s}  {top1_score:.4f}")

elapsed = time() - t_start

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"Real-world eval results  ({total} images, {elapsed:.1f}s, {elapsed/max(1,total)*1000:.0f}ms/img)")
print(f"  Rank-1 accuracy : {correct}/{total} = {100*correct/max(1,total):.1f}%")
print(f"  Top-{args.top_k} accuracy : {top_k_hit}/{total} = {100*top_k_hit/max(1,total):.1f}%")
yolo_str = "with YOLO" if yolo_model else "no YOLO (full image)"
print(f"  Mode            : DINOv2 + FAISS  ({yolo_str})")

if errors:
    with open(args.save_errors, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"  Errors saved to : {args.save_errors}")

    print(f"\n  Misclassified ({len(errors)}):")
    for e in errors:
        top1 = e["predictions"][0][0] if e["predictions"] else "?"
        print(f"    {e['file']:<25s}  GT: {e['ground_truth'][:30]:<32s}  → {top1[:30]}")
