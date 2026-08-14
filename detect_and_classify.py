#!/usr/bin/env python3
"""
Detect all cards in an image with YOLO, classify each with DINOv2+FAISS,
and display the result with blue bounding boxes and card-name labels.

Usage:
  python detect_and_classify.py --image path/to/photo.jpg
  python detect_and_classify.py --image path/to/photo.jpg --top-k 3
  python detect_and_classify.py --image path/to/photo.jpg --conf 0.3
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import cv2

from yugioh_utils import DINOv2Extractor
import faiss

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--image",   required=True, help="Path to input image")
ap.add_argument("--index",   default="./savedModels/dino_index.faiss")
ap.add_argument("--meta",    default="./savedModels/dino_meta.json")
ap.add_argument("--yolo",    default="./runs/detect/card_yolov8n_120ep/weights/best.pt")
ap.add_argument("--conf",      type=float, default=0.20, help="YOLO confidence threshold")
ap.add_argument("--min-score", type=float, default=0.70, help="Min DINOv2 similarity score to show a box")
ap.add_argument("--top-k",     type=int,   default=1,   help="Number of top predictions to print")
ap.add_argument("--lora-checkpoint", default=None)
args = ap.parse_args()

for path, name in [(args.image, "image"), (args.index, "FAISS index"), (args.meta, "metadata")]:
    if not os.path.exists(path):
        print(f"Error: {name} not found: {path}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading FAISS index...")
index = faiss.read_index(args.index)
with open(args.meta) as f:
    meta = json.load(f)
print(f"  {index.ntotal} vectors")

print("Loading DINOv2...")
extractor = DINOv2Extractor(lora_checkpoint=args.lora_checkpoint)
print(f"  Device: {extractor.device}")

print(f"Loading YOLO from {args.yolo}...")
from ultralytics import YOLO
yolo = YOLO(args.yolo)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def folder_to_display(folder_name):
    """BlueEyes-White-Dragon-1-89631139  →  'Blue-Eyes White Dragon'"""
    parts = folder_name.split("-")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = parts[:-2]
    elif len(parts) >= 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)

# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------
img_bgr = cv2.imread(args.image)
if img_bgr is None:
    print(f"Error: cannot read image {args.image}")
    sys.exit(1)

print(f"\nRunning YOLO on {args.image} ...")
results = yolo(img_bgr, conf=args.conf, verbose=False)
boxes = results[0].boxes

if boxes is None or len(boxes) == 0:
    print("No cards detected.")
    cv2.imshow("Detection", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

xyxy    = boxes.xyxy.cpu().numpy()
confs   = boxes.conf.cpu().numpy()
classes = boxes.cls.cpu().numpy().astype(int)
print(f"  {len(xyxy)} detection(s)\n")

# ---------------------------------------------------------------------------
# Classify each crop and annotate
# ---------------------------------------------------------------------------
display = img_bgr.copy()
BLUE  = (255, 80, 0)
FONT  = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.6
THICK = 2

for i, (box, yolo_conf, cls) in enumerate(zip(xyxy, confs, classes)):
    if cls == 1:
        print(f"  Detection {i+1}  yolo={yolo_conf:.2f}  → card back (skipped)")
        continue

    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
    crop = img_bgr[y1:y2, x1:x2]

    try:
        emb = extractor.extract_array(crop)
    except Exception as e:
        print(f"  Card {i}: embed failed — {e}")
        continue

    query = emb.reshape(1, -1).astype("float32")
    scores, indices = index.search(query, k=args.top_k)
    predictions = [(folder_to_display(meta[idx]["card_name"]), float(scores[0][j]))
                   for j, idx in enumerate(indices[0])]

    top_name, top_score = predictions[0]
    if top_score < args.min_score:
        print(f"  Card {i+1}  yolo={yolo_conf:.2f}  → SKIPPED (score {top_score:.4f} < {args.min_score})")
        continue
    print(f"  Card {i+1}  yolo={yolo_conf:.2f}  → {top_name}  ({top_score:.4f})")
    if args.top_k > 1:
        for rank, (name, score) in enumerate(predictions[1:], 2):
            print(f"           top-{rank}: {name}  ({score:.4f})")

    # Bounding box
    cv2.rectangle(display, (x1, y1), (x2, y2), BLUE, THICK)

    # Label background + text
    label = f"{top_name} ({top_score:.2f})"
    (tw, th), baseline = cv2.getTextSize(label, FONT, SCALE, THICK)
    label_y = y1 - 8 if y1 - 8 > th else y2 + th + 8
    cv2.rectangle(display, (x1, label_y - th - baseline), (x1 + tw, label_y + baseline), BLUE, -1)
    cv2.putText(display, label, (x1, label_y), FONT, SCALE, (255, 255, 255), THICK, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------
cv2.imshow("Card Detection & Classification", display)
print("\nPress any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
