#!/usr/bin/env python3
"""
Full two-stage card recognition pipeline.

  Stage 1 — YOLO detects every card bounding box in the photo.
  Stage 2 — For each crop: DINOv2 embedding → FAISS cosine search → top-K card names.

Usage:
  python recognize.py photo.jpg
  python recognize.py photo.jpg --top-k 3 --save result.jpg
  python recognize.py photo.jpg --lora-checkpoint savedModels/dino_lora_arcface.pth
"""

import os
import json
import argparse
import numpy as np
import cv2

from yugioh_utils import DINOv2Extractor, imgArrayToRGBTensor
import faiss

# ---------------------------------------------------------------------------
# Defaults — override via CLI or CardRecognizer constructor
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = "./runs/detect/card_yolov8n_120ep/weights/best.pt"
INDEX_PATH   = "./savedModels/dino_index.faiss"
META_PATH    = "./savedModels/dino_meta.json"
YOLO_CONF    = 0.25   # minimum YOLO detection confidence
TOP_K        = 3      # top predictions to return per card


# ---------------------------------------------------------------------------
# Main recognizer class
# ---------------------------------------------------------------------------

class CardRecognizer:
    """
    Loads all three models once and exposes a `recognize(image)` method.

    Parameters
    ----------
    yolo_path        : path to YOLOv8 best.pt
    index_path       : path to FAISS index
    meta_path        : path to dino_meta.json
    device           : torch device string (auto-detected if None)
    lora_checkpoint  : optional LoRA fine-tune checkpoint from finetune_arcface.py
    top_k            : predictions to return per detected card
    yolo_conf        : YOLO detection confidence threshold
    """

    def __init__(
        self,
        yolo_path       = YOLO_WEIGHTS,
        index_path      = INDEX_PATH,
        meta_path       = META_PATH,
        device          = None,
        lora_checkpoint = None,
        top_k           = TOP_K,
        yolo_conf       = YOLO_CONF,
    ):
        from ultralytics import YOLO

        print("Loading YOLO detector...")
        self.detector = YOLO(yolo_path)

        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.meta = json.load(f)
        print(f"  Index ready: {self.index.ntotal} vectors")

        print("Loading DINOv2 extractor...")
        self.extractor = DINOv2Extractor(device=device, lora_checkpoint=lora_checkpoint)
        print(f"  Device: {self.extractor.device}")

        self.top_k     = top_k
        self.yolo_conf = yolo_conf

    # ------------------------------------------------------------------

    def recognize(self, image):
        """
        Run detection + recognition on one image.

        Parameters
        ----------
        image : str (path) or numpy BGR array

        Returns
        -------
        list of dicts, one per detected card:
          {
            "box"        : [x1, y1, x2, y2]   (pixel coords in original image),
            "yolo_conf"  : float,
            "predictions": [
                {"rank": 0, "card_name": str, "score": float, "path": str},
                ...  (top_k entries)
            ],
          }
        """
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img_bgr = image

        # ---- Stage 1: YOLO detect ----------------------------------------
        yolo_results = self.detector(img_bgr, conf=self.yolo_conf, verbose=False)
        detections   = []

        if not yolo_results or yolo_results[0].boxes is None:
            return []

        boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
        confs      = yolo_results[0].boxes.conf.cpu().numpy()

        # ---- Stage 2: per-card recognition --------------------------------
        for box_xyxy, yolo_conf in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = map(int, box_xyxy)
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2);  y2 = min(img_bgr.shape[0], y2)

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            card_crop_bgr = img_bgr[y1:y2, x1:x2]

            # DINOv2 embedding
            try:
                emb = self.extractor.extract_array(card_crop_bgr)
            except Exception as e:
                print(f"  Warning: embedding failed for box {box_xyxy}: {e}")
                continue

            query_vec = emb.reshape(1, -1).astype("float32")
            scores, indices = self.index.search(query_vec, k=self.index.ntotal)
            scores  = scores[0]
            indices = indices[0]

            rank_list = [
                (self.meta[i]["card_name"], float(scores[j]), self.meta[i]["abs_path"])
                for j, i in enumerate(indices)
            ]

            predictions = [
                {
                    "rank":      rank,
                    "card_name": name,
                    "score":     float(score),
                    "path":      path,
                }
                for rank, (name, score, path) in enumerate(rank_list[:self.top_k])
            ]

            detections.append({
                "box":        [x1, y1, x2, y2],
                "yolo_conf":  float(yolo_conf),
                "predictions": predictions,
            })

        return detections

    # ------------------------------------------------------------------

    def annotate(self, image, detections, show_score=True):
        """
        Draw bounding boxes + top-1 card name onto a copy of the image.

        Parameters
        ----------
        image      : str (path) or BGR numpy array
        detections : output of recognize()

        Returns
        -------
        Annotated BGR numpy array.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            top1 = det["predictions"][0] if det["predictions"] else None

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 2)

            if top1:
                label = top1["card_name"].replace("-", " ")
                if show_score:
                    label += f"  {top1['score']:.3f}"
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 80), -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        return img


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Yu-Gi-Oh card recognizer")
    ap.add_argument("image",              help="Input image path")
    ap.add_argument("--yolo",             default=YOLO_WEIGHTS, help="YOLO weights path")
    ap.add_argument("--index",            default=INDEX_PATH,   help="FAISS index path")
    ap.add_argument("--meta",             default=META_PATH,    help="Metadata JSON path")
    ap.add_argument("--top-k",     type=int,   default=TOP_K,   help="Predictions per card")
    ap.add_argument("--save",             default=None,         help="Save annotated image here")
    ap.add_argument("--lora-checkpoint",  default=None,
                    help="LoRA fine-tune checkpoint (savedModels/dino_lora_arcface.pth)")
    args = ap.parse_args()

    recognizer = CardRecognizer(
        yolo_path       = args.yolo,
        index_path      = args.index,
        meta_path       = args.meta,
        lora_checkpoint = args.lora_checkpoint,
        top_k           = args.top_k,
    )

    print(f"\nRecognizing: {args.image}")
    detections = recognizer.recognize(args.image)

    if not detections:
        print("No cards detected.")
        return

    print(f"\nDetected {len(detections)} card(s):\n")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        print(f"  Card {i+1}  box=({x1},{y1})→({x2},{y2})  yolo_conf={det['yolo_conf']:.3f}")
        for pred in det["predictions"]:
            print(f"    #{pred['rank']+1}  {pred['card_name']:<45s}  score={pred['score']:.4f}")
        print()

    if args.save:
        annotated = recognizer.annotate(args.image, detections)
        cv2.imwrite(args.save, annotated)
        print(f"Annotated image saved to {args.save}")


if __name__ == "__main__":
    main()
