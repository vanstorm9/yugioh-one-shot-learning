#!/usr/bin/env python3
"""
Video / webcam Yu-Gi-Oh card recognizer.

Runs the same YOLO + DINOv2 + FAISS pipeline as recognize.py on every frame
of a video file or live webcam stream.

Usage:
  python recognize_video.py --source 0                        # webcam
  python recognize_video.py --source video.mp4                # video file
  python recognize_video.py --source 0 --skip-frames 2        # infer every 3rd frame
  python recognize_video.py --source video.mp4 --save out.mp4
  python recognize_video.py --source 0 --lora-checkpoint savedModels/dino_lora_arcface.pth
"""

import argparse
import time
import cv2

from recognize import CardRecognizer, YOLO_WEIGHTS, INDEX_PATH, META_PATH, TOP_K, YOLO_CONF


def parse_source(source_str):
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main():
    ap = argparse.ArgumentParser(description="Yu-Gi-Oh video/webcam card recognizer")
    ap.add_argument("--source",          default="0",          help="Video file path or webcam index (default: 0)")
    ap.add_argument("--yolo",            default=YOLO_WEIGHTS, help="YOLO weights path")
    ap.add_argument("--index",           default=INDEX_PATH,   help="FAISS index path")
    ap.add_argument("--meta",            default=META_PATH,    help="Metadata JSON path")
    ap.add_argument("--top-k",    type=int, default=TOP_K,     help="Predictions per card")
    ap.add_argument("--yolo-conf", type=float, default=YOLO_CONF, help="YOLO confidence threshold")
    ap.add_argument("--skip-frames", type=int, default=2,
                    help="Run inference every N+1 frames, reuse last result in between (default: 2)")
    ap.add_argument("--min-score",  type=float, default=0.70,
                    help="Min DINOv2 similarity score to display a detection (default: 0.70)")
    ap.add_argument("--save",            default="recognize_output.mp4", help="Save annotated video to this path")
    ap.add_argument("--lora-checkpoint", default=None,         help="Optional LoRA checkpoint path")
    args = ap.parse_args()

    recognizer = CardRecognizer(
        yolo_path       = args.yolo,
        index_path      = args.index,
        meta_path       = args.meta,
        lora_checkpoint = args.lora_checkpoint,
        top_k           = args.top_k,
        yolo_conf       = args.yolo_conf,
    )

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (w, h))

    detections   = []
    frame_idx    = 0
    fps_display  = 0.0
    t_fps        = time.perf_counter()
    fps_counter  = 0

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on every (skip_frames+1)-th frame
        if frame_idx % (args.skip_frames + 1) == 0:
            raw = recognizer.recognize(frame)
            detections = [d for d in raw
                          if d["predictions"] and d["predictions"][0]["score"] >= args.min_score]

        annotated = recognizer.annotate(frame, detections, show_score=True)

        # FPS overlay
        fps_counter += 1
        elapsed = time.perf_counter() - t_fps
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            t_fps = time.perf_counter()

        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Cards: {len(detections)}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2, cv2.LINE_AA)

        if writer:
            writer.write(annotated)

        cv2.imshow("YuGiOh Recognizer", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
