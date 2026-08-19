#!/usr/bin/env python3
"""
Generate a synthetic YOLO detection dataset for Yu-Gi-Oh card detection.

Each composite image places 1-4 random card scans onto a background with:

  Per-card  : random scale, rotation (±30°), mild perspective warp,
              specular glare highlight, drop shadow
  Occlusion : cards overlap each other; random dark-rectangle occluder (15%)
  Whole-img : brightness/contrast/saturation jitter, optional motion blur,
              random noise overlay

Output structure (YOLO format, class 0 = "card", class 1 = "card_back"):
  detection_dataset/
    images/train/*.jpg   images/val/*.jpg
    labels/train/*.txt   labels/val/*.txt
    card.yaml

Usage:
  python generate_detection_dataset.py
  python generate_detection_dataset.py --n-train 5000 --n-val 800
  python generate_detection_dataset.py --bg-dir ./backgrounds   # use real bg photos
  python generate_detection_dataset.py --card-back-dir ./card_backs  # adds card_back class
"""

import os
import sys
import random
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
IMG_SIZE         = 640
N_TRAIN          = 3000
N_VAL            = 500
CARDS_MIN        = 1
CARDS_MAX        = 4
CARD_SCALE_MIN   = 0.18    # card width as fraction of image width
CARD_SCALE_MAX   = 0.52
ROTATION_MAX     = 30      # degrees
GLARE_PROB       = 0.40
PERSPECTIVE_PROB = 0.60
SHADOW_PROB      = 0.55
OCCLUDER_PROB    = 0.15
MOTION_BLUR_PROB = 0.10
DEFOCUS_PROB     = 0.45
NOISE_PROB       = 0.30
JPEG_PROB        = 0.50
VIGNETTE_PROB    = 0.35
HAND_OCCLUDE_PROB = 0.25
COLOR_TEMP_PROB  = 0.40
CARD_BACK_PROB   = 0.30  # fraction of placed cards that use a card back (class 1)
MIN_VISIBLE_FRAC = 0.25    # skip label if < 25% of card is visible
WORKERS          = 8


# ---------------------------------------------------------------------------
# Background generation (procedural — no external images required)
# ---------------------------------------------------------------------------

def _make_background(w, h):
    """Return an (H, W, 3) BGR uint8 background."""
    style = random.choice(['felt', 'felt', 'wood', 'plain', 'plain', 'busy'])

    if style == 'felt':
        palette = [
            (34, 139, 34),   # green felt
            (0, 90, 160),    # blue playmat
            (150, 25, 25),   # red felt
            (30, 30, 30),    # dark mat
            (90, 60, 35),    # brown table cloth
        ]
        r, g, b = random.choice(palette)
        img = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
        noise = np.random.randint(-28, 28, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)

    elif style == 'wood':
        brt = random.randint(80, 155)
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[:, :, 0] = brt * 0.35    # B
        img[:, :, 1] = brt * 0.55    # G
        img[:, :, 2] = brt           # R  (warm tone)
        for _ in range(random.randint(30, 80)):
            y0 = random.randint(0, h)
            t  = random.randint(1, 8)
            img[y0:y0 + t] *= random.uniform(0.55, 0.95)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)

    elif style == 'plain':
        r = random.randint(110, 230)
        g = random.randint(110, 230)
        b = random.randint(110, 230)
        img = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
        noise = np.random.randint(-12, 12, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    else:  # busy
        img = np.random.randint(40, 200, (h, w, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (71, 71), 0)

    # Soft directional gradient for depth variation
    if random.random() < 0.5:
        grad = np.linspace(0.82, 1.0, w, dtype=np.float32)
        if random.random() < 0.5:
            grad = grad[::-1]
        img = np.clip(img.astype(np.float32) * grad[np.newaxis, :, np.newaxis],
                      0, 255).astype(np.uint8)

    return img   # BGR


# ---------------------------------------------------------------------------
# Per-card augmentations
# ---------------------------------------------------------------------------

def _add_glare(card_rgb: np.ndarray) -> np.ndarray:
    """Overlay a soft specular highlight. Returns RGB uint8."""
    h, w = card_rgb.shape[:2]
    glare = np.zeros((h, w), dtype=np.float32)
    cx  = random.randint(w // 5, 4 * w // 5)
    cy  = random.randint(h // 5, 4 * h // 5)
    ax  = random.randint(w // 8, w // 2)
    ay  = random.randint(h // 8, h // 2)
    ang = random.randint(0, 180)
    cv2.ellipse(glare, (cx, cy), (ax, ay), ang, 0, 360, 1.0, -1)

    sigma = random.randint(20, 60)
    glare = cv2.GaussianBlur(glare, (sigma * 6 + 1, sigma * 6 + 1), sigma)
    glare *= random.uniform(0.30, 0.75)

    out = card_rgb.astype(np.float32) / 255.0
    out = np.clip(out + glare[:, :, np.newaxis], 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def _add_perspective(rgba_np: np.ndarray, max_skew: float = 0.08) -> np.ndarray:
    """Apply mild random perspective warp (RGBA in → RGBA out)."""
    h, w = rgba_np.shape[:2]
    s = max_skew
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * random.uniform(0,   s),   h * random.uniform(0,   s)],
        [w * random.uniform(1-s, 1),   h * random.uniform(0,   s)],
        [w * random.uniform(1-s, 1),   h * random.uniform(1-s, 1)],
        [w * random.uniform(0,   s),   h * random.uniform(1-s, 1)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgba_np, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


def _drop_shadow(bg_pil: Image.Image, x, y, rw, rh) -> Image.Image:
    """Paint a blurred drop shadow beneath the card AABB."""
    offset = max(6, min(rw, rh) // 18)
    shadow = Image.new("RGBA", bg_pil.size, (0, 0, 0, 0))
    draw   = ImageDraw.Draw(shadow)
    draw.rectangle(
        [x + offset, y + offset,
         x + rw + offset, y + rh + offset],
        fill=(0, 0, 0, 110)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=offset))
    composite = Image.alpha_composite(bg_pil.convert("RGBA"), shadow)
    return composite.convert("RGB")


def _add_defocus(img_np: np.ndarray) -> np.ndarray:
    """Gaussian defocus blur to simulate out-of-focus webcam capture."""
    sigma = random.uniform(1.0, 5.0)
    k = int(sigma * 4) * 2 + 1
    return cv2.GaussianBlur(img_np, (k, k), sigma)


def _add_vignette(img_np: np.ndarray) -> np.ndarray:
    """Darken corners to simulate webcam lens falloff."""
    h, w = img_np.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    strength = random.uniform(0.3, 0.7)
    mask = np.clip(1.0 - dist * strength, 0.2, 1.0).astype(np.float32)
    return np.clip(img_np.astype(np.float32) * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)


def _add_jpeg_artifacts(img_np: np.ndarray) -> np.ndarray:
    """Re-encode at low JPEG quality to simulate webcam compression."""
    quality = random.randint(40, 85)
    _, buf = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _add_hand_occluder(bg_pil: Image.Image, card_box) -> Image.Image:
    """Draw a skin-toned hand/finger strip along one edge of the card."""
    x1, y1, x2, y2 = card_box
    card_w, card_h = x2 - x1, y2 - y1
    draw = ImageDraw.Draw(bg_pil)
    # Skin tones: range from pale to darker brown
    r = random.randint(150, 220)
    g = random.randint(100, 170)
    b = random.randint(70, 130)
    edge = random.choice(['bottom', 'left', 'right'])
    thickness = int(card_h * random.uniform(0.08, 0.22))
    if edge == 'bottom':
        draw.rectangle([x1, y2 - thickness, x2, y2], fill=(r, g, b))
    elif edge == 'left':
        draw.rectangle([x1, y1, x1 + thickness, y2], fill=(r, g, b))
    else:
        draw.rectangle([x2 - thickness, y1, x2, y2], fill=(r, g, b))
    return bg_pil


def _add_color_temp(img_np: np.ndarray) -> np.ndarray:
    """Shift white balance warm or cool to simulate room lighting."""
    img = img_np.astype(np.float32)
    if random.random() < 0.5:  # warm
        img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(1.05, 1.20), 0, 255)  # R (BGR)
        img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(0.80, 0.95), 0, 255)  # B
    else:  # cool
        img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(1.05, 1.20), 0, 255)  # B
        img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(0.80, 0.95), 0, 255)  # R
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Core image generator
# ---------------------------------------------------------------------------

def _generate_one(card_paths, img_w, img_h, bg_paths=None, card_back_paths=None):
    """
    Build one synthetic image.
    Returns (bgr_np: ndarray, labels: list[str]).
    Class 0 = card front, class 1 = card back.
    """
    # ---- Background ----
    if bg_paths and random.random() < 0.5:
        try:
            bg_pil = Image.open(random.choice(bg_paths)).convert("RGB")
            bg_pil = bg_pil.resize((img_w, img_h), Image.LANCZOS)
        except Exception:
            bg_pil = Image.fromarray(
                cv2.cvtColor(_make_background(img_w, img_h), cv2.COLOR_BGR2RGB))
    else:
        bg_pil = Image.fromarray(
            cv2.cvtColor(_make_background(img_w, img_h), cv2.COLOR_BGR2RGB))

    n_cards = random.randint(CARDS_MIN, CARDS_MAX)
    labels  = []
    placed  = []   # (x1,y1,x2,y2) of each placed card for occluder logic

    for _ in range(n_cards):
        use_back = (card_back_paths and random.random() < CARD_BACK_PROB)
        pool = card_back_paths if use_back else card_paths
        card_class = 1 if use_back else 0
        try:
            card_pil = Image.open(random.choice(pool)).convert("RGB")
        except Exception:
            continue

        # Scale
        card_w = int(img_w * random.uniform(CARD_SCALE_MIN, CARD_SCALE_MAX))
        card_h = int(card_w * card_pil.height / card_pil.width)
        card_pil = card_pil.resize((card_w, card_h), Image.LANCZOS)

        # Glare (on RGB before alpha conversion)
        if random.random() < GLARE_PROB:
            card_pil = Image.fromarray(_add_glare(np.array(card_pil)))

        # RGBA with fully opaque alpha
        card_rgba = card_pil.convert("RGBA")
        card_rgba.putalpha(255)

        # Perspective warp (preserves alpha)
        if random.random() < PERSPECTIVE_PROB:
            card_rgba = Image.fromarray(
                _add_perspective(np.array(card_rgba)), "RGBA")

        # Rotation — expand=True grows canvas; corners become transparent
        angle       = random.uniform(-ROTATION_MAX, ROTATION_MAX)
        card_rotated = card_rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
        rw, rh      = card_rotated.size

        # Random placement (allow up to 40% off-screen on any edge)
        margin = 0.40
        x = random.randint(int(-rw * margin), int(img_w - rw * (1 - margin)))
        y = random.randint(int(-rh * margin), int(img_h - rh * (1 - margin)))

        # Visible bounding box (clipped)
        vx1 = max(0, x)
        vy1 = max(0, y)
        vx2 = min(img_w, x + rw)
        vy2 = min(img_h, y + rh)

        if (vx2 - vx1) * (vy2 - vy1) < rw * rh * MIN_VISIBLE_FRAC:
            continue

        # Drop shadow (drawn under the card)
        if random.random() < SHADOW_PROB:
            bg_pil = _drop_shadow(bg_pil, x, y, rw, rh)

        # Composite card onto background
        bg_pil.paste(card_rotated, (x, y), card_rotated)

        # YOLO label: class cx cy w h (normalised)
        cx = (vx1 + vx2) / 2 / img_w
        cy = (vy1 + vy2) / 2 / img_h
        bw = (vx2 - vx1) / img_w
        bh = (vy2 - vy1) / img_h
        labels.append(f"{card_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        placed.append((vx1, vy1, vx2, vy2))

    # ---- Random dark occluder ----
    if placed and random.random() < OCCLUDER_PROB:
        bx1, by1, bx2, by2 = random.choice(placed)
        bw_b = bx2 - bx1
        bh_b = by2 - by1
        ox1 = int(bx1 + bw_b * random.uniform(0.0, 0.5))
        oy1 = int(by1 + bh_b * random.uniform(0.0, 0.5))
        ox2 = int(ox1 + bw_b * random.uniform(0.15, 0.55))
        oy2 = int(oy1 + bh_b * random.uniform(0.15, 0.55))
        draw = ImageDraw.Draw(bg_pil)
        fill = tuple(random.randint(0, 50) for _ in range(3))
        draw.rectangle([ox1, oy1, ox2, oy2], fill=fill)

    # ---- Skin-toned hand occluder ----
    if placed and random.random() < HAND_OCCLUDE_PROB:
        bg_pil = _add_hand_occluder(bg_pil, random.choice(placed))

    # ---- Whole-image augmentations ----
    bg_pil = ImageEnhance.Brightness(bg_pil).enhance(random.uniform(0.55, 1.45))
    bg_pil = ImageEnhance.Contrast(bg_pil).enhance(random.uniform(0.65, 1.35))
    bg_pil = ImageEnhance.Color(bg_pil).enhance(random.uniform(0.55, 1.45))

    bg_np = np.array(bg_pil)  # RGB uint8

    if random.random() < NOISE_PROB:
        noise = np.random.randint(-22, 22, bg_np.shape, dtype=np.int16)
        bg_np = np.clip(bg_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < MOTION_BLUR_PROB:
        ksize = random.choice([5, 7, 9, 11])
        K     = np.zeros((ksize, ksize), dtype=np.float32)
        K[ksize // 2, :] = 1.0 / ksize
        rot_M = cv2.getRotationMatrix2D(
            (ksize // 2, ksize // 2), random.uniform(0, 180), 1)
        K     = cv2.warpAffine(K, rot_M, (ksize, ksize))
        bg_np = cv2.filter2D(bg_np, -1, K)

    # Convert to BGR for cv2 operations below
    bgr = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)

    if random.random() < DEFOCUS_PROB:
        bgr = _add_defocus(bgr)

    if random.random() < JPEG_PROB:
        bgr = _add_jpeg_artifacts(bgr)

    if random.random() < VIGNETTE_PROB:
        bgr = _add_vignette(bgr)

    if random.random() < COLOR_TEMP_PROB:
        bgr = _add_color_temp(bgr)

    return bgr, labels


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------

def _write_yaml(out_dir, has_card_backs=False):
    path = os.path.join(out_dir, "card.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n\n")
        if has_card_backs:
            f.write("nc: 2\n")
            f.write("names: ['card', 'card_back']\n")
        else:
            f.write("nc: 1\n")
            f.write("names: ['card']\n")
    return path


def _load_card_paths(card_db):
    paths = []
    for root, _, files in os.walk(card_db):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(root, f))
    return paths


def generate_split(split, n, card_paths, out_dir, img_size, bg_paths, workers,
                   card_back_paths=None):
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    def _worker(idx):
        img_np, labels = _generate_one(card_paths, img_size, img_size,
                                       bg_paths, card_back_paths)
        stem = f"{split}_{idx:05d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img_np,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(labels))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, i) for i in range(n)]
        done = 0
        for fut in futures:
            fut.result()
            done += 1
            if done % 200 == 0 or done == n:
                print(f"  [{split}] {done}/{n}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic Yu-Gi-Oh YOLO detection dataset")
    ap.add_argument("--n-train",  type=int, default=N_TRAIN)
    ap.add_argument("--n-val",    type=int, default=N_VAL)
    ap.add_argument("--out",      type=str, default="./detection_dataset")
    ap.add_argument("--card-db",  type=str, default="./cardDatabaseFull")
    ap.add_argument("--bg-dir",        type=str, default=None,
                    help="Optional folder of real background photos")
    ap.add_argument("--card-back-dir", type=str, default=None,
                    help="Folder of card back images (adds card_back class 1)")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers",  type=int, default=WORKERS)
    args = ap.parse_args()

    if not os.path.exists(args.card_db):
        print(f"ERROR: card database not found at {args.card_db}")
        sys.exit(1)

    print("Collecting card image paths...")
    card_paths = _load_card_paths(args.card_db)
    if not card_paths:
        print("ERROR: no images found in card database.")
        sys.exit(1)
    print(f"  {len(card_paths)} card front images found")

    bg_paths = None
    if args.bg_dir and os.path.isdir(args.bg_dir):
        bg_paths = [
            os.path.join(args.bg_dir, f)
            for f in os.listdir(args.bg_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"  {len(bg_paths)} background images found in {args.bg_dir}")
    else:
        print("  Using procedural backgrounds (pass --bg-dir for real photos)")

    card_back_paths = None
    if args.card_back_dir and os.path.isdir(args.card_back_dir):
        card_back_paths = [
            os.path.join(args.card_back_dir, f)
            for f in os.listdir(args.card_back_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if card_back_paths:
            print(f"  {len(card_back_paths)} card back images found — enabling card_back class")
        else:
            print(f"  WARNING: --card-back-dir specified but no images found, skipping")
            card_back_paths = None
    else:
        print("  No card back dir — training single-class (card only)")

    os.makedirs(args.out, exist_ok=True)

    print(f"\nGenerating {args.n_train} train + {args.n_val} val images "
          f"at {args.img_size}x{args.img_size}...")

    generate_split("train", args.n_train, card_paths, args.out,
                   args.img_size, bg_paths, args.workers, card_back_paths)
    generate_split("val",   args.n_val,   card_paths, args.out,
                   args.img_size, bg_paths, args.workers, card_back_paths)

    yaml_path = _write_yaml(args.out, has_card_backs=bool(card_back_paths))

    print(f"\nDone.  Dataset written to {os.path.abspath(args.out)}")
    print(f"YAML:  {yaml_path}")
    print()
    print("Next — train YOLOv8:")
    print(f"  pip install ultralytics")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=120 imgsz={args.img_size}")


if __name__ == "__main__":
    main()
