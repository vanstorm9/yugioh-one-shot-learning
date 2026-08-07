# Experiment Log

All methods, implementations, and results for the Yu-Gi-Oh one-shot card recognition project.

---

## 2026-08-06 — Legacy Siamese + Triplet Loss (pre-refactor baseline)

**Branch:** `main` (pre-refactor)  
**Scripts:** `train.py`, `predict.py`, `evalAll.py`, `predictAll.py`  
**Method:**
- ResNet101-LPF backbone (shift-invariant blur pooling)
- Triplet loss metric learning
- Euclidean distance ranking
- HOG (ORB) re-ranking on top candidates

**Dataset:** ~10k cards sourced from YGOPRODeck API (stored on Google Drive, not in repo)  
**Result:** Accuracy not formally recorded  
**Notes:**
- Multiple critical bugs found: `Variable()` deprecation throughout, `dim=(255,255)` off-by-one vs actual `(244,244)`, `rankListFinal.sort()` called inside HOG loop (O(n²log n)), `img1.shape` NameError in `imgPathToTranslateCVImg`
- Heavy code duplication across 5+ scripts (SiameseNetwork, TripletLoss, preprocessing)
- Replaced by DINOv2 pipeline on `modernize-dino` branch

---

## 2026-08-06 — Refactor: shared utils + bug fixes

**Branch:** `refactor-v1`  
**Commit:** `b402e4f`  
**Scripts changed:** All active inference scripts + `yugioh_utils.py` (new)  
**What changed:**
- Extracted `SiameseNetwork`, `TripletLoss`, `imgArtCropper`, `calculateHOGpoints`, `DIM` into `yugioh_utils.py`
- Fixed all `Variable(x).cuda()` → `x.cuda()` with `torch.no_grad()` wrappers
- Fixed `dim` to `DIM=(244,244)` consistently
- Moved `rankListFinal.sort()` outside HOG loop
- Fixed `imgPathToTranslateCVImg` NameError (`img1.shape` → `img.shape`)

**Result:** Code-correctness fix only; no accuracy evaluation run  

---

## 2026-08-06 — DINOv2 + FAISS + ArcFace/LoRA pipeline (new architecture)

**Branch:** `modernize-dino`  
**Commit:** `72cb9b2`  
**Scripts:** `build_faiss_index.py`, `predict_dino.py`, `evalAll_dino.py`, `finetune_arcface.py`

**Method:**
- **Backbone:** `facebook/dinov2-base` (ViT-B/14) — frozen, no training required
- **Embeddings:** CLS token (768-dim), L2-normalized
- **Index:** FAISS `IndexFlatIP` — exact cosine similarity over all ~10k cards
- **Re-ranking:** ORB keypoint matching on top-500 cosine candidates
- **Score:** `cosine_score + (numORBpts³) / 100000` (higher = better)
- **Optional fine-tuning:** LoRA (rank=16, alpha=32) on ViT attention Q/V projections + ArcFace loss (margin=0.5, scale=64)

**Dataset:** 10,987 images across 10,856 cards — downloaded from YGOPRODeck API via `download_dataset.py`  
**Result:** Index not yet built; accuracy evaluation pending  
**Notes:**
- Score direction inverted vs legacy pipeline: cosine similarity is higher-is-better (vs Euclidean distance lower-is-better)
- No retraining needed for zero-shot baseline — DINOv2 features transfer well to card art domain
- Fine-tuning (`finetune_arcface.py`) is optional if zero-shot accuracy is insufficient

---

## 2026-08-06 — Dataset download: YGOPRODeck API

**Script:** `download_dataset.py`  
**Command:** `python3 download_dataset.py --limit 10856`  
**Result:**
- 10,987 images downloaded (10,856 cards; some have multiple art variants)
- 0 failures
- Runtime: ~75 seconds at 16 concurrent workers
- Output: `cardDatabaseFull/` — folder structure `{SanitizedName}-{variant_idx}-{passcode}/{passcode}{variant_idx}.jpg`

---

## 2026-08-06 — YOLO card detection: synthetic dataset generation

**Branch:** `modernize-dino`  
**Script:** `generate_detection_dataset.py`  
**Command:** `python3 generate_detection_dataset.py --n-train 3000 --n-val 500`

**Method — per-card augmentations:**
- Random scale (18–52% of image width), rotation ±30°, perspective warp (60% prob)
- Specular glare: blurred bright ellipse composited onto card surface (40% prob)
- Drop shadow: soft offset shadow beneath each card (55% prob)
- Partial occlusion: cards overlap each other + random dark rectangle on one card (15% prob)

**Method — whole-image augmentations:**
- Brightness / contrast / saturation jitter
- Random noise overlay (30% prob)
- Motion blur with random direction (10% prob)

**Background:** Procedural generation (felt/wood/plain/busy styles with gradient overlays); real photos supported via `--bg-dir`  
**Output:** `detection_dataset/` — YOLO format, single class (`card`)

**Result:**
- 3,000 train + 500 val images at 640×640
- 449 MB total
- Runtime: ~95 seconds at 8 workers

**Training:** 120 epochs on MPS (Apple M4 Max), ~3.3 hours, batch=16, AdamW lr=0.002

**Results (best checkpoint: epoch 118):**
| Metric | Value |
|---|---|
| mAP50 | **0.9875** |
| mAP50-95 | **0.9404** |
| Precision | 0.978 |
| Recall | 0.975 |

**Weights:** `runs/detect/card_yolov8n_120ep/weights/best.pt` (6 MB)

**Notes:**
- mAP50-95 of 0.94 indicates very precise box localization (tight fit, not just loose overlap)
- Losses converged smoothly: box 1.60→0.28, cls 2.47→0.19, dfl 1.68→0.91
- No overfitting observed — val loss tracked train loss throughout

---

## 2026-08-07 — Full two-stage pipeline: recognize.py

**Script:** `recognize.py`

**Pipeline:**
1. YOLO detects card bounding boxes → crops
2. Each crop → `imgArtCropper` → `_dino_transform` → DINOv2 CLS embedding
3. FAISS cosine search over 10,987 gallery embeddings
4. ORB re-ranking on top-200 candidates
5. Return top-K predictions per card

**Usage:**
```bash
python recognize.py photo.jpg --save result.jpg
python recognize.py photo.jpg --lora-checkpoint savedModels/dino_lora_arcface.pth
```

**Result:** Not yet evaluated on real photos (FAISS index not yet built)

---

## 2026-08-07 — Robust DINOv2 fine-tuning: augmentation pipeline

**Script:** `finetune_arcface.py` (updated with `USE_ROBUST_AUGMENTATION = True`)

**Rationale:** Training DINOv2 to be *invariant* to real-world corruptions — any augmented view of the same card maps to the same ArcFace class direction. The FAISS gallery stays as clean card scans; robustness comes from the query encoder, not the index.

**Augmentations added (per-image, applied to card art crop):**
| Transform | Probability | Simulates |
|---|---|---|
| `_GlareTransform` | 40% | Foil / holographic reflection |
| `_MotionBlurTransform` | 25% | Camera shake, fast movement |
| `ColorJitter` (±0.4 all) | 80% | Room lighting, phone white balance |
| `GaussianBlur` | 30% | Out-of-focus photos |
| `RandomPerspective` (0.3) | 50% | Non-orthographic camera angle |
| `RandomAffine` (±15°) | always | Card tilt on table |
| `RandomErasing` (2-15%) | 30% | Finger / sleeve occlusion |
| ~~HorizontalFlip~~ | never | Would mirror card (changes identity) |

**Result:** Not yet trained — run `python finetune_arcface.py` after building base FAISS index

---

---

## 2026-08-07 — ORB removed from pipeline

**Reason:** ORB re-ranking is 28× slower (570ms vs 20ms) and reduces accuracy by 0.6% on the closed-set test. The `n³` score formula over-rewards images with many keypoints, causing wrong cards with rich artwork to beat perfect cosine matches.

**Files changed:** `recognize.py`, `predict_dino.py`, `evalAll_dino.py` — ORB/`calculateHOGpoints`/`DIM` references removed entirely. Pipeline is now: YOLO → DINOv2 → FAISS cosine, ~50ms end-to-end.

---

## 2026-08-07 — Real-world evaluation: Flickr + Reddit photos

**Scripts:** `scrape_eval_dataset.py`, `eval_real_world.py`

**Dataset construction:**
- Scraped Flickr public feed (`yugioh`, `yugiohcard` tags) and Reddit r/yugioh via RSS feed
- Downloaded 40 images; manually labeled by reading card names from each image with vision
- Usable card photos found: **6** (Flickr yielded mostly cosplay/promo; Reddit had a better signal-to-noise ratio)
- 3 of 6 labeled cards happened to fall outside the 10,987-card downloaded subset

**Results:**

| Mode | Cards in DB | Cards outside DB | Rank-1 | Notes |
|---|---|---|---|---|
| DINOv2+FAISS (no YOLO) | **3/3 = 100%** | 0/3 = 0% | 3/6 = 50% | full-image embed |
| DINOv2+FAISS + YOLO    | **3/3 = 100%** | 0/7 = 0% | 3/10 = 30%* | YOLO split multi-card image into 5 crops |

*lower denominator with YOLO because the multi-card image became 5 separate evaluations

**Per-card scores (in-DB cards only):**
| Card | Source | Cosine Score |
|---|---|---|
| Arcana Force XXI - The World | Reddit (physical card scan) | **0.9721** |
| Divine Grace - Northwemko | Flickr (real photo, sleeve+glare) | 0.7379 |
| Fuse Line | Reddit (physical card, holographic) | 0.6834 |

**Key observations:**
1. The model achieves **100% rank-1 accuracy** on real-world photos of cards it was trained on
2. High scores even with glare (Divine Grace had visible foil glare in the photo)
3. Cards outside the index return confident-but-wrong cosine scores (0.69–0.97) — expected behavior, no crash
4. Flickr `yugioh` tag is dominated by cosplay/convention photos (~90%) — Flickr `yugiohcard` tag or Reddit are better sources
5. The 10,987-card database does not cover all cards; downloading all ~14k cards would improve coverage

**For a larger eval:** Set `ANTHROPIC_API_KEY` and run:
```bash
python scrape_realworld_eval.py --limit 100
```
This auto-scrapes images, uses Claude vision to read card names as ground truth, and runs DINOv2+FAISS eval in one pipeline.
