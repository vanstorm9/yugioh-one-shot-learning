#!/usr/bin/env python3
"""
Scrape real-world Yu-Gi-Oh card photos from Reddit, Flickr, eBay, and Bing
Image Search for ground-truth evaluation.

Ground truth is determined by reading the card name printed at the top of each
card using Claude's vision API (claude-sonnet-5).  The DINOv2+FAISS pipeline
then predicts the card name; we compare the two.

Output:
  eval_realworld/
    images/           — downloaded photos (jpg/png)
    results.json      — [{image_path, gt_name, pred_name, pred_score, correct}, ...]
    summary.txt       — accuracy + per-source breakdown

Usage:
  python scrape_realworld_eval.py
  python scrape_realworld_eval.py --limit 50 --sources reddit bing
  python scrape_realworld_eval.py --index savedModels/dino_index.faiss
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--limit',   type=int, default=100,
                help='Max images to download across all sources')
ap.add_argument('--sources', nargs='+',
                default=['reddit', 'flickr', 'ebay', 'bing'],
                choices=['reddit', 'flickr', 'ebay', 'bing'])
ap.add_argument('--out',     default='./eval_realworld')
ap.add_argument('--index',   default='./savedModels/dino_index.faiss')
ap.add_argument('--meta',    default='./savedModels/dino_meta.json')
ap.add_argument('--lora-checkpoint', default=None)
ap.add_argument('--skip-scrape', action='store_true',
                help='Skip scraping — use images already in --out/images/')
ap.add_argument('--skip-eval',   action='store_true',
                help='Skip DINOv2 eval — only label ground truth')
ap.add_argument('--yolo',        default='./runs/detect/card_yolov8n_120ep/weights/best.pt',
                help='YOLO weights used to gate images without a visible card')
ap.add_argument('--yolo-conf',   type=float, default=0.25,
                help='Minimum YOLO confidence to accept an image as containing a card')
args = ap.parse_args()

OUT_DIR    = Path(args.out)
IMG_DIR    = OUT_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0.0.0 Safari/537.36'),
    'Accept-Language': 'en-US,en;q=0.9',
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get(url, params=None, timeout=15, retries=3):
    if params:
        url = url + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _download_image(url, dest_path, timeout=20):
    """Download image URL to dest_path. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 2000:   # skip tiny placeholder images
            return False
        with open(dest_path, 'wb') as f:
            f.write(data)
        return True
    except Exception:
        return False


def _url_to_filename(url, prefix=''):
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    ext = url.split('?')[0].rsplit('.', 1)[-1].lower()
    if ext not in ('jpg', 'jpeg', 'png', 'webp'):
        ext = 'jpg'
    return f'{prefix}{h}.{ext}'


# ---------------------------------------------------------------------------
# Source scrapers
# ---------------------------------------------------------------------------

def scrape_reddit(max_images=30):
    """Pull card photos from r/yugioh new/hot posts via Reddit JSON API."""
    collected = []
    after = ''
    while len(collected) < max_images:
        params = {'limit': 100, 't': 'month'}
        if after:
            params['after'] = after
        try:
            raw = _get('https://www.reddit.com/r/yugioh/search.json',
                       params={'q': 'card photo', 'sort': 'new', 'limit': 100,
                               't': 'month', 'type': 'link', **({'after': after} if after else {})})
        except Exception as e:
            print(f'  Reddit error: {e}')
            break

        data = json.loads(raw)
        posts = data.get('data', {}).get('children', [])
        after = data.get('data', {}).get('after', '')

        for post in posts:
            pd = post.get('data', {})
            url = pd.get('url', '')
            # Only direct image links
            if not any(url.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
                # Reddit hosted images via i.redd.it
                preview = pd.get('preview', {})
                images  = preview.get('images', [])
                if images:
                    url = images[0].get('source', {}).get('url', '').replace('&amp;', '&')
                else:
                    continue
            if not url:
                continue
            fname = _url_to_filename(url, 'reddit_')
            dest  = IMG_DIR / fname
            if not dest.exists():
                ok = _download_image(url, dest)
                if not ok:
                    continue
            collected.append({'source': 'reddit', 'url': url, 'image_path': str(dest)})
            if len(collected) >= max_images:
                break

        if not after:
            break
        time.sleep(1)

    print(f'  Reddit: {len(collected)} images')
    return collected


def scrape_flickr(max_images=30):
    """Search Flickr for public yugioh card photos (no API key needed via feeds)."""
    collected = []
    try:
        raw = _get('https://api.flickr.com/services/feeds/photos_public.gne',
                   params={'tags': 'yugioh,card', 'format': 'json', 'nojsoncallback': 1})
        # strip jsonFlickrFeed( wrapper if present
        text = raw.decode('utf-8').strip()
        if text.startswith('jsonFlickrFeed('):
            text = text[len('jsonFlickrFeed('):-1]
        data = json.loads(text)
        for item in data.get('items', []):
            media = item.get('media', {})
            url   = media.get('m', '')
            if not url:
                continue
            # Get larger size by replacing _m with _b
            url_large = url.replace('_m.', '_b.')
            fname = _url_to_filename(url_large, 'flickr_')
            dest  = IMG_DIR / fname
            if not dest.exists():
                ok = _download_image(url_large, dest)
                if not ok:
                    ok = _download_image(url, dest)
                if not ok:
                    continue
            collected.append({'source': 'flickr', 'url': url_large, 'image_path': str(dest)})
            if len(collected) >= max_images:
                break
    except Exception as e:
        print(f'  Flickr error: {e}')

    print(f'  Flickr: {len(collected)} images')
    return collected


def scrape_ebay(max_images=30):
    """
    Scrape eBay search results for Yu-Gi-Oh card listings.
    Extracts image URLs from the search results HTML.
    """
    collected = []
    query = urllib.parse.quote('yugioh card single')
    url   = f'https://www.ebay.com/sch/i.html?_nkw={query}&_sacat=0&LH_BIN=1&_ipg=60'
    try:
        raw  = _get(url).decode('utf-8', errors='replace')
        # eBay puts listing images in <img> tags with s-item__image-img class
        # Extract from JSON-like data-src or src attributes
        patterns = [
            r'<img[^>]+class="[^"]*s-item__image-img[^"]*"[^>]+src="([^"]+)"',
            r'"imageUrl":"(https://i\.ebayimg\.com/[^"]+)"',
            r'<img[^>]+src="(https://i\.ebayimg\.com/[^"]+)"',
        ]
        urls = []
        for pat in patterns:
            urls += re.findall(pat, raw)

        seen = set()
        for img_url in urls:
            img_url = img_url.replace('&amp;', '&')
            # Get larger eBay image by replacing size suffix
            img_url = re.sub(r'/s-l\d+\.', '/s-l640.', img_url)
            if img_url in seen:
                continue
            seen.add(img_url)
            fname = _url_to_filename(img_url, 'ebay_')
            dest  = IMG_DIR / fname
            if not dest.exists():
                ok = _download_image(img_url, dest)
                if not ok:
                    continue
            collected.append({'source': 'ebay', 'url': img_url, 'image_path': str(dest)})
            if len(collected) >= max_images:
                break
    except Exception as e:
        print(f'  eBay error: {e}')

    print(f'  eBay: {len(collected)} images')
    return collected


def scrape_bing(max_images=30):
    """
    Scrape Bing Image Search for Yu-Gi-Oh card photos.
    Parses image URLs from the murl fields in the HTML response.
    """
    collected = []
    queries = [
        'yugioh card photo real',
        'yu-gi-oh single card picture',
        'yugioh card collection photo',
    ]
    for q in queries:
        if len(collected) >= max_images:
            break
        try:
            raw = _get('https://www.bing.com/images/search',
                       params={'q': q, 'form': 'HDRSC2', 'first': 1, 'count': 50})
            text = raw.decode('utf-8', errors='replace')
            # Bing encodes image URLs in murl= fields
            urls = re.findall(r'"murl":"(https?://[^"]+)"', text)
            for img_url in urls:
                img_url = img_url.replace('\\u0026', '&')
                if not any(img_url.lower().endswith(ext)
                           for ext in ('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                # Skip tiny thumbnails and placeholder domains
                skip_domains = ['placeholder', 'gstatic', 'ytimg', 'favicon']
                if any(d in img_url for d in skip_domains):
                    continue
                fname = _url_to_filename(img_url, 'bing_')
                dest  = IMG_DIR / fname
                if not dest.exists():
                    ok = _download_image(img_url, dest)
                    if not ok:
                        continue
                collected.append({'source': 'bing', 'url': img_url, 'image_path': str(dest)})
                if len(collected) >= max_images:
                    break
        except Exception as e:
            print(f'  Bing error ({q}): {e}')
        time.sleep(1)

    print(f'  Bing: {len(collected)} images')
    return collected


# ---------------------------------------------------------------------------
# Guard 1 — YOLO card-presence gate
# ---------------------------------------------------------------------------

_yolo_model = None   # loaded lazily once

def _yolo_has_card(image_path):
    """Return True if YOLO detects at least one card in the image."""
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(args.yolo):
            return True   # weights missing → skip gate, warn once
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(args.yolo)
            _yolo_model.overrides['verbose'] = False
        except Exception as e:
            print(f'  [YOLO] load failed ({e}), gate disabled')
            _yolo_model = 'disabled'
    if _yolo_model == 'disabled':
        return True

    import cv2 as _cv2
    img = _cv2.imread(image_path)
    if img is None:
        return False
    results = _yolo_model(img, conf=args.yolo_conf, verbose=False)
    if not results or results[0].boxes is None:
        return False
    return len(results[0].boxes) > 0


# ---------------------------------------------------------------------------
# Guard 2 — Official-database name check
# ---------------------------------------------------------------------------

_official_names_norm = None   # set of normalised card names from dino_meta.json

def _load_official_names(meta_path):
    global _official_names_norm
    if _official_names_norm is not None:
        return
    if not os.path.exists(meta_path):
        _official_names_norm = set()
        return
    with open(meta_path) as f:
        meta = json.load(f)
    names = set()
    for entry in meta:
        folder = entry.get('card_name', '')
        parts  = folder.rsplit('-', 2)
        display = parts[0] if (len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()) else folder
        names.add(_normalize_name(display.replace('-', ' ')))
    _official_names_norm = names


def _is_official_card(card_name):
    """True if card_name fuzzy-matches any folder in the official database."""
    if _official_names_norm is None:
        return True   # fallback: db not loaded, allow through
    norm = _normalize_name(card_name)
    if norm in _official_names_norm:
        return True
    # Partial containment — handles "Dark Magician" matching "Dark Magician 1"
    return any(norm in db or db in norm for db in _official_names_norm)


# ---------------------------------------------------------------------------
# Ground-truth labeling via Claude vision
# ---------------------------------------------------------------------------

def label_with_claude(image_records):
    """
    For each image, ask Claude to read the card name printed at the top.
    Returns the same list with 'gt_raw' and 'gt_name' fields populated.
    A gt_name of None means Claude could not identify a card.
    """
    import anthropic
    import base64

    client = anthropic.Anthropic()
    labeled = []

    for i, rec in enumerate(image_records):
        path = rec['image_path']
        print(f'  [{i+1}/{len(image_records)}] Labeling {Path(path).name} ...', end=' ', flush=True)

        # Guard 1 — YOLO must detect a physical card in the image
        if not _yolo_has_card(path):
            print('no card detected by YOLO — skip')
            rec['gt_raw']    = 'NO_CARD'
            rec['gt_name']   = None
            rec['skip_reason'] = 'yolo_no_card'
            labeled.append(rec)
            continue

        try:
            with open(path, 'rb') as f:
                img_bytes = f.read()
        except Exception as e:
            print(f'read error: {e}')
            rec['gt_raw'] = None
            rec['gt_name'] = None
            labeled.append(rec)
            continue

        # Detect media type
        if path.lower().endswith('.png'):
            media_type = 'image/png'
        elif path.lower().endswith('.webp'):
            media_type = 'image/webp'
        else:
            media_type = 'image/jpeg'

        b64 = base64.standard_b64encode(img_bytes).decode('utf-8')

        prompt = (
            "Look at this image. If it shows a Yu-Gi-Oh trading card, "
            "read the card name printed at the top of the card and reply with ONLY that name, "
            "exactly as printed (preserve spacing and punctuation). "
            "If the image contains multiple cards, read the name of the most prominently "
            "featured / largest card. "
            "If you cannot find a Yu-Gi-Oh card name, reply with exactly: NO_CARD"
        )

        try:
            msg = client.messages.create(
                model='claude-sonnet-5',
                max_tokens=64,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': b64,
                        }},
                        {'type': 'text', 'text': prompt},
                    ],
                }],
            )
            gt_raw = msg.content[0].text.strip()
        except Exception as e:
            print(f'API error: {e}')
            rec['gt_raw'] = None
            rec['gt_name'] = None
            labeled.append(rec)
            continue

        if gt_raw == 'NO_CARD' or not gt_raw:
            print('no card found in image')
            rec['gt_raw']      = gt_raw
            rec['gt_name']     = None
            rec['skip_reason'] = 'vision_no_card'
        else:
            # Guard 2 — name must exist in the official 10k-card database
            if not _is_official_card(gt_raw):
                print(f'"{gt_raw}" — not in official DB (fan-made?), skip')
                rec['gt_raw']      = gt_raw
                rec['gt_name']     = None
                rec['skip_reason'] = 'not_official_card'
            else:
                print(f'"{gt_raw}"')
                rec['gt_raw']  = gt_raw
                rec['gt_name'] = gt_raw

        labeled.append(rec)
        time.sleep(0.3)   # be gentle with the API

    return labeled


# ---------------------------------------------------------------------------
# DINOv2+FAISS evaluation
# ---------------------------------------------------------------------------

def _normalize_name(name):
    """Lowercase, strip punctuation — used for fuzzy match."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _folder_to_display(folder_name):
    """BlueEyes-White-Dragon-1-89631139  →  Blue-Eyes White Dragon"""
    # folder = {SanitizedName}-{variant_idx}-{passcode}
    # strip trailing -idx-passcode (two trailing numeric-heavy segments)
    parts = folder_name.rsplit('-', 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        card_part = parts[0]
    else:
        card_part = folder_name
    return card_part.replace('-', ' ')


def run_eval(labeled_records, index_path, meta_path, lora_checkpoint=None):
    """Run DINOv2+FAISS on each labeled image; compare to gt_name."""
    # Import here so the BLAS preload in yugioh_utils fires first
    from yugioh_utils import DINOv2Extractor
    import faiss
    import cv2 as _cv2

    print('\nLoading FAISS index for eval...')
    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        meta_list = json.load(f)
    print(f'  {index.ntotal} vectors')

    # Pre-load official name set so _is_official_card works inside eval
    _load_official_names(meta_path)

    extractor = DINOv2Extractor(lora_checkpoint=lora_checkpoint)
    print(f'  DINOv2 on {extractor.device}')

    MIN_COSINE = 0.50   # below this the prediction is almost certainly wrong

    results = []
    for rec in labeled_records:
        if rec.get('gt_name') is None:
            continue  # already rejected by YOLO / DB gate in labeling step

        img_path = rec['image_path']
        img = _cv2.imread(img_path)
        if img is None:
            print(f'  Cannot read {img_path}')
            continue

        # Crop to best YOLO box so DINOv2 sees the card, not the background
        crop = img
        if _yolo_model not in (None, 'disabled') and os.path.exists(args.yolo):
            yolo_results = _yolo_model(img, conf=args.yolo_conf, verbose=False)
            if yolo_results and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
                import numpy as _np
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                confs  = yolo_results[0].boxes.conf.cpu().numpy()
                best   = int(_np.argmax(confs))
                x1, y1, x2, y2 = map(int, boxes[best])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    crop = img[y1:y2, x1:x2]

        try:
            emb = extractor.extract_array(crop)
        except Exception as e:
            print(f'  Embedding failed for {img_path}: {e}')
            continue

        qvec = emb.reshape(1, -1).astype('float32')
        scores, indices = index.search(qvec, k=5)
        scores  = scores[0]
        indices = indices[0]

        top1_folder  = meta_list[indices[0]]['card_name']
        top1_score   = float(scores[0])
        top1_display = _folder_to_display(top1_folder)

        # Guard: if best cosine similarity is very low, the image likely doesn't
        # contain a recognisable official card (fan art, heavy occlusion, etc.)
        if top1_score < MIN_COSINE:
            print(f'  [SKIP] cosine={top1_score:.3f} < {MIN_COSINE}  '
                  f'gt="{rec["gt_name"]}"  ({Path(img_path).name}) — likely not a real card')
            continue

        gt_norm   = _normalize_name(rec['gt_name'])
        pred_norm = _normalize_name(top1_display)
        correct   = gt_norm == pred_norm

        result = {
            'image_path':  img_path,
            'source':      rec.get('source', ''),
            'gt_raw':      rec['gt_raw'],
            'gt_name':     rec['gt_name'],
            'pred_folder': top1_folder,
            'pred_name':   top1_display,
            'pred_score':  top1_score,
            'correct':     correct,
            'top5':        [
                {'folder': meta_list[indices[j]]['card_name'],
                 'name':   _folder_to_display(meta_list[indices[j]]['card_name']),
                 'score':  float(scores[j])}
                for j in range(len(indices))
            ],
        }
        results.append(result)

        status = 'OK ' if correct else 'MISS'
        print(f'  [{status}] gt="{rec["gt_name"]}"  pred="{top1_display}"  '
              f'score={top1_score:.4f}  ({Path(img_path).name})')

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=== Yu-Gi-Oh Real-World Eval ===\n')

    # Pre-load the official card database so both guards are ready
    _load_official_names(args.meta)
    n_official = len(_official_names_norm) if _official_names_norm else 0
    print(f'Official DB: {n_official} card names loaded from {args.meta}')

    # ---- 1. Scrape images ------------------------------------------------
    if args.skip_scrape:
        print('Skip-scrape: loading existing images...')
        all_records = [
            {'source': 'existing', 'url': '', 'image_path': str(p)}
            for p in sorted(IMG_DIR.glob('*'))
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
        ]
        print(f'  {len(all_records)} images found in {IMG_DIR}')
    else:
        per_source = max(1, args.limit // len(args.sources))
        all_records = []

        print('Scraping images...')
        scrapers = {
            'reddit': lambda: scrape_reddit(per_source),
            'flickr': lambda: scrape_flickr(per_source),
            'ebay':   lambda: scrape_ebay(per_source),
            'bing':   lambda: scrape_bing(per_source),
        }
        for src in args.sources:
            recs = scrapers[src]()
            all_records.extend(recs)

        all_records = all_records[:args.limit]
        print(f'\nTotal downloaded: {len(all_records)} images')

    if not all_records:
        print('No images to evaluate. Exiting.')
        sys.exit(0)

    # Save download manifest
    manifest_path = OUT_DIR / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(all_records, f, indent=2)
    print(f'Manifest saved: {manifest_path}')

    # ---- 2. Label ground truth -------------------------------------------
    print('\nLabeling ground truth with Claude vision...')
    labeled = label_with_claude(all_records)

    cards_found = [r for r in labeled if r.get('gt_name')]
    no_card     = [r for r in labeled if not r.get('gt_name')]
    print(f'\n  Cards identified: {len(cards_found)} / {len(labeled)}  '
          f'(skipping {len(no_card)} non-card images)')

    with open(OUT_DIR / 'labeled.json', 'w') as f:
        json.dump(labeled, f, indent=2)

    if not cards_found:
        print('No cards identified in any image. Nothing to evaluate.')
        sys.exit(0)

    if args.skip_eval:
        print('Skip-eval flag set — ground-truth labels saved, no DINOv2 run.')
        sys.exit(0)

    # ---- 3. DINOv2+FAISS eval -------------------------------------------
    assert os.path.exists(args.index), f'FAISS index not found: {args.index}'
    assert os.path.exists(args.meta),  f'Meta JSON not found: {args.meta}'

    print('\nRunning DINOv2+FAISS evaluation...')
    results = run_eval(cards_found, args.index, args.meta,
                       lora_checkpoint=args.lora_checkpoint)

    # ---- 4. Report -------------------------------------------------------
    if not results:
        print('No results to report.')
        sys.exit(0)

    correct_total = sum(1 for r in results if r['correct'])
    accuracy      = correct_total / len(results)

    by_source = {}
    for r in results:
        src = r['source']
        by_source.setdefault(src, {'correct': 0, 'total': 0})
        by_source[src]['total'] += 1
        if r['correct']:
            by_source[src]['correct'] += 1

    summary_lines = [
        '=== Real-World Eval Results ===',
        f'Total evaluated : {len(results)}',
        f'Correct (rank-1): {correct_total}',
        f'Accuracy        : {accuracy:.4f} ({accuracy*100:.1f}%)',
        '',
        'Per-source breakdown:',
    ]
    for src, counts in sorted(by_source.items()):
        acc = counts['correct'] / counts['total']
        summary_lines.append(f'  {src:<10} {counts["correct"]}/{counts["total"]}  '
                              f'({acc*100:.1f}%)')
    summary_lines.append('')
    summary_lines.append('Incorrect predictions:')
    for r in results:
        if not r['correct']:
            summary_lines.append(
                f'  gt="{r["gt_name"]}"  pred="{r["pred_name"]}"  '
                f'score={r["pred_score"]:.4f}  ({Path(r["image_path"]).name})'
            )

    summary = '\n'.join(summary_lines)
    print('\n' + summary)

    with open(OUT_DIR / 'summary.txt', 'w') as f:
        f.write(summary + '\n')
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
