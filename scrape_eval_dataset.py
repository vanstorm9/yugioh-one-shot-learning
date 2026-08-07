#!/usr/bin/env python3
"""
Scrape real-world Yu-Gi-Oh card photos for the recognition eval.

Sources  : Bing Image Search · Reddit r/yugioh · eBay listings · Flickr feed
Output   : eval_real_world/images/{source}_{n:03d}.jpg
           eval_real_world/manifest.json   (url, source, local path)

Usage:
  python scrape_eval_dataset.py
  python scrape_eval_dataset.py --per-source 25
"""

import os
import re
import sys
import json
import time
import random
import argparse
import requests
from pathlib import Path
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR      = "eval_real_world/images"
MANIFEST     = "eval_real_world/manifest.json"
PER_SOURCE   = 25        # images to try per source
MAX_BYTES    = 8_000_000 # skip images > 8 MB
MIN_BYTES    = 5_000     # skip suspiciously small files
DELAY        = (0.8, 1.8) # random sleep between downloads (polite)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def _get(url, **kwargs):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"    GET failed {url[:80]}: {e}")
        return None


def scrape_bing(n=PER_SOURCE):
    """Bing Images — parses embedded JSON blobs from search results."""
    queries = [
        "yugioh card photo",
        "yugioh dark magician real photo",
        "yugioh blue eyes white dragon photo",
        "yugioh ultra rare card",
        "yugioh card in hand photo",
    ]
    urls = []
    for q in queries:
        if len(urls) >= n:
            break
        url = f"https://www.bing.com/images/search?q={quote_plus(q)}&form=HDRSC2&first=1"
        r = _get(url)
        if not r:
            continue
        # Bing embeds image data as JSON in `m` attribute of <a class="iusc"> tags
        for m in re.findall(r'"murl"\s*:\s*"(https?://[^"]+\.(?:jpg|jpeg|png)[^"]*)"',
                             r.text, re.IGNORECASE):
            if m not in urls:
                urls.append(m)
        time.sleep(random.uniform(*DELAY))
    return urls[:n]


def scrape_reddit(n=PER_SOURCE):
    """Reddit r/yugioh — uses the RSS feed (no auth required, JSON API is blocked)."""
    urls = []
    # Reddit RSS still works without OAuth; iterate new/hot/top feeds
    feeds = [
        "https://www.reddit.com/r/yugioh/new.rss",
        "https://www.reddit.com/r/yugioh/hot.rss",
        "https://www.reddit.com/r/yugioh/top.rss?t=month",
    ]
    for feed_url in feeds:
        if len(urls) >= n:
            break
        r = _get(feed_url)
        if not r:
            continue
        # Extract i.redd.it direct image links
        for m in re.findall(
            r'(https://i\.redd\.it/[^\s"<]+\.(?:jpg|jpeg|png|gif))',
            r.text, re.IGNORECASE):
            if m not in urls:
                urls.append(m)
        # Also grab preview.redd.it links as fallback
        for m in re.findall(
            r'(https://preview\.redd\.it/[^\s"<&]+\.(?:jpg|jpeg|png))',
            r.text, re.IGNORECASE):
            clean = m.split("?")[0]  # strip query params — preview still works
            if clean not in urls:
                urls.append(clean)
        time.sleep(random.uniform(*DELAY))
    return urls[:n]


def scrape_ebay(n=PER_SOURCE):
    """eBay — card listing photos from search results."""
    queries = [
        "yugioh card single",
        "yugioh dark magician card",
        "yugioh ultra rare single card",
    ]
    urls = []
    for q in queries:
        if len(urls) >= n:
            break
        r = _get(f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(q)}&_pgn=1")
        if not r:
            continue
        # eBay listing images: <img class="s-item__image-img" src="...">
        for m in re.findall(
            r'<img[^>]+class="[^"]*s-item__image-img[^"]*"[^>]+src="(https?://[^"]+)"',
            r.text):
            if m not in urls and "thumb" not in m.lower():
                urls.append(m)
        # Also pull from data-src lazy-load attributes
        for m in re.findall(
            r'<img[^>]+class="[^"]*s-item__image-img[^"]*"[^>]+data-src="(https?://[^"]+)"',
            r.text):
            if m not in urls:
                urls.append(m)
        time.sleep(random.uniform(*DELAY))
    return urls[:n]


def scrape_flickr(n=PER_SOURCE):
    """Flickr public photo feed — no API key needed."""
    tags = ["yugioh", "yugiohcard", "yugioh+card"]
    urls = []
    for tag in tags:
        if len(urls) >= n:
            break
        r = _get(
            f"https://api.flickr.com/services/feeds/photos_public.gne"
            f"?tags={tag}&format=json&nojsoncallback=1"
        )
        if not r:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        for item in data.get("items", []):
            media = item.get("media", {})
            img_url = media.get("m", "").replace("_m.jpg", "_b.jpg")  # get larger version
            if img_url and img_url not in urls:
                urls.append(img_url)
        time.sleep(random.uniform(*DELAY))
    return urls[:n]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def download_image(url, dest_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        if r.status_code != 200:
            return False
        content_type = r.headers.get("content-type", "")
        if "image" not in content_type and not dest_path.endswith((".jpg", ".jpeg", ".png")):
            return False
        data = b""
        for chunk in r.iter_content(8192):
            data += chunk
            if len(data) > MAX_BYTES:
                return False
        if len(data) < MIN_BYTES:
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        return False


def run(per_source):
    os.makedirs(OUT_DIR, exist_ok=True)

    scrapers = {
        "bing":    scrape_bing,
        "reddit":  scrape_reddit,
        "ebay":    scrape_ebay,
        "flickr":  scrape_flickr,
    }

    manifest = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
    existing_urls = {e["url"] for e in manifest}

    for source, scraper in scrapers.items():
        print(f"\n{'='*60}")
        print(f"Scraping {source} ({per_source} images)...")
        print(f"{'='*60}")

        try:
            urls = scraper(per_source)
        except Exception as e:
            print(f"  Scraper failed: {e}")
            continue

        print(f"  Found {len(urls)} candidate URLs")
        saved = 0

        for url in urls:
            if url in existing_urls:
                print(f"  skip (already have): {url[:60]}")
                continue

            idx = len(manifest)
            ext = ".jpg"
            for candidate_ext in (".png", ".jpeg", ".jpg"):
                if candidate_ext in url.lower():
                    ext = candidate_ext
                    break
            fname = f"{source}_{idx:03d}{ext}"
            dest  = os.path.join(OUT_DIR, fname)

            ok = download_image(url, dest)
            if ok:
                entry = {"idx": idx, "source": source, "url": url,
                         "file": fname, "label": None}
                manifest.append(entry)
                existing_urls.add(url)
                saved += 1
                print(f"  [{saved:2d}] saved {fname}  ({url[:60]})")
            else:
                print(f"  [ ] failed  {url[:60]}")

            time.sleep(random.uniform(*DELAY))

            if saved >= per_source:
                break

        print(f"  Saved {saved} images from {source}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    total = len([e for e in manifest if e["label"] is None])
    print(f"\nDone. {len(manifest)} total images in manifest.")
    print(f"Unlabeled (need ground truth): {total}")
    print(f"Images at: {os.path.abspath(OUT_DIR)}")
    print(f"Manifest:  {os.path.abspath(MANIFEST)}")
    print()
    print("Next step: run  python label_eval_dataset.py  to add ground-truth labels.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=PER_SOURCE)
    ap.add_argument("--out",        type=str, default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR  = args.out
    MANIFEST = os.path.join(os.path.dirname(args.out), "manifest.json")
    run(args.per_source)
