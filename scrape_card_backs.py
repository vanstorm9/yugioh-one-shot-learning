#!/usr/bin/env python3
"""
Scrape Yu-Gi-Oh card back images for YOLO training data.

Card backs are a single fixed design, so we just need variety in
lighting, angle, and quality — not card identity.

Output:
  card_backs/*.jpg   (ready to pass as --card-back-dir to generate_detection_dataset.py)

Usage:
  python scrape_card_backs.py
  python scrape_card_backs.py --n 30 --out ./card_backs
"""

import os
import re
import sys
import time
import random
import argparse
import requests

OUT_DIR = "./card_backs"
TARGET  = 20

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

QUERIES = [
    "yugioh card back",
    "yugioh trading card back design",
    "yu-gi-oh card reverse side",
    "yugioh card sleeve back",
]


def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"    GET failed {url[:70]}: {e}")
        return None


def _download(url, dest):
    try:
        r = requests.get(url, headers=HEADERS, timeout=18, stream=True)
        if r.status_code != 200:
            return False
        data = b""
        for chunk in r.iter_content(8192):
            data += chunk
            if len(data) > 10_000_000:
                return False
        if len(data) < 8_000:
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def search_reddit(query, n=10):
    urls = []
    r = _get("https://www.reddit.com/r/yugioh/search.rss",
             params={"q": query, "sort": "relevance", "t": "all", "limit": 25})
    if not r:
        return []
    for m in re.findall(r'(https://i\.redd\.it/[^\s"<&]+)', r.text):
        if m not in urls:
            urls.append(m)
    for m in re.findall(r'(https://preview\.redd\.it/[^\s"<&]+\.(?:jpg|jpeg|png))', r.text):
        clean = m.split("?")[0]
        if clean not in urls:
            urls.append(clean)
    time.sleep(random.uniform(1.5, 2.5))
    return urls[:n]


def search_flickr(query, n=10):
    urls = []
    tag = re.sub(r'[^a-z0-9]', '', query.lower())
    r = _get("https://api.flickr.com/services/feeds/photos_public.gne",
             params={"tags": tag, "format": "json", "nojsoncallback": 1})
    if not r:
        return []
    try:
        data = r.json()
        for item in data.get("items", []):
            url = item.get("media", {}).get("m", "").replace("_m.", "_b.")
            if url and url not in urls:
                urls.append(url)
    except Exception:
        pass
    time.sleep(1)
    return urls[:n]


def search_ddg(query, n=10):
    urls = []
    r = _get("https://duckduckgo.com/", params={"q": query})
    if not r:
        return []
    vqd_m = re.search(r'vqd=([\d-]+)', r.text)
    if not vqd_m:
        return []
    time.sleep(0.5)
    r2 = _get("https://duckduckgo.com/i.js",
              params={"l": "us-en", "o": "json", "q": query,
                      "vqd": vqd_m.group(1), "f": ",,,,,", "p": "1"})
    if not r2:
        return []
    try:
        for item in r2.json().get("results", [])[:n * 2]:
            url = item.get("image", "")
            if url and re.search(r'\.(jpg|jpeg|png)', url, re.I) and url not in urls:
                urls.append(url)
    except Exception:
        pass
    return urls[:n]


def run(n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    existing = set(os.listdir(out_dir))
    collected = []
    seen_urls = set()

    for query in QUERIES:
        if len(collected) >= n:
            break
        print(f"\nQuery: '{query}'")
        print("  Reddit...")
        reddit = search_reddit(query, n)
        print("  Flickr...")
        flickr = search_flickr(query, n // 2)
        print("  DuckDuckGo...")
        ddg = search_ddg(query, n)

        candidates = reddit + flickr + ddg
        for url in candidates:
            if len(collected) >= n:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            ext = url.rsplit(".", 1)[-1].lower().split("?")[0]
            if ext not in ("jpg", "jpeg", "png"):
                ext = "jpg"
            fname = f"card_back_{len(collected):04d}.{ext}"
            dest  = os.path.join(out_dir, fname)

            ok = _download(url, dest)
            if ok:
                collected.append(fname)
                source = ("reddit" if "redd.it" in url
                          else "flickr" if "flickr" in url else "ddg")
                print(f"  [{len(collected):2d}/{n}] {fname}  ({source})")
            time.sleep(random.uniform(0.3, 0.8))

    print(f"\nDone. {len(collected)} card back images saved to {os.path.abspath(out_dir)}")
    if len(collected) < 5:
        print("WARNING: fewer than 5 images collected. Consider adding card back images manually.")
        print("  Any photo of a Yu-Gi-Oh card back works — the generator will augment heavily.")
    print(f"\nNext step:")
    print(f"  python generate_detection_dataset.py --card-back-dir {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",   type=int, default=TARGET, help="Number of images to collect")
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()
    run(args.n, args.out)
