#!/usr/bin/env python3
"""
Download all Yu-Gi-Oh card images from YGOPRODeck API.

API docs:  https://db.ygoprodeck.com/api-guide/
Cards API: https://db.ygoprodeck.com/api/v7/cardinfo.php

Directory structure produced (matches original cardDatabaseFull):
  cardDatabaseFull/{SanitizedName}-{variant_idx}-{passcode}/{passcode}{variant_idx}.jpg

Name sanitization rule (reverse-engineered from existing paths):
  "Blue-Eyes White Dragon" -> "BlueEyes-White-Dragon"
  "Dark Magician"          -> "Dark-Magician"
  i.e. strip non-alphanumeric non-space chars, then replace spaces with hyphens.

Usage:
  python download_dataset.py [--workers N] [--output-dir PATH] [--cropped]
"""

import os
import re
import sys
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
API_URL     = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
OUTPUT_DIR  = "./cardDatabaseFull"
MAX_WORKERS = 16
RETRY_MAX   = 3
RETRY_DELAY = 2   # seconds between retries
SESSION_TIMEOUT = 15


def sanitize_name(name: str) -> str:
    """
    Reproduce the folder-name convention used in the original dataset.
    "Blue-Eyes White Dragon" -> "BlueEyes-White-Dragon"
    """
    # Remove anything that isn't a letter, digit, or space
    name = re.sub(r"[^a-zA-Z0-9 ]", "", name)
    # Collapse multiple spaces, strip edges
    name = re.sub(r" +", " ", name).strip()
    # Replace spaces with hyphens
    return name.replace(" ", "-")


def folder_and_file(card_name: str, passcode: int, variant_idx: int) -> tuple:
    """Return (folder_name, file_name) for a given card+variant."""
    safe_name   = sanitize_name(card_name)
    folder_name = f"{safe_name}-{variant_idx}-{passcode}"
    file_name   = f"{passcode}{variant_idx}.jpg"
    return folder_name, file_name


def download_image(session: requests.Session, url: str, dest_path: str) -> bool:
    """Download url to dest_path with retries. Returns True on success."""
    for attempt in range(RETRY_MAX):
        try:
            r = session.get(url, timeout=SESSION_TIMEOUT)
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return True
            if r.status_code == 404:
                return False  # no point retrying
        except requests.RequestException:
            pass
        if attempt < RETRY_MAX - 1:
            time.sleep(RETRY_DELAY)
    return False


def process_card(card: dict, output_dir: str, use_cropped: bool) -> dict:
    """Download all art variants for a single card. Returns a result dict."""
    passcode    = card["id"]
    card_name   = card["name"]
    card_images = card.get("card_images", [])

    result = {"card": card_name, "downloaded": 0, "skipped": 0, "failed": 0}
    session = requests.Session()

    for variant_idx, img_info in enumerate(card_images):
        url = img_info["image_url_cropped"] if use_cropped else img_info["image_url"]
        folder_name, file_name = folder_and_file(card_name, passcode, variant_idx)
        folder_path = os.path.join(output_dir, folder_name)
        dest_path   = os.path.join(folder_path, file_name)

        if os.path.exists(dest_path):
            result["skipped"] += 1
            continue

        os.makedirs(folder_path, exist_ok=True)
        ok = download_image(session, url, dest_path)
        if ok:
            result["downloaded"] += 1
        else:
            result["failed"] += 1
            print(f"  FAIL: {card_name} variant {variant_idx}  url={url}", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Download YGOPRODeck card image dataset")
    parser.add_argument("--workers",    type=int,  default=MAX_WORKERS,
                        help="Concurrent download threads (default: 16)")
    parser.add_argument("--output-dir", type=str,  default=OUTPUT_DIR,
                        help="Root output directory (default: ./cardDatabaseFull)")
    parser.add_argument("--cropped",    action="store_true",
                        help="Download art-only crops instead of full card images")
    parser.add_argument("--limit",      type=int,  default=None,
                        help="Only process the first N cards from the API response")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetch card list (single request, ~12 k cards)
    # ------------------------------------------------------------------
    print("Fetching card list from YGOPRODeck API...")
    try:
        r = requests.get(API_URL, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR fetching card list: {e}")
        sys.exit(1)

    cards = r.json().get("data", [])
    if args.limit:
        cards = cards[:args.limit]
    total_cards  = len(cards)
    total_images = sum(len(c.get("card_images", [])) for c in cards)
    print(f"  {total_cards} cards  /  {total_images} total images")
    image_type = "cropped art" if args.cropped else "normal (full card)"
    print(f"  Image type : {image_type}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Workers    : {args.workers}")
    print()

    # ------------------------------------------------------------------
    # Download concurrently
    # ------------------------------------------------------------------
    downloaded = skipped = failed = 0
    completed_cards = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_card, card, args.output_dir, args.cropped): card
            for card in cards
        }
        for future in as_completed(futures):
            completed_cards += 1
            try:
                res = future.result()
                downloaded += res["downloaded"]
                skipped    += res["skipped"]
                failed     += res["failed"]
            except Exception as e:
                card = futures[future]
                print(f"  ERROR: {card['name']}: {e}", flush=True)
                failed += 1

            if completed_cards % 500 == 0 or completed_cards == total_cards:
                total_done = downloaded + skipped
                pct = total_done / total_images * 100 if total_images else 0
                print(
                    f"  [{completed_cards}/{total_cards} cards]  "
                    f"downloaded={downloaded}  skipped={skipped}  "
                    f"failed={failed}  ({pct:.1f}%)",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("Done.")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped}  (already existed)")
    print(f"  Failed     : {failed}")
    print(f"  Output     : {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
