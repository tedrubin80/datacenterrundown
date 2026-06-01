#!/usr/bin/env python3
"""Download FEMA DisasterDeclarationsSummaries from the OpenFEMA API.

This dataset is required for the Idea 5 extreme weather models (Notebook 04).
Without it, the pipeline falls back to synthetic event data and the Weibull
survival model concordance index drops from 0.944 to ~0.57.

Usage:
    python pipelines/download_fema.py

Output:
    data/raw/fema/DisasterDeclarationsSummaries.csv  (~35 MB, ~70K rows)

Source:
    OpenFEMA API v2 — https://www.fema.gov/about/openfema/api
    Dataset: DisasterDeclarationsSummaries
    License: Public domain (US government data)
"""

import sys
import json
import time
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "fema"
OUTPUT_FILE = OUTPUT_DIR / "DisasterDeclarationsSummaries.csv"

BASE_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
PAGE_SIZE = 1000


def get_total_count() -> int:
    url = f"{BASE_URL}?$inlinecount=allpages&$top=1&$select=id"
    with urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    return data.get("metadata", {}).get("count", 0)


def download_fema(output_path: Path = OUTPUT_FILE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"File already exists: {output_path} ({size_mb:.1f} MB)")
        print("Delete it and re-run to refresh.")
        return

    print("Fetching record count from OpenFEMA API...")
    try:
        total = get_total_count()
    except URLError as e:
        print(f"Network error: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    print(f"Total records: {total:,}")
    pages = (total // PAGE_SIZE) + 1
    print(f"Downloading {pages} pages of {PAGE_SIZE} records each...")

    # Write header + all pages
    header_written = False
    with open(output_path, "w", encoding="utf-8") as out:
        for page in range(pages):
            skip = page * PAGE_SIZE
            url = (
                f"{BASE_URL}?$format=csv&$top={PAGE_SIZE}&$skip={skip}"
                f"&$orderby=declarationDate asc"
            )
            try:
                with urlopen(url, timeout=60) as r:
                    content = r.read().decode("utf-8")
            except URLError as e:
                print(f"\nError on page {page}: {e}. Retrying once...")
                time.sleep(5)
                with urlopen(url, timeout=60) as r:
                    content = r.read().decode("utf-8")

            lines = content.strip().splitlines()
            if not lines:
                continue

            if not header_written:
                out.write("\n".join(lines) + "\n")
                header_written = True
            else:
                # Skip header row on subsequent pages
                out.write("\n".join(lines[1:]) + "\n")

            downloaded = min((page + 1) * PAGE_SIZE, total)
            pct = downloaded / total * 100
            print(f"\r  {downloaded:,} / {total:,} ({pct:.0f}%)", end="", flush=True)
            time.sleep(0.2)  # be polite to the API

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nDone. Saved to {output_path} ({size_mb:.1f} MB)")
    print("\nNext step: run  make idea5  or  python pipelines/train_idea5.py")


if __name__ == "__main__":
    download_fema()
