#!/usr/bin/env python3
"""
Download Sefaria texts via official API.
Sefaria is an open-source project with freely available texts.
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm
import time

SEFARIA_API_BASE = "https://www.sefaria.org/api"

def get_index_list():
    """Get list of all available texts in Sefaria."""
    response = requests.get(f"{SEFARIA_API_BASE}/index/")
    response.raise_for_status()
    return response.json()

def get_text(title, context=0):
    """Get text content for a specific title."""
    url = f"{SEFARIA_API_BASE}/texts/{title}"
    if context > 0:
        url += f"?context={context}"
    
    response = requests.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()

def download_all_texts(output_dir="sefaria_texts", delay=0.1):
    """Download all available texts from Sefaria."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Fetching index list...")
    indices = get_index_list()
    print(f"Found {len(indices)} texts")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for index in tqdm(indices):
        title = index.get("title")
        if not title:
            continue
        
        # Skip non-text categories
        categories = index.get("categories", [])
        if not categories:
            continue
        
        # Create safe filename
        safe_title = title.replace("/", "_").replace(":", "-")
        category_path = output_path / "_".join(categories)
        category_path.mkdir(parents=True, exist_ok=True)
        
        output_file = category_path / f"{safe_title}.json"
        
        # Skip if already downloaded
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            text_data = get_text(title, context=3)  # Get context for better understanding
            if text_data is None:
                failed += 1
                continue
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(text_data, f, ensure_ascii=False, indent=2)
            
            downloaded += 1
            
            # Be respectful to their API
            time.sleep(delay)
            
        except Exception as e:
            print(f"Failed to download {title}: {e}")
            failed += 1
    
    print(f"\nDownload complete:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_path.absolute()}")

if __name__ == "__main__":
    download_all_texts(delay=0.2)