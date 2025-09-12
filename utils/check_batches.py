# scripts/check_batches.py
import os
import re
from pathlib import Path

BATCH_DIR = Path("data/working/translation_batches")

def main():
    if not BATCH_DIR.exists():
        print(f"[ERROR] Folder {BATCH_DIR} nie istnieje.")
        return

    # znajdź wszystkie pliki batch_*_request.jsonl (to nasza referencja)
    request_files = sorted(BATCH_DIR.glob("batch_*_request.jsonl"))
    if not request_files:
        print("[WARN] Brak plików request w katalogu.")
        return

    missing = []
    for req_file in request_files:
        batch_id = re.search(r"batch_(\d{4})_request\.jsonl", req_file.name)
        if not batch_id:
            continue
        clean_file = BATCH_DIR / f"batch_{batch_id.group(1)}_clean.jsonl"
        if not clean_file.exists():
            missing.append(batch_id.group(1))

    if missing:
        print(f"[MISSING] Brak clean.jsonl dla batchy: {', '.join(missing)}")
    else:
        print("[OK] Wszystkie batch'e mają pliki clean.jsonl")

if __name__ == "__main__":
    main()