#!/usr/bin/env python3
# scripts/debug_jsonl.py
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Użycie: python scripts/debug_jsonl.py <ścieżka_do_pliku.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERROR] Plik {path} nie istnieje.")
        sys.exit(1)

    print(f"[INFO] Sprawdzam plik: {path}")
    invalid_count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                print(f"[WARN] Linia {i}: pusta linia (do usunięcia)")
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                invalid_count += 1
                snippet = line[:200].replace("\n", "\\n")
                print(f"[ERROR] Linia {i} ma błąd JSON: {e}")
                print(f"        Początek linii: {snippet}...")
    if invalid_count == 0:
        print("[OK] Wszystkie linie są poprawnym JSON-em ✅")
    else:
        print(f"[FAIL] Znaleziono {invalid_count} błędnych linii w {path}")

if __name__ == "__main__":
    main()
