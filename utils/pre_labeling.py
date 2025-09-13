# src/labeling_prepare.py
# Step 1: Audit & inputs split for labeling
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

EMOTIONS = [
    "Positive", "Negative", "Happiness", "Delight", "Inspiring",
    "Surprise", "Compassion", "Fear", "Sadness", "Disgust", "Anger",
    "Ironic", "Political", "Interesting", "Understandable", "Offensive", "Funny"
]

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e
    return rows

def hash_personas(personas: List[Dict[str, Any]]) -> str:
    """
    Stabilny hash: sortujemy po 'id', serializujemy 'id,name,sensitivity,values,lang'
    i haszujemy sha256. Zmiana treści lub kolejności zostanie wykryta.
    """
    norm = []
    for p in personas:
        norm.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "sensitivity": p.get("sensitivity"),
            "values": p.get("values"),
            "lang": p.get("lang"),
        })
    norm_sorted = sorted(norm, key=lambda x: (x["lang"], x["id"]))
    blob = json.dumps(norm_sorted, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def build_audit(
    opinions_path: Path,
    personas_path: Path,
    out_dir: Path,
    batch_size: int,
    concurrency: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    opinions = read_jsonl(opinions_path)
    personas = read_jsonl(personas_path)

    # Split opinions by lang
    opinions_pl = [o for o in opinions if o.get("lang") == "pl"]
    opinions_en = [o for o in opinions if o.get("lang") == "en"]

    # Split personas by lang
    personas_pl = [p for p in personas if p.get("lang") == "pl"]
    personas_en = [p for p in personas if p.get("lang") == "en"]

    # Basic validations
    if not opinions_pl and not opinions_en:
        raise SystemExit("No opinions found (neither 'pl' nor 'en').")

    if not personas_pl or not personas_en:
        raise SystemExit("Missing personas for one of the languages (need both 'pl' and 'en').")

    # Persona IDs (must match across langs)
    ids_pl = sorted({p.get("id") for p in personas_pl})
    ids_en = sorted({p.get("id") for p in personas_en})
    if ids_pl != ids_en:
        print("[WARN] Persona ID sets differ between PL and EN.")
    persona_ids = ids_pl if ids_pl == ids_en else sorted(set(ids_pl) | set(ids_en))

    # Hashes
    personas_hash_pl = hash_personas(personas_pl)
    personas_hash_en = hash_personas(personas_en)

    audit = {
        "total_opinions": len(opinions),
        "opinions_per_lang": {
            "pl": len(opinions_pl),
            "en": len(opinions_en),
        },
        "personas": {
            "count_pl": len(personas_pl),
            "count_en": len(personas_en),
            "ids": persona_ids,
            "hash_pl": personas_hash_pl,
            "hash_en": personas_hash_en,
        },
        "emotions": EMOTIONS,
        "batching": {
            "batch_size": batch_size,
            "concurrency": concurrency,
        },
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "notes": "Generated before labeling run. Hashes ensure personas consistency."
    }

    with (out_dir / "labeling_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # Console summary
    print("✅ labeling_audit.json created")
    print(f" - opinions: total={len(opinions)} | pl={len(opinions_pl)} | en={len(opinions_en)}")
    print(f" - personas: pl={len(personas_pl)} | en={len(personas_en)} | ids={len(persona_ids)}")
    print(f" - hashes: pl={personas_hash_pl[:12]}… en={personas_hash_en[:12]}…")
    print(f" - batching: batch_size={batch_size}, concurrency={concurrency}")
    print(f" -> {out_dir / 'labeling_audit.json'}")
    return audit

def main():
    parser = argparse.ArgumentParser(description="Prepare audit for labeling step (step 1).")
    parser.add_argument("--opinions", default="dataset/opinions.jsonl", help="Path to opinions.jsonl")
    parser.add_argument("--personas", default="dataset/personas.jsonl", help="Path to personas.jsonl")
    parser.add_argument("--outdir", default="data/working", help="Output directory for audit")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    build_audit(
        opinions_path=Path(args.opinions),
        personas_path=Path(args.personas),
        out_dir=Path(args.outdir),
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )

if __name__ == "__main__":
    main()
