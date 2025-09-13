# src/builder.py
# Build final dataset: join opinions + embeddings (+ personas) into target JSONL format.
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
from collections import defaultdict

EMOTIONS = [
    "Positive","Negative","Happiness","Delight","Inspiring",
    "Surprise","Compassion","Fear","Sadness","Disgust","Anger",
    "Ironic","Political","Interesting","Understandable","Offensive","Funny"
]

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] Invalid JSON at {path}:{ln}: {e}")
    return rows

def write_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def persona_desc(p: Dict[str, Any], lang: str) -> str:
    # Budujemy opis zgodnie z wymaganym formatem.
    # EN: "<name> Sensitivity: ... Values: ..."
    # PL: "<name> Wrażliwość: ... Wartości: ..."
    name = p.get("name", "").strip()
    sens = p.get("sensitivity", "").strip()
    vals = p.get("values", "").strip()
    if lang == "pl":
        return f"{name} Wrażliwość: {sens}. Wartości: {vals}."
    return f"{name} Sensitivity: {sens}. Values: {vals}."

def build_final(
    opinions_path: Path,
    embeddings_path: Path,
    personas_path: Path,
    out_path: Path,
    missing_path: Path,
    audit_path: Path,
):
    # Load
    opinions = read_jsonl(opinions_path)
    emb = read_jsonl(embeddings_path)
    personas = read_jsonl(personas_path)

    # Index opinions: (opinion_id, lang) -> text
    opin_idx: Dict[Tuple[str, str], str] = {}
    for o in opinions:
        oid = o.get("opinion_id")
        lg = o.get("lang")
        txt = o.get("text")
        if not oid or not lg or txt is None:
            continue
        opin_idx[(oid, lg)] = txt

    # Index personas by (persona_id, lang)
    per_idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in personas:
        pid = p.get("id")
        lg = p.get("lang")
        if not pid or not lg:
            continue
        per_idx[(pid, lg)] = p

    # Prepare outputs
    final_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    # Validate labels and build final
    # We preserve order as in embeddings.jsonl (which is PL first then EN if you used our consolidation)
    for r in emb:
        oid = r.get("opinion_id")
        lg = r.get("lang")
        pid = r.get("persona_id")
        labels = r.get("labels")

        # basic validation
        if not oid or not lg or not pid or not isinstance(labels, list) or len(labels) != len(EMOTIONS):
            missing_rows.append({"opinion_id": oid, "lang": lg, "persona_id": pid, "reason": "invalid_row"})
            continue

        # normalize labels to 0/1 ints
        try:
            labels = [1 if int(x) == 1 else 0 for x in labels]
        except Exception:
            missing_rows.append({"opinion_id": oid, "lang": lg, "persona_id": pid, "reason": "invalid_labels"})
            continue

        # fetch text
        text = opin_idx.get((oid, lg))
        if text is None:
            missing_rows.append({"opinion_id": oid, "lang": lg, "persona_id": pid, "reason": "missing_text"})
            continue

        # fetch persona in this lang (fallback: try the other lang if strict match not found)
        pobj = per_idx.get((pid, lg))
        if not pobj:
            # fallback to 'en' if record is 'pl', or to 'pl' if 'en' missing — optional
            fallback_lg = "en" if lg == "pl" else "pl"
            pobj = per_idx.get((pid, fallback_lg))
        if not pobj:
            missing_rows.append({"opinion_id": oid, "lang": lg, "persona_id": pid, "reason": "missing_persona"})
            continue

        pdesc = persona_desc(pobj, lg)

        final_rows.append({
            "id": f"{oid}__p={pid}",
            "opinion_id": oid,
            "text": text,
            "persona_id": pid,
            "persona_desc": pdesc,
            "lang": lg,
            "labels": labels
        })

    # Write outputs
    write_jsonl(final_rows, out_path)
    if missing_rows:
        write_jsonl(missing_rows, missing_path)

    # Audit
    audit = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "opinions": str(opinions_path),
            "embeddings": str(embeddings_path),
            "personas": str(personas_path)
        },
        "emotions": EMOTIONS,
        "counts": {
            "embeddings_in": len(emb),
            "final_written": len(final_rows),
            "missing": len(missing_rows)
        },
        "outputs": {
            "final_dataset": str(out_path),
            "missing_report": str(missing_path)
        }
    }
    write_json(audit, audit_path)
    print(f"✅ Final dataset written: {out_path} ({len(final_rows)} rows)")
    if missing_rows:
        print(f"⚠️ Missing/skipped records: {len(missing_rows)} → {missing_path}")
    print(f"🧾 Audit: {audit_path}")

def main():
    parser = argparse.ArgumentParser(description="Build final dataset (opinions + embeddings + personas).")
    parser.add_argument("--opinions", default="dataset/opinions.jsonl", help="Path to opinions.jsonl")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.jsonl (balanced or raw)")
    parser.add_argument("--personas", default="dataset/personas.jsonl", help="Path to personas.jsonl")
    parser.add_argument("--out", default="data/processed/final_dataset.jsonl", help="Output final dataset path")
    parser.add_argument("--missing-out", default="data/processed/builder_missing.jsonl", help="Missing/invalid rows report")
    parser.add_argument("--audit-out", default="data/processed/builder_audit.json", help="Audit json path")
    args = parser.parse_args()

    # Auto-pick embeddings file if not provided
    if args.embeddings is None:
        balanced = Path("data/working/embeddings_balanced.jsonl")
        raw = Path("data/working/embeddings.jsonl")
        if balanced.exists():
            embeddings_path = balanced
        elif raw.exists():
            embeddings_path = raw
        else:
            raise SystemExit("No embeddings file found. Expected data/working/embeddings_balanced.jsonl or data/working/embeddings.jsonl")
    else:
        embeddings_path = Path(args.embeddings)

    build_final(
        opinions_path=Path(args.opinions),
        embeddings_path=embeddings_path,
        personas_path=Path(args.personas),
        out_path=Path(args.out),
        missing_path=Path(args.missing_out),
        audit_path=Path(args.audit_out),
    )

if __name__ == "__main__":
    main()
