# src/balance.py
# Balance embeddings by capping Negative, keeping rare labels, and deduping.
# Outputs a FULL balanced dataset (embeddings_balanced.jsonl) with the same schema as embeddings.jsonl.

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable
from collections import Counter

EMOTIONS = [
    "Positive", "Negative", "Happiness", "Delight", "Inspiring", "Surprise", 
    "Compassion", "Fear", "Sadness", "Disgust", "Anger", "Ironic", "Political", 
    "Interesting", "Understandable", "Offensive", "Funny"
]
IDX = {name: i for i, name in enumerate(EMOTIONS)}
RARE_LABELS = {"Delight", "Inspiring", "Disgust", "Political", "Offensive", "Funny"}

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] Invalid JSON at {path}:{ln}: {e}")
    return out

def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def text_fingerprint(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def compute_stats(emb_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(emb_rows)
    per_label = [0]*len(EMOTIONS)
    per_lang = {"pl":[0]*len(EMOTIONS), "en":[0]*len(EMOTIONS)}
    totals_lang = Counter()
    for r in emb_rows:
        labels = r["labels"]
        lg = r["lang"]
        totals_lang[lg] += 1
        for i,v in enumerate(labels):
            if v == 1:
                per_label[i] += 1
                per_lang[lg][i] += 1

    def shares(cnts, denom):
        return {EMOTIONS[i]: (cnts[i]/denom if denom else 0.0) for i in range(len(EMOTIONS))}

    return {
        "total_records": total,
        "global_share": shares(per_label, total),
        "share_per_lang": {
            "pl": shares(per_lang["pl"], totals_lang["pl"]),
            "en": shares(per_lang["en"], totals_lang["en"]),
        }
    }

def group_by_lang_persona(rows: List[Dict[str, Any]]) -> Dict[Tuple[str,str], List[Dict[str, Any]]]:
    buckets: Dict[Tuple[str,str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["lang"], r["persona_id"])
        buckets.setdefault(key, []).append(r)
    return buckets

def balance_group(
    rows: List[Dict[str, Any]],
    opinions_map: Dict[Tuple[str,str], str],
    max_negative_share: float,
    duplicate_cap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Operuje na jednej grupie (lang, persona). Zwraca (kept_full_rows, removed_rows_with_reason).
    Reguły:
      - keep all rows that contain any rare label (RARE_LABELS)
      - dedup by text fingerprint up to duplicate_cap per fingerprint
      - negative cap: if share(Negative==1) exceeds cap, downsample negatives
        preferring to remove "Negative-only" (sum(labels)==1); then low-rarity
    """
    if not rows:
        return [], []
    lang = rows[0]["lang"]

    rare_indices = {IDX[l] for l in RARE_LABELS}
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    # 1) split rare vs non-rare
    rare_rows, nonrare_rows = [], []
    for r in rows:
        labels = r["labels"]
        is_rare = any(labels[i]==1 for i in rare_indices)
        (rare_rows if is_rare else nonrare_rows).append(r)
    kept.extend(rare_rows)

    # 2) dedup non-rare by text fingerprint
    def fp_for(r):
        txt = opinions_map.get((r["opinion_id"], r["lang"]), "")
        return text_fingerprint(txt) if txt else None

    fp_count: Dict[str,int] = {}
    deduped = []
    for r in nonrare_rows:
        fp = fp_for(r)
        if not fp:
            deduped.append(r)
            continue
        c = fp_count.get(fp, 0)
        if c < duplicate_cap:
            fp_count[fp] = c + 1
            deduped.append(r)
        else:
            removed.append({**r, "reason":"duplicate_cap"})

    # 3) apply Positive cap to combined pool
    pool = kept + deduped
    if not pool:
        return kept, removed

    pos_idx = IDX["Negative"]
    total = len(pool)
    pos = sum(1 for r in pool if r["labels"][pos_idx]==1)
    current_share = pos/total if total else 0.0
    if current_share <= max_negative_share:
        return pool, removed

    # need to remove some negatives
    max_pos = int(max_negative_share * total)
    to_remove = max(0, pos - max_pos)
    negatives = [r for r in pool if r["labels"][pos_idx]==1]

    # priority 1: negative-only
    p_only = [r for r in negatives if sum(r["labels"]) == 1]
    # priority 2: low rarity score
    def rarity_score(r):  # smaller first
        return sum(r["labels"][i] for i in rare_indices)
    p_lowrare = sorted([r for r in negatives if r not in p_only], key=rarity_score)

    keep_ids = {id(r) for r in pool}
    def remove_from(cands, needed):
        removed_local = []
        for r in cands:
            if needed <= 0:
                break
            rid = id(r)
            if rid in keep_ids:
                keep_ids.remove(rid)
                removed_local.append(r)
                needed -= 1
        return removed_local, needed

    rem1, to_remove = remove_from(p_only, to_remove)
    rem2, to_remove = remove_from(p_lowrare, to_remove)
    removed_now = rem1 + rem2

    final_kept = [r for r in pool if id(r) in keep_ids]
    removed.extend([{**r, "reason":"negative_cap"} for r in removed_now])
    return final_kept, removed

def main():
    parser = argparse.ArgumentParser(description="Balance embeddings and output a full balanced dataset.")
    parser.add_argument("--opinions", default="dataset/opinions.jsonl")
    parser.add_argument("--embeddings", default="data/working/embeddings.jsonl")
    parser.add_argument("--outdir", default="data/working")
    parser.add_argument("--max-negative-share", type=float, default=0.40, help="Cap share of Negative==1 per (lang×persona)")
    parser.add_argument("--duplicate-cap", type=int, default=3, help="Max items per identical text fingerprint per (lang×persona)")
    parser.add_argument("--out-file", default="embeddings_balanced.jsonl", help="Output filename for balanced full dataset")
    args = parser.parse_args()

    opinions = read_jsonl(Path(args.opinions))
    emb = read_jsonl(Path(args.embeddings))

    # Validate embeddings schema
    for r in emb:
        if not isinstance(r.get("labels"), list) or len(r["labels"]) != len(EMOTIONS):
            raise SystemExit(f"[ERROR] Invalid labels length for {r.get('opinion_id')} {r.get('lang')} {r.get('persona_id')}")

    # map (opinion_id, lang) -> text for dedup
    opinions_map: Dict[Tuple[str,str], str] = {}
    for o in opinions:
        opinions_map[(o["opinion_id"], o["lang"])] = o["text"]

    # stats before
    stats_before = compute_stats(emb)

    # group by (lang, persona)
    buckets = group_by_lang_persona(emb)

    kept_all: List[Dict[str, Any]] = []
    removed_all: List[Dict[str, Any]] = []

    for (lg, pid), rows in buckets.items():
        kept, removed = balance_group(
            rows=rows,
            opinions_map=opinions_map,
            max_negative_share=args.max_negative_share,
            duplicate_cap=args.duplicate_cap,
        )
        kept_all.extend(kept)
        removed_all.extend(removed)
        print(f"→ balanced ({lg}, {pid}): kept={len(kept)} removed={len(removed)}")

    # stats after
    stats_after = compute_stats(kept_all)

    outdir = Path(args.outdir)
    # 🔑 NOWE: pełny zbalansowany dataset (taki sam schemat jak embeddings.jsonl)
    balanced_path = outdir / args.out_file
    write_jsonl(kept_all, balanced_path)

    # zachowujemy też poprzednie artefakty (dla audytu)
    write_jsonl(
        [{"opinion_id": r["opinion_id"], "lang": r["lang"], "persona_id": r["persona_id"]} for r in kept_all],
        outdir / "kept_records.jsonl"
    )
    write_jsonl(removed_all, outdir / "removed_records.jsonl")
    write_json({
        "emotions": EMOTIONS,
        "rare_labels": sorted(RARE_LABELS),
        "params": {
            "max_negative_share": args.max_negative_share,
            "duplicate_cap": args.duplicate_cap
        },
        "before": stats_before,
        "after": stats_after,
        "counts": {
            "embeddings_total": len(emb),
            "kept": len(kept_all),
            "removed": len(removed_all)
        },
        "outputs": {
            "balanced_embeddings": str(balanced_path)
        }
    }, outdir / "balance_report.json")

    print("✅ Balance done.")
    print(f"  embeddings_balanced.jsonl: {balanced_path}")
    print(f"  kept_records.jsonl:        {outdir/'kept_records.jsonl'}")
    print(f"  removed_records.jsonl:     {outdir/'removed_records.jsonl'}")
    print(f"  balance_report.json:       {outdir/'balance_report.json'}")

if __name__ == "__main__":
    main()