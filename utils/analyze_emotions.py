# utils/analyze_emotions.py
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

EMOTIONS = [
    "Positive", "Negative", "Happiness", "Inspiring", "Surprise", 
    "Compassion", "Sadness", "Anger", "Ironic", "Interesting", 
    "Understandable", "Offensive"
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def pretty_table(title: str, counts: Counter, total: int):
    print(f"\n📊 {title}")
    print(f"{'Emotion':<15} {'Count':>10} {'Share':>9}")
    for e in EMOTIONS:
        cnt = counts.get(e, 0)
        share = (cnt / total * 100.0) if total else 0.0
        print(f"{e:<15} {cnt:>10} {share:>8.2f}%")
    print(f"{'-'*36}\nTotal records: {total}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze emotion distribution (global + per language)")
    parser.add_argument("--embeddings", default="data/working/embeddings.jsonl", help="Path to embeddings.jsonl")
    args = parser.parse_args()

    path = Path(args.embeddings)
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing file: {path}")

    global_counts = Counter()
    total_global = 0

    per_lang_counts = defaultdict(Counter)
    per_lang_totals = Counter()

    for row in read_jsonl(path):
        labels = row.get("labels")
        lang = row.get("lang", "unknown")
        if not isinstance(labels, list) or len(labels) != len(EMOTIONS):
            continue  # pomiń błędne wiersze

        # zlicz globalnie
        for i, v in enumerate(labels):
            if v == 1:
                global_counts[EMOTIONS[i]] += 1
                per_lang_counts[lang][EMOTIONS[i]] += 1
        total_global += 1
        per_lang_totals[lang] += 1

    # Wydruk
    pretty_table("GLOBAL (all languages)", global_counts, total_global)
    for lang in sorted(per_lang_totals):
        pretty_table(f"LANG = {lang}", per_lang_counts[lang], per_lang_totals[lang])

if __name__ == "__main__":
    main()