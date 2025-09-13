# utils/filter_emotions.py
import json
import argparse
from pathlib import Path

ORIGINAL_EMOTIONS = [
    "Positive", "Negative", "Happiness", "Delight", "Inspiring", "Surprise", 
    "Compassion", "Fear", "Sadness", "Disgust", "Anger", "Ironic", "Political", 
    "Interesting", "Understandable", "Offensive", "Funny"
]

EMOTIONS_TO_DROP = ["Delight", "Disgust", "Funny", "Fear", "Political"]

NEW_EMOTIONS = [
    "Positive", "Negative", "Happiness", "Inspiring", "Surprise", 
    "Compassion", "Sadness", "Anger", "Ironic", "Interesting", 
    "Understandable", "Offensive"
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Remove selected emotions from embeddings.jsonl")
    parser.add_argument("--embeddings", default="data/working/embeddings.jsonl")
    parser.add_argument("--output", default="data/working/embeddings_filtered_delight.jsonl")
    parser.add_argument("--drop", nargs="+", required=True, help="List of emotions to drop")
    args = parser.parse_args()

    drop_set = set(args.drop)
    keep_indices = [i for i, e in enumerate(ORIGINAL_EMOTIONS) if e not in drop_set]
    new_emotions = [ORIGINAL_EMOTIONS[i] for i in keep_indices]

    print(f"🔧 Usuwam emocje: {drop_set}")
    print(f"🆕 Zostawiam emocje: {new_emotions}")

    out = []
    for row in read_jsonl(Path(args.embeddings)):
        labels = row.get("labels")
        if not isinstance(labels, list) or len(labels) != len(ORIGINAL_EMOTIONS):
            continue
        new_labels = [labels[i] for i in keep_indices]
        row["labels"] = new_labels
        out.append(row)

    write_jsonl(out, Path(args.output))
    print(f"✅ Zapisano {len(out)} rekordów do {args.output}")
    print(f"🆕 Nowa długość wektora labels: {len(new_emotions)}")

if __name__ == "__main__":
    main()
