import sys
from itertools import count
from pathlib import Path

# Add the root directory to the Python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.loader import load_opinions_tsv, load_opinions_json, save_jsonl

tsv_opinions = load_opinions_tsv("data/raw/meta.tsv")
json_opinions = load_opinions_json("data/raw/opinions.json")

# Połącz listy i nadaj globalne ID
all_opinions = []
counter = count(1)

for op in tsv_opinions + json_opinions:
    opinion_id = f"op_{next(counter):06d}"
    all_opinions.append({
        "opinion_id": opinion_id,
        "text": op["text"],
        "lang": op["lang"]
    })

# Zapisz do JSONL
save_jsonl(all_opinions, "data/working/raw_opinions.jsonl")