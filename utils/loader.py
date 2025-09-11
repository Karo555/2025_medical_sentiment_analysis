import csv
import json
from pathlib import Path


def load_opinions_tsv(path: str) -> list[dict]:
    """
    Wczytuje opinie z pliku TSV (meta.tsv).
    Zwraca listę słowników w formacie:
    [{"opinion_id": "op_000001", "text": "...", "lang": "pl"}, ...]
    """
    opinions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            category, text = row[0], row[1]
            if category.strip().lower() != "medycyna":
                continue
            opinions.append({"text": text.strip(), "lang": "pl"})
    return opinions


def load_opinions_json(path: str) -> list[dict]:
    """
    Wczytuje opinie z pliku JSON (opinions.json).
    Zwraca listę słowników w formacie:
    [{"opinion_id": "op_000001", "text": "...", "lang": "pl"}, ...]
    """
    opinions = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Obsłuż przypadek listy lub pojedynczych rekordów
    if isinstance(data, dict):
        data = [data]

    for entry in data:
        text = entry.get("tekst", "")
        if text is not None:
            text = text.strip()
            if text:
                opinions.append({"text": text, "lang": "pl"})
    return opinions


def save_jsonl(data: list[dict], path: str) -> None:
    """
    Zapisuje listę słowników jako plik JSONL (jedna linia = jeden rekord).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
