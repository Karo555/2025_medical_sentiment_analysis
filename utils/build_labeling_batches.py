# src/labeling_build_batches.py
# Step 2: Deterministic batching + prompt snapshots for labeling (PL/EN)
import argparse
import json
from pathlib import Path
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

def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def chunkify(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

def personas_by_lang(personas_all: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"pl": [], "en": []}
    for p in personas_all:
        lg = p.get("lang")
        if lg in out:
            out[lg].append(p)
    # stabilna kolejność: numerycznie po id (person_1, person_2, ..., person_10)
    def sort_key(x):
        person_id = x.get("id", "")
        # Extract number from person_X format and convert to int
        try:
            if "_" in person_id:
                number = int(person_id.split("_")[1])
                return number
        except (ValueError, IndexError):
            pass
        # Fallback to original id for non-standard formats
        return person_id
    
    out["pl"] = sorted(out["pl"], key=sort_key)
    out["en"] = sorted(out["en"], key=sort_key)
    return out

# ---------- Prompt builders ----------

def build_prompt_pl(personas: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> str:
    personas_block = []
    for p in personas:
        personas_block.append(
            f'- {p["id"]}: {p.get("name","")} '
            f'| Wrażliwość: {p.get("sensitivity","")} '
            f'| Wartości: {p.get("values","")}'
        )
    personas_txt = "\n".join(personas_block)

    emotions_txt = ", ".join(EMOTIONS)

    example = (
        'INPUT (opinie):\n'
        '[{"opinion_id":"op_demo","lang":"pl","text":"Lekarz był miły i wszystko dokładnie wyjaśnił."}]\n'
        'OUTPUT (JSON, tylko tablica):\n'
        '[{"opinion_id":"op_demo","lang":"pl","results":[\n'
        '  {"persona_id":"person_1","labels":[1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,0,0]},\n'
        '  {"persona_id":"person_2","labels":[0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0]}\n'
        ']}]\n'
    )

    return (
        "Jesteś systemem do etykietowania emocji w opiniach medycznych.\n"
        "ZADANIE: Dla KAŻDEJ z poniższych opinii w języku polskim oceń emocje z perspektywy KAŻDEJ persony.\n"
        "Nie tłumacz tekstów. Nie zmieniaj treści. Zwróć wyłącznie JSON.\n\n"
        "PERSONY (użyj wszystkich):\n"
        f"{personas_txt}\n\n"
        "ETYKIETY (kolejność i znaczenie, 0/1 każda):\n"
        f"{emotions_txt}\n\n"
        "WYJŚCIE:\n"
        "Zwróć TYLKO tablicę JSON, bez komentarzy i bez code fences. Każdy element tablicy odpowiada jednej opinii i ma postać:\n"
        '{ "opinion_id": str, "lang": "pl", "results": [ { "persona_id": str, "labels": [17 liczb 0/1 w dokładnej kolejności etykiet] }, ... ] }\n'
        "Liczba elementów w 'results' MUSI równać się liczbie person (10). Długość 'labels' MUSI wynosić 17.\n\n"
        f"{example}"
    )

def build_prompt_en(personas: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> str:
    personas_block = []
    for p in personas:
        personas_block.append(
            f'- {p["id"]}: {p.get("name","")} '
            f'| Sensitivity: {p.get("sensitivity","")} '
            f'| Values: {p.get("values","")}'
        )
    personas_txt = "\n".join(personas_block)

    emotions_txt = ", ".join(EMOTIONS)

    example = (
        'INPUT (opinions):\n'
        '[{"opinion_id":"op_demo","lang":"en","text":"The doctor was kind and explained everything clearly."}]\n'
        'OUTPUT (JSON, array only):\n'
        '[{"opinion_id":"op_demo","lang":"en","results":[\n'
        '  {"persona_id":"person_1","labels":[1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,0,0]},\n'
        '  {"persona_id":"person_2","labels":[0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0]}\n'
        ']}]\n'
    )

    return (
        "You are an emotion labeling system for medical reviews.\n"
        "TASK: For EACH English review below, evaluate emotions from the perspective of EACH persona.\n"
        "Do not translate. Do not modify content. Return JSON only.\n\n"
        "PERSONAS (use all):\n"
        f"{personas_txt}\n\n"
        "LABELS (order, binary 0/1 each):\n"
        f"{emotions_txt}\n\n"
        "OUTPUT:\n"
        "Return ONLY a JSON array. Each element corresponds to one opinion:\n"
        '{ "opinion_id": str, "lang": "en", "results": [ { "persona_id": str, "labels": [17 numbers 0/1 in the exact label order] }, ... ] }\n'
        "The number of elements in 'results' MUST equal the number of personas (10). 'labels' MUST have length 17.\n\n"
        f"{example}"
    )

def build_batches_and_prompts(
    opinions_path: Path,
    personas_path: Path,
    out_dir: Path,
    batch_size: int = 15,
) -> Dict[str, Any]:
    opinions = read_jsonl(opinions_path)
    personas_all = read_jsonl(personas_path)

    # Split by lang
    opinions_pl = [o for o in opinions if o.get("lang") == "pl"]
    opinions_en = [o for o in opinions if o.get("lang") == "en"]
    personas = personas_by_lang(personas_all)

    batches_dir = out_dir / "labeling_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    # deterministic batches (PL then EN)
    batches_pl = chunkify(opinions_pl, batch_size)
    batches_en = chunkify(opinions_en, batch_size)

    index: Dict[str, Any] = {
        "emotions": EMOTIONS,
        "batch_size": batch_size,
        "counts": {
            "opinions_pl": len(opinions_pl),
            "opinions_en": len(opinions_en),
            "batches_pl": len(batches_pl),
            "batches_en": len(batches_en),
            "batches_total": len(batches_pl) + len(batches_en),
        },
        "batches": []
    }

    batch_counter = 0

    # helper to write one batch
    def write_batch(items: List[Dict[str, Any]], lang: str):
        nonlocal batch_counter
        if not items:
            return
        batch_counter += 1
        bid = f"{batch_counter:04d}"

        # request
        req_path = batches_dir / f"batch_{bid}_request.jsonl"
        write_jsonl(items, req_path)

        # personas snapshot
        pers = personas[lang]
        if not pers or len(pers) != 10:
            print(f"[WARN] Personas for lang={lang} count={len(pers)} (expected 10).")
        pers_path = batches_dir / f"batch_{bid}_personas_snapshot.json"
        write_json(pers, pers_path)

        # prompt
        prompt = build_prompt_pl(pers, items) if lang == "pl" else build_prompt_en(pers, items)
        prompt_path = batches_dir / f"batch_{bid}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        index["batches"].append({
            "batch_id": bid,
            "lang": lang,
            "count": len(items),
            "request": str(req_path),
            "personas_snapshot": str(pers_path),
            "prompt": str(prompt_path),
        })

        print(f"📦 Prepared batch #{bid} lang={lang} items={len(items)}")
        print(f"  ↪ {req_path.name}")
        print(f"  ↪ {pers_path.name}")
        print(f"  ↪ {prompt_path.name}")

    # PL first, then EN (warstwowo jak w opinions)
    for chunk in batches_pl:
        write_batch(chunk, "pl")
    for chunk in batches_en:
        write_batch(chunk, "en")

    # write index
    write_json(index, batches_dir / "batches_index.json")
    print("✅ Done. Index written to:", batches_dir / "batches_index.json")
    return index

def main():
    parser = argparse.ArgumentParser(description="Build labeling batches and prompt snapshots (step 2).")
    parser.add_argument("--opinions", default="dataset/opinions.jsonl", help="Path to opinions.jsonl")
    parser.add_argument("--personas", default="dataset/personas.jsonl", help="Path to personas.jsonl")
    parser.add_argument("--outdir", default="data/working", help="Output base dir")
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    build_batches_and_prompts(
        opinions_path=Path(args.opinions),
        personas_path=Path(args.personas),
        out_dir=Path(args.outdir),
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()