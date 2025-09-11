# src/translator.py
# Async translation + anonymization + batching using OpenAI SDK (against DeepSeek-compatible endpoint)
# Resume-safe + deterministic batching + final consolidation

import os
import re
import json
import orjson
import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"  # OpenAI-compatible
DEFAULT_MODEL = "deepseek-chat"

# ---------- IO utils ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(records: List[Dict[str, Any]], path: Path, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def ensure_dirs(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batches_dir = out_path.parent / "translation_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    return batches_dir

# ---------- Prompt builder ----------

def build_prompt_for_translation(batch_items: List[Dict[str, Any]]) -> str:
    """
    Jeden prompt: tłumaczenie PL->EN + anonimizacja (bez wymyślania nazw).
    """
    instruction = (
    "Jesteś uważnym tłumaczem w dziedzinie medycyny.\n"
    "\n"
    "ZADANIE:\n"
    "1. Dla każdej opinii w języku polskim wykonaj TŁUMACZENIE na naturalny, płynny język angielski, zachowując znaczenie, ton i emocje.\n"
    "2. WYKONAJ ANONIMIZACJĘ zgodnie z zasadami poniżej.\n"
    "\n"
    "ZASADY ANONIMIZACJI:\n"
    "- Jeśli w tekście występuje imię i nazwisko osoby, zamień je na neutralny placeholder:\n"
    "  • 'Jan Kowalski' (mężczyzna)\n"
    "  • 'Anna Kowalska' (kobieta)\n"
    "- Jeśli występuje tylko imię → zamień na 'Jan' lub 'Anna'.\n"
    "- Jeśli występuje tylko nazwisko → zamień na 'Pan Kowalski' lub 'Pani Kowalska'.\n"
    "- W miarę możliwości zachowaj poprawną odmianę w kontekście zdania.\n"
    "- NIE wymyślaj imion/nazwisk, jeśli w tekście ich nie ma.\n"
    "- Nazwy klinik, szpitali i marek zostaw w oryginale (chyba że to nazwisko osoby prywatnej).\n"
    "- Nie dodawaj ani nie usuwaj treści.\n"
    "\n"
    "WYJŚCIE:\n"
    "- Zwróć ŚCISŁĄ tablicę JSON (tylko tablicę, bez komentarzy, bez code fences).\n"
    "- Każdy element:\n"
    "{ \"opinion_id\": str, \"lang\": \"en\", \"text\": str, \"redacted\": int }\n"
    "- 'redacted' = liczba dokonanych zamian imion/nazwisk (0 jeśli brak zmian).\n"
    "\n"
    "WAŻNE:\n"
    "- Zachowaj dokładnie format JSON.\n"
    "- Upewnij się, że 'text' zawiera zanonimizowaną wersję tłumaczenia.\n"
    "- Jeżeli nie było nazw osobowych, 'redacted' = 0.\n"
    "\n"
    "PRZYKŁAD:\n"
    "INPUT:\n"
    "[ {\"opinion_id\": \"op_test1\", \"text\": \"Byłam u dr Kowalskiego i było super.\"} ]\n"
    "OUTPUT:\n"
    "[ {\"opinion_id\": \"op_test1\", \"lang\": \"en\", \"text\": \"I went to Dr. Jan Kowalski and it was great.\", \"redacted\": 1} ]\n"
)
    payload = [{"opinion_id": it["opinion_id"], "text": it["text"]} for it in batch_items]
    return instruction + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

# ---------- OpenAI SDK client (to DeepSeek) ----------

class OpenAICompatClient:
    """
    OpenAI SDK klient wskazujący na DeepSeek (OpenAI-compatible).
    """
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, timeout: float = 120.0):
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def close(self):
        await self._client.close()

    async def translate_batch(self, items: List[Dict[str, Any]], retries: int = 4, backoff_base: float = 1.5) -> List[Dict[str, Any]]:
        """
        Jeden wsad -> jedno wywołanie chat.completions.
        Zwraca listę {opinion_id, lang:'en', text, redacted}.
        """
        prompt = build_prompt_for_translation(items)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that strictly follows output format instructions."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(retries + 1):
            try:
                completion: ChatCompletion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )
                content = (completion.choices[0].message.content or "").strip()
                parsed = _extract_json_array(content)
                if not isinstance(parsed, list):
                    raise ValueError("Model did not return a JSON array.")
                cleaned = []
                for el in parsed:
                    if not isinstance(el, dict):
                        continue
                    if "opinion_id" in el and "text" in el:
                        cleaned.append({
                            "opinion_id": el["opinion_id"],
                            "lang": "en",
                            "text": el["text"],
                            "redacted": int(el.get("redacted", 0) or 0)
                        })
                if not cleaned:
                    raise ValueError("Empty/invalid JSON array from model.")
                return cleaned
            except Exception:
                if attempt == retries:
                    raise
                sleep_s = backoff_base ** attempt + (0.25 * attempt)
                await asyncio.sleep(sleep_s)
        return []

def _extract_json_array(s: str) -> Any:
    """
    Wyciąga pierwszą tablicę JSON ze stringa (usuwając ewentualne code fences).
    """
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in response.")
    segment = s[start:end+1]
    return json.loads(segment)

# ---------- Batching helpers (deterministic) ----------

def make_batches(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for r in rows:
        current.append(r)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches

def clean_path_for(batches_dir: Path, idx: int) -> Path:
    return batches_dir / f"batch_{idx:04d}_clean.jsonl"

def request_path_for(batches_dir: Path, idx: int) -> Path:
    return batches_dir / f"batch_{idx:04d}_request.jsonl"

def response_path_for(batches_dir: Path, idx: int) -> Path:
    return batches_dir / f"batch_{idx:04d}_response.jsonl"

def prompt_path_for(batches_dir: Path, idx: int) -> Path:
    return batches_dir / f"batch_{idx:04d}_prompt.txt"

def is_batch_completed(batches_dir: Path, idx: int, expected_len: int) -> bool:
    p = clean_path_for(batches_dir, idx)
    if not p.exists():
        return False
    try:
        lines = read_jsonl(p)
        return len(lines) == expected_len
    except Exception:
        return False

# ---------- Single-batch processing ----------

async def process_batch(
    client: OpenAICompatClient,
    batch_items: List[Dict[str, Any]],
    batches_dir: Path,
    batch_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Przetwarza jeden batch: zapisuje snapshoty, woła API, waliduje,
    zwraca czyste EN rekordy + porażki + statystyki.
    """
    print(f"📦 Processing batch #{batch_idx} with {len(batch_items)} items...")
    
    # request snapshot
    write_jsonl(batch_items, request_path_for(batches_dir, batch_idx))
    print(f"  ↪ Saved request snapshot to {request_path_for(batches_dir, batch_idx).name}")

    # prompt snapshot
    prompt = build_prompt_for_translation(batch_items)
    with open(prompt_path_for(batches_dir, batch_idx), "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  ↪ Saved prompt to {prompt_path_for(batches_dir, batch_idx).name}")

    # API call
    print(f"  ↪ Calling DeepSeek API for batch #{batch_idx}...")
    try:
        responses = await client.translate_batch(batch_items)
        print(f"  ✅ API call successful, received {len(responses)} responses")
    except Exception as e:
        # snapshot błędu zamiast response
        print(f"  ❌ API call failed: {str(e)}")
        write_jsonl([{"error": str(e)}], response_path_for(batches_dir, batch_idx))
        failures = [{"opinion_id": it["opinion_id"], "error": str(e), "attempts": 1} for it in batch_items]
        print(f"  ↪ Saved error snapshot to {response_path_for(batches_dir, batch_idx).name}")
        return [], failures, {"batch": batch_idx, "ok": 0, "failed": len(batch_items)}

    # raw response snapshot (parsowana lista obiektów)
    write_jsonl(responses, response_path_for(batches_dir, batch_idx))
    print(f"  ↪ Saved raw responses to {response_path_for(batches_dir, batch_idx).name}")

    # validate/clean (bez 'redacted' w finalnym dataset)
    print(f"  ↪ Validating responses...")
    clean = []
    failures = []
    for resp in responses:
        if not isinstance(resp, dict) or "opinion_id" not in resp or "text" not in resp:
            failures.append({"opinion_id": resp.get("opinion_id"), "error": "invalid_response_schema", "attempts": 1})
            continue
        clean.append({
            "opinion_id": resp["opinion_id"],
            "text": resp["text"],
            "lang": "en"
        })

    # clean snapshot (checkpoint zakończonego batcha)
    write_jsonl(clean, clean_path_for(batches_dir, batch_idx))
    print(f"  ↪ Saved clean translations to {clean_path_for(batches_dir, batch_idx).name}")

    # redaction flags (append) – tylko do audytu
    redaction_path = batches_dir.parent / "translation_redaction_flags.jsonl"
    flags = [{
        "opinion_id": r.get("opinion_id"),
        "names_detected": (r.get("redacted", 0) or 0) > 0,
        "redacted_count": r.get("redacted", 0) or 0
    } for r in responses]
    write_jsonl(flags, redaction_path, mode="a")
    print(f"  ↪ Updated redaction flags in {redaction_path.name}")
    
    redacted_count = sum(1 for f in flags if f["names_detected"])
    if redacted_count > 0:
        print(f"  ℹ️ Detected personal names in {redacted_count} opinions")

    stats = {"batch": batch_idx, "ok": len(clean), "failed": len(failures)}
    print(f"✅ Batch #{batch_idx} complete: {len(clean)} ok, {len(failures)} failed")
    return clean, failures, stats

# ---------- Consolidation ----------

def consolidate_translations(
    raw_opinions_path: Path,
    batches_dir: Path,
    out_path: Path
) -> None:
    """
    Buduje finalny translated_opinions.jsonl z:
    - PL z raw_opinions.jsonl
    - EN ze wszystkich batch_XXXX_clean.jsonl (posortowanych po idx)
    Zapis atomowy (tmp -> replace).
    """
    print(f"📊 Consolidating translations to {out_path.name}...")
    
    # 1) wczytaj PL z raw
    print(f"  ↪ Loading original PL opinions from {raw_opinions_path.name}")
    raw_rows = read_jsonl(raw_opinions_path)
    pl_rows = [r for r in raw_rows if r.get("lang") == "pl"]
    print(f"  ↪ Found {len(pl_rows)} PL opinions")

    # 2) zbierz EN ze wszystkich clean'ów w kolejności idx
    clean_files = sorted(batches_dir.glob("batch_*_clean.jsonl"))
    print(f"  ↪ Found {len(clean_files)} completed batch files")
    en_rows: List[Dict[str, Any]] = []
    for cf in clean_files:
        batch_rows = read_jsonl(cf)
        en_rows.extend(batch_rows)
        print(f"  ↪ Added {len(batch_rows)} translations from {cf.name}")

    # 3) zbuduj wynik: PL + EN
    print(f"  ↪ Building final output with {len(pl_rows)} PL + {len(en_rows)} EN opinions...")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    # nadpisz tmp
    if tmp_path.exists():
        tmp_path.unlink()
    # zapis PL
    write_jsonl(pl_rows, tmp_path, mode="w")
    # dopisz EN
    write_jsonl(en_rows, tmp_path, mode="a")
    # atomowy replace
    os.replace(tmp_path, out_path)
    print(f"✅ Successfully created {out_path.name} with {len(pl_rows) + len(en_rows)} total opinions")

# ---------- Orchestrator ----------

async def translate_file_async(
    in_path: Path,
    out_path: Path,
    batch_size: int = 20,
    concurrency: int = 4,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    resume: bool = False,
    from_batch: Optional[int] = None,
    to_batch: Optional[int] = None,
    rebuild_output_only: bool = False,
):
    print(f"🚀 Starting translation pipeline...")
    print(f"  • Input: {in_path}")
    print(f"  • Output: {out_path}")
    print(f"  • Model: {model}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Concurrency: {concurrency}")
    print(f"  • Resume mode: {'enabled' if resume else 'disabled'}")
    
    batches_dir = ensure_dirs(out_path)
    print(f"  • Batches directory: {batches_dir}")
    
    data = read_jsonl(in_path)
    print(f"  • Loaded {len(data)} records from input file")

    # 1) bierzemy tylko PL; deterministyczne batchowanie
    pl_rows = [r for r in data if r.get("lang") == "pl"]
    print(f"  • Found {len(pl_rows)} Polish records to translate")
    
    batches = make_batches(pl_rows, batch_size)
    print(f"  • Created {len(batches)} batches of size {batch_size}")

    # Tylko konsolidacja (bez API)
    if rebuild_output_only:
        print("ℹ️ Running in rebuild-output-only mode (no API calls)")
        consolidate_translations(in_path, batches_dir, out_path)
        return

    # 2) zbuduj zakres batchy do przetworzenia
    total_batches = len(batches)
    start_idx = 1 if from_batch is None else max(1, from_batch)
    end_idx = total_batches if to_batch is None else min(total_batches, to_batch)
    print(f"  • Processing batch range: {start_idx} to {end_idx} (of {total_batches} total)")

    # 3) klient
    print(f"  • Initializing API client to {base_url}")
    client = OpenAICompatClient(api_key=api_key, base_url=base_url, model=model)

    sem = asyncio.Semaphore(concurrency)
    all_failures: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []

    async def runner(idx: int, items: List[Dict[str, Any]]):
        expected_len = len(items)
        # resume: pomiń ukończone batch'e
        if resume and is_batch_completed(batches_dir, idx, expected_len):
            print(f"⏭️ Skipping batch #{idx} (already completed with {expected_len} items)")
            stats.append({"batch": idx, "ok": expected_len, "failed": 0, "skipped": True})
            return
        async with sem:
            clean, fails, s = await process_batch(client, items, batches_dir, idx)
            all_failures.extend(fails)
            stats.append(s)

    tasks = []
    for i in range(start_idx, end_idx + 1):
        tasks.append(asyncio.create_task(runner(i, batches[i-1])))

    t0 = time.time()
    print(f"⏱️ Starting batch processing at {time.strftime('%H:%M:%S')}")
    try:
        await asyncio.gather(*tasks)
        print(f"✅ All batches processed successfully")
    except Exception as e:
        print(f"❌ Error during batch processing: {str(e)}")
        raise
    finally:
        print(f"  ↪ Closing API client connection")
        await client.close()
    t1 = time.time()
    elapsed = t1 - t0
    print(f"⏱️ Processing completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # 4) zapis błędów (jeśli są)
    failures_path = out_path.parent / "translation_failures.jsonl"
    if all_failures:
        print(f"⚠️ Writing {len(all_failures)} failed translations to {failures_path.name}")
        write_jsonl(all_failures, failures_path)

    # 5) konsolidacja do pliku wynikowego
    consolidate_translations(in_path, batches_dir, out_path)

    # 6) audit
    completed = sum(1 for s in stats if s.get("ok", 0) > 0 or s.get("skipped"))
    audit = {
        "model": model,
        "base_url": base_url,
        "total_inputs_pl": len(pl_rows),
        "total_batches": len(batches),
        "executed_batches": len(tasks),
        "completed_batches": completed,
        "failed_records": len(all_failures),
        "batch_size": batch_size,
        "concurrency": concurrency,
        "resume": resume,
        "from_batch": start_idx,
        "to_batch": end_idx,
        "elapsed_sec": round(t1 - t0, 3),
    }
    audit_path = out_path.parent / "translation_audit.json"
    write_json_atomic(audit, audit_path)
    print(f"📊 Saved audit information to {audit_path.name}")
    print(f"🏁 Translation pipeline completed successfully!")

# ---------- CLI ----------

def main():
    print("🔄 Medical Opinion Translator")
    print("----------------------------")
    
    load_dotenv()  # loads .env if present
    print(f"ℹ️ Loaded environment variables from .env (if present)")
    
    parser = argparse.ArgumentParser(description="Async translation + anonymization via OpenAI SDK (DeepSeek-compatible) with resume & consolidation")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to data/interim/raw_opinions.jsonl")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to data/interim/translated_opinions.jsonl")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    # akceptujemy oba: DEEPSEEK_API_KEY lub OPENAI_API_KEY
    default_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    parser.add_argument("--api-key", type=str, default=default_key)
    # Resume / range / consolidate-only
    parser.add_argument("--resume", action="store_true", help="Skip already completed batches (based on *_clean.jsonl)")
    parser.add_argument("--from-batch", type=int, default=None, help="Start processing from this batch index (1-based)")
    parser.add_argument("--to-batch", type=int, default=None, help="Stop processing at this batch index (1-based, inclusive)")
    parser.add_argument("--rebuild-output-only", action="store_true", help="Only rebuild translated_opinions.jsonl from existing clean batches (no API calls)")
    args = parser.parse_args()

    if not args.api_key and not args.rebuild_output_only:
        print("❌ Error: Missing API key")
        raise SystemExit("Missing API key. Put DEEPSEEK_API_KEY or OPENAI_API_KEY in .env or pass --api-key.")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    print(f"ℹ️ Parsed arguments successfully")

    asyncio.run(translate_file_async(
        in_path=in_path,
        out_path=out_path,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        resume=args.resume,
        from_batch=args.from_batch,
        to_batch=args.to_batch,
        rebuild_output_only=args.rebuild_output_only,
    ))

if __name__ == "__main__":
    main()