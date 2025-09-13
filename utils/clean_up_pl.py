# Async polish-only cleanup: spacing + anonymization (no translation), then merge back.
import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------- IO ----------

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

def write_jsonl(rows: List[Dict[str, Any]], path: Path, mode: str = "w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)

def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- Prompt ----------

INSTRUCTION_PL = (
    "Jesteś uważnym edytorem tekstów medycznych w języku polskim.\n"
    "TWOJE ZADANIE: Dla KAŻDEJ podanej opinii wykonaj WYŁĄCZNIE następujące kroki:\n"
    "  1) Popraw formatowanie i białe znaki:\n"
    "     - usuń podwójne spacje,\n"
    "     - usuń spacje przed znakami interpunkcyjnymi,\n"
    "     - zachowaj poprawne odstępy po kropkach i przecinkach.\n"
    "  2) Zanonimizuj dane osobowe (tylko imiona i nazwiska osób prywatnych):\n"
    "     - pełne imię i nazwisko → zastąp neutralnym placeholderem: „Jan Kowalski” (mężczyzna) lub „Anna Kowalska” (kobieta),\n"
    "     - samo imię → „Jan” lub „Anna”,\n"
    "     - samo nazwisko → „Pan Kowalski” lub „Pani Kowalska”,\n"
    "       ALE jeżeli nazwisko występuje po tytule zawodowym (np. „Dr. Kowalski” lub „Prof. Nowak”), zamień tylko nazwisko (nie dodawaj Pan/Pani).\n"
    "     - zawsze odmieniaj poprawnie w kontekście zdania.\n"
    "  3) Nie zmieniaj znaczenia ani kolejności zdań.\n"
    "  4) Nie tłumacz na inny język i nie dodawaj nowych informacji.\n"
    "  5) Nazwy instytucji, szpitali, klinik i marek pozostaw w oryginale (chyba że jednoznacznie identyfikują osobę prywatną).\n\n"
    "WAŻNE:\n"
    " - Zwróć wyłącznie poprawione teksty, nie dodawaj komentarzy ani wyjaśnień.\n"
    " - Dla każdej opinii zachowaj identyfikator opinion_id.\n\n"
    "FORMAT ODPOWIEDZI:\n"
    "Zwróć WYŁĄCZNIE OBIEKT JSON (bez code fences, bez komentarzy), o strukturze:\n"
    "{\n"
    '  "data": [\n'
    '    { "opinion_id": "op_000001", "lang": "pl", "text": "..." },\n'
    "    ...\n"
    "  ]\n"
    "}\n"
)

def build_payload(items: List[Dict[str, Any]]) -> str:
    payload = [{"opinion_id": it["opinion_id"], "lang": "pl", "text": it["text"]} for it in items]
    return "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

# ---------- Helpers ----------
def parse_json_payload(s: str) -> Dict[str, Any]:
    """
    Akceptuje:
      1) {"data": [ ... ]}
      2) [ ... ]  -> zamienia na {"data":[...]}
      3) {"anything": [ ... ]} -> wybiera pierwszą listę jako data
    """
    obj = json.loads(s.strip())
    if isinstance(obj, list):
        return {"data": obj}
    if isinstance(obj, dict):
        if isinstance(obj.get("data"), list):
            return obj
        # fallback: wybierz pierwszą listę w obiekcie
        for k, v in obj.items():
            if isinstance(v, list):
                return {"data": v}
    raise ValueError("No array found in JSON payload")


# ---------- Cleaner core (async) ----------

async def call_api(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Zwracasz wyłącznie poprawny JSON zgodny ze wskazanym schematem. Bez komentarzy i bez code fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""

async def process_batch(
    client: AsyncOpenAI,
    model: str,
    items: List[Dict[str, Any]],
    batches_dir: Path,
    bid: int,
    retries: int,
    backoff: float,
    max_tokens: int,
    save_prompt: bool = True,
) -> List[Dict[str, Any]]:
    print(f"🔄 Batch {bid:04d}: Starting processing ({len(items)} opinions)")
    req_path = batches_dir / f"batch_{bid:04d}_request.jsonl"
    prompt_path = batches_dir / f"batch_{bid:04d}_prompt.txt"
    input_array_path = batches_dir / f"batch_{bid:04d}_input.json"
    clean_path = batches_dir / f"batch_{bid:04d}_clean.jsonl"
    raw_path = batches_dir / f"batch_{bid:04d}_raw.txt"

    # 1) zapisz request
    write_jsonl(items, req_path)
    print(f"📄 Batch {bid:04d}: Request saved to {req_path.name}")

    # 2) wczytaj request i zbuduj INPUT
    req_items = read_jsonl(req_path)
    payload_array = [{"opinion_id": it["opinion_id"], "lang": "pl", "text": it["text"]} for it in req_items]

    # 3) snapshot wejścia jako jedna tablica JSON
    write_text(json.dumps(payload_array, ensure_ascii=False, indent=2), input_array_path)

    # 4) pełny prompt
    full_prompt = INSTRUCTION_PL + build_payload(req_items)
    if save_prompt:
        write_text(full_prompt, prompt_path)
        print(f"📝 Batch {bid:04d}: Prompt saved to {prompt_path.name}")

    # 5) call + retry
    attempt = 0
    while attempt <= retries:
        try:
            print(f"🚀 Batch {bid:04d}: Calling API (attempt {attempt+1}/{retries+1})")
            content = await call_api(client, model, full_prompt, max_tokens=max_tokens)
            # snapshot raw (nadpisanie na sukces – zostawiamy tylko przy błędach)
            if raw_path.exists():
                raw_path.unlink(missing_ok=True)

            obj = parse_json_payload(content)
            arr = obj.get("data", [])
            out: List[Dict[str, Any]] = []
            for el in arr:
                if not isinstance(el, dict):
                    continue
                oid = el.get("opinion_id")
                txt = el.get("text")
                lg = el.get("lang")
                if isinstance(oid, str) and isinstance(txt, str) and lg == "pl":
                    out.append({"opinion_id": oid, "lang": "pl", "text": txt})

            # dopasuj brakujące po opinion_id → fallback do oryginału
            want = {it["opinion_id"] for it in req_items}
            got = {o["opinion_id"] for o in out}
            missing = want - got
            if missing:
                base_map = {it["opinion_id"]: it["text"] for it in req_items}
                for oid in missing:
                    out.append({"opinion_id": oid, "lang": "pl", "text": base_map[oid]})

            # zapis
            out_sorted = sorted(out, key=lambda x: x["opinion_id"])
            write_jsonl(out_sorted, clean_path)
            print(f"✅ Batch {bid:04d}: Successfully processed and saved {len(out_sorted)} opinions")
            return out_sorted

        except Exception as e:
            # zapisz raw odpowiedź do debugowania
            try:
                if 'content' in locals() and content:
                    write_text(content[:20000], raw_path)
            except Exception:
                pass

            attempt += 1
            print(f"⚠️ Batch {bid:04d}: API call failed (attempt {attempt}/{retries+1}): {e}")
            if attempt > retries:
                print(f"❌ Batch {bid:04d}: All attempts failed. Writing fallback originals.")
                write_jsonl(req_items, clean_path)
                return [{"opinion_id": it["opinion_id"], "lang": "pl", "text": it["text"]} for it in req_items]
            sleep_time = backoff ** attempt + 0.25 * attempt
            print(f"⏱️ Batch {bid:04d}: Retrying in {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)

# ---------- Orchestrator ----------

async def run_cleanup(
    opinions_path: Path,
    outdir: Path,
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int = 20,
    concurrency: int = 4,
    retries: int = 3,
    backoff: float = 1.5,
    max_tokens: int = 2048,
    timeout: float = 90.0,
):
    print("\n" + "=" * 60)
    print("🚀 Starting Polish text cleanup pipeline")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Input file: {opinions_path}")
    print(f"   - Output directory: {outdir}")
    print(f"   - Model: {model}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Concurrency: {concurrency}")
    print(f"   - Max tokens: {max_tokens}")
    print(f"   - Timeout: {timeout}s")
    print("=" * 60)
    
    print("📂 Loading input files...")
    rows = read_jsonl(opinions_path)
    pl_rows = [r for r in rows if r.get("lang") == "pl"]
    en_rows = [r for r in rows if r.get("lang") == "en"]
    print(f"📊 Found {len(pl_rows)} Polish opinions and {len(en_rows)} English opinions")

    batches_dir = outdir / "pl_cleanup_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created batch directory: {batches_dir}")

    # chunk
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(pl_rows), batch_size):
        batches.append(pl_rows[i : i + batch_size])
    print(f"🧩 Split into {len(batches)} batches of size {batch_size}")

    print("🔌 Creating API client...")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    sem = asyncio.Semaphore(concurrency)
    cleaned_all: List[Dict[str, Any]] = []
    print(f"⚙️ Using concurrency level: {concurrency} (max parallel requests)")

    async def worker(bid: int, items: List[Dict[str, Any]]):
        async with sem:
            start_time = datetime.now()
            print(f"🧹 Processing PL cleanup batch #{bid:04d} ({len(items)} items)")
            out = await process_batch(
                client=client,
                model=model,
                items=items,
                batches_dir=batches_dir,
                bid=bid,
                retries=retries,
                backoff=backoff,
                max_tokens=max_tokens,
                save_prompt=True,
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"⏱️ Batch #{bid:04d} completed in {duration:.2f} seconds")
            cleaned_all.extend(out)

    print(f"🚀 Starting {len(batches)} batch processing tasks...")
    start_time_all = datetime.now()
    
    tasks = [asyncio.create_task(worker(i + 1, b)) for i, b in enumerate(batches)]
    await asyncio.gather(*tasks)
    
    end_time_all = datetime.now()
    duration_all = (end_time_all - start_time_all).total_seconds()
    print(f"✅ All batches processed in {duration_all:.2f} seconds")
    
    await client.close()

    # Save Polish-only cleaned
    print("💾 Saving Polish cleaned opinions...")
    pl_clean_path = outdir / "opinions_pl_cleaned.jsonl"
    write_jsonl(cleaned_all, pl_clean_path)
    print(f"✅ Saved {len(cleaned_all)} cleaned Polish opinions to {pl_clean_path}")

    # Merge back with EN (unchanged). Kolejność: PL → EN.
    print("🔄 Merging with English opinions...")
    merged = []
    cleaned_map = {r["opinion_id"]: r["text"] for r in cleaned_all}
    for r in pl_rows:
        text_new = cleaned_map.get(r["opinion_id"], r["text"])
        merged.append({"opinion_id": r["opinion_id"], "lang": "pl", "text": text_new})
    merged.extend(en_rows)

    merged_path = outdir / "opinions_cleaned.jsonl"
    write_jsonl(merged, merged_path)
    print(f"✅ Saved {len(merged)} merged opinions to {merged_path}")

    # Audit
    print("📊 Generating audit report...")
    audit = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "inputs": str(opinions_path),
        "outputs": {
            "pl_cleaned": str(pl_clean_path),
            "opinions_cleaned": str(merged_path)
        },
        "counts": {
            "pl_in": len(pl_rows),
            "pl_cleaned": len(cleaned_all),
            "en_passthrough": len(en_rows),
            "total_out": len(merged)
        },
        "params": {
            "batch_size": batch_size,
            "concurrency": concurrency,
            "retries": retries,
            "backoff": backoff,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "model": model,
            "base_url": base_url
        }
    }
    audit_path = outdir / "pl_cleanup_audit.json"
    write_json(audit, audit_path)
    print("📋 Audit report saved to:", audit_path)
    
    print("\n" + "=" * 60)
    print("✅ Polish text cleanup pipeline completed successfully!")
    print("=" * 60)
    print("📄 Output files:")
    print(f"   - PL cleaned: {pl_clean_path}")
    print(f"   - Merged: {merged_path}")
    print(f"   - Audit: {audit_path}")
    print("=" * 60)

def main():
    print("📋 Polish Text Cleanup Tool - Starting")
    load_dotenv()
    print("🔄 Loaded environment variables from .env")
    parser = argparse.ArgumentParser(description="Polish-only cleanup (spacing + anonymization) and merge back.")
    parser.add_argument("--opinions", default="dataset/opinions.jsonl")
    parser.add_argument("--outdir", default="data/working")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    print("✅ Parsed command line arguments")

    if not args.api_key:
        print("❌ ERROR: Missing API key")
        raise SystemExit("Missing API key. Set OPENAI_API_KEY/DEEPSEEK_API_KEY or pass --api-key.")
    asyncio.run(run_cleanup(
        opinions_path=Path(args.opinions),
        outdir=Path(args.outdir),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff=args.backoff,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    ))

if __name__ == "__main__":
    main()