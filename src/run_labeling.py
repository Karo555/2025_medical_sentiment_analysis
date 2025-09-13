# src/labeling_run.py
# Step 3: Async labeling run (per-opinion, all personas), resume + delta-retry + consolidation
import os
import json
import argparse
import asyncio
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

EMOTIONS = [
    "Positive","Negative","Happiness","Delight","Inspiring",
    "Surprise","Compassion","Fear","Sadness","Disgust","Anger",
    "Ironic","Political","Interesting","Understandable","Offensive","Funny"
]

# ---------- Logging utils ----------

def get_timestamp() -> str:
    """Return a formatted timestamp for logging purposes."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(message: str, level: str = "INFO") -> None:
    """Print a timestamped log message."""
    print(f"[{level}] {get_timestamp()} | {message}")

# ---------- IO utils ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(rows: List[Dict[str, Any]], path: Path, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_user_payload(items: List[Dict[str, Any]]) -> str:
    """We append the concrete INPUT array to the static prompt snapshot."""
    payload = [{"opinion_id": it["opinion_id"], "lang": it["lang"], "text": it["text"]} for it in items]
    return "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

# ---------- Parsing & validation ----------

def extract_json_array(s: str) -> Any:
    # strip code fences if any
    s = s.strip()
    if s.startswith("```"):
        # naive fence strip
        s = s.strip("`")
        # after stripping, might still contain language hint, try to find array
    # find array bounds
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in response.")
    segment = s[start:end+1]
    return json.loads(segment)

def validate_and_flatten(
    opinions: List[Dict[str, Any]],
    personas_ids: List[str],
    result_array: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Returns:
      clean_flat: [{"opinion_id","lang","persona_id","labels":[...]}...]
      missing: list of (opinion_id, persona_id) pairs that were not returned or invalid
    """
    # index expected langs per opinion_id
    opin_by_id = {o["opinion_id"]: o for o in opinions}
    personas_set = set(personas_ids)
    clean_flat: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str]] = []

    # track seen personas per opinion
    seen: Dict[str, set] = {o["opinion_id"]: set() for o in opinions}

    for el in result_array:
        oid = el.get("opinion_id")
        lang = el.get("lang")
        if oid not in opin_by_id:
            # unknown opinion id -> ignore
            continue
        # enforce lang from source
        lang_src = opin_by_id[oid]["lang"]
        lang = lang_src

        results = el.get("results", [])
        if not isinstance(results, list):
            # everything missing for this opinion
            for pid in personas_ids:
                missing.append((oid, pid))
            continue

        for r in results:
            pid = r.get("persona_id")
            labels = r.get("labels")
            if pid not in personas_set or not isinstance(labels, list) or len(labels) != len(EMOTIONS):
                # invalid -> mark missing for this persona
                # (only if pid is known; if pid missing we mark all personas as missing later)
                if pid in personas_set:
                    missing.append((oid, pid))
                continue
            # coerce ints 0/1
            try:
                labels_bin = [1 if int(x) == 1 else 0 for x in labels]
            except Exception:
                missing.append((oid, pid))
                continue
            clean_flat.append({
                "opinion_id": oid,
                "lang": lang,
                "persona_id": pid,
                "labels": labels_bin
            })
            seen[oid].add(pid)

    # add missing for any not seen persona per opinion
    for oid in seen:
        for pid in personas_ids:
            if pid not in seen[oid]:
                missing.append((oid, pid))

    return clean_flat, missing

def build_delta_prompt(base_prompt: str, items: List[Dict[str, Any]], missing_map: Dict[str, List[str]]) -> str:
    """
    Tell the model to return ONLY the missing personas for given opinions.
    """
    instr = (
        "\nDELTA REQUEST:\n"
        "Zwróć TYLKO brakujące persony dla danych opinii (pozostałe pomiń). "
        "Format odpowiedzi identyczny jak wcześniej, ale 'results' powinno zawierać wyłącznie wymienione 'persona_id'.\n"
    )
    # construct compact hint about missing per opinion
    miss_pairs = [{"opinion_id": oid, "missing_personas": pids} for oid, pids in missing_map.items()]
    return base_prompt + instr + "MISSING_HINT:\n" + json.dumps(miss_pairs, ensure_ascii=False) + build_user_payload(items)

# ---------- Runner per batch ----------

async def call_api(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    log(f"Sending API request to model: {model} (max tokens: {max_tokens})")
    t0 = time.time()
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You strictly follow output schemas and return ONLY raw JSON arrays."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    log(f"API response received in {elapsed:.2f}s, response length: {len(completion.choices[0].message.content or '0')} chars")
    return completion.choices[0].message.content or ""

async def process_one_batch(
    client: AsyncOpenAI,
    model: str,
    batches_dir: Path,
    batch_id: str,
    retries: int,
    backoff_base: float,
    timeout: float,
    max_tokens: int,
    personas: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """
    Returns: (ok_pairs, failed_pairs)
    """
    req_path = batches_dir / f"batch_{batch_id}_request.jsonl"
    prompt_path = batches_dir / f"batch_{batch_id}_prompt.txt"
    resp_path = batches_dir / f"batch_{batch_id}_response.jsonl"
    clean_path = batches_dir / f"batch_{batch_id}_clean.jsonl"
    missing_path = batches_dir / f"batch_{batch_id}_missing.jsonl"
    personas_path = batches_dir / f"batch_{batch_id}_personas_snapshot.json"

    items = read_jsonl(req_path)
    lang = items[0]["lang"] if items else "pl"
    personas_ids = [p["id"] for p in personas]
    base_prompt = read_text(prompt_path)
    user_payload = build_user_payload(items)
    full_prompt = base_prompt + user_payload
    prompt_path.write_text(full_prompt, encoding="utf-8")


    # initial call
    attempt = 0
    parsed_array: List[Dict[str, Any]] = []
    log(f"Starting initial API call for batch {batch_id} ({len(items)} opinions, {len(personas_ids)} personas)")
    while attempt <= retries:
        try:
            if attempt > 0:
                sleep_s = (backoff_base ** attempt) + 0.25 * attempt
                log(f"RETRY {attempt}/{retries} for batch {batch_id} - sleeping {sleep_s:.2f}s...", level="WARN")
                await asyncio.sleep(sleep_s)

            content = await call_api(client, model, full_prompt, max_tokens=max_tokens)
            parsed_array = extract_json_array(content)
            log(f"Successfully parsed API response for batch {batch_id} - received {len(parsed_array)} opinion results")
            break
        except Exception as e:
            attempt += 1
            if attempt > retries:
                write_jsonl([{"error": repr(e)}], resp_path)
                log(f"FAILED initial call for batch {batch_id}: {e}", level="ERROR")
                # mark all pairs as failed
                all_pairs = [(it["opinion_id"], pid) for it in items for pid in personas_ids]
                write_jsonl([{"opinion_id": oid, "persona_id": pid, "reason": "initial_call_failed"} for oid, pid in all_pairs], missing_path)
                return 0, len(all_pairs)

    # save parsed response (per opinion)
    # each line is a dict with opinion_id, lang, results:[{persona_id, labels}]
    write_jsonl(parsed_array, resp_path)

    # validate & flatten
    clean_flat, missing = validate_and_flatten(items, personas_ids, parsed_array)

    if not missing:
        write_jsonl(clean_flat, clean_path)
        return len(clean_flat), 0

    # delta-retries for missing only (max 2)
    missing_map: Dict[str, List[str]] = {}
    for oid, pid in missing:
        missing_map.setdefault(oid, []).append(pid)

    max_delta_rounds = 2
    for r in range(1, max_delta_rounds + 1):
        delta_items = [it for it in items if it["opinion_id"] in missing_map]
        delta_prompt = build_delta_prompt(base_prompt, delta_items, missing_map)

        print(f"  ↪ DELTA RETRY round {r}: missing_pairs={sum(len(v) for v in missing_map.values())}")
        attempt = 0
        parsed_delta: List[Dict[str, Any]] = []
        while attempt <= retries:
            try:
                if attempt > 0:
                    sleep_s = (backoff_base ** attempt) + 0.25 * attempt
                    print(f"    ↪ delta RETRY {attempt}/{retries} sleeping {sleep_s:.2f}s...")
                    await asyncio.sleep(sleep_s)
                content = await call_api(client, model, delta_prompt, max_tokens=max_tokens)
                parsed_delta = extract_json_array(content)
                print(f"    ✓ DELTA round {r} success: received {len(parsed_delta)} opinion results")
                break
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    print(f"    ✗ DELTA round {r} failed: {e}")
                    break

        if parsed_delta:
            add_flat, _ = validate_and_flatten(delta_items, personas_ids, parsed_delta)
            # merge: prefer newer results for missing pairs
            have_pairs = {(c["opinion_id"], c["persona_id"]) for c in clean_flat}
            for row in add_flat:
                key = (row["opinion_id"], row["persona_id"])
                if key not in have_pairs:
                    clean_flat.append(row)

        # recompute missing
        still_missing: List[Tuple[str, str]] = []
        seen_pairs = {(c["opinion_id"], c["persona_id"]) for c in clean_flat}
        for it in items:
            for pid in personas_ids:
                if (it["opinion_id"], pid) not in seen_pairs:
                    still_missing.append((it["opinion_id"], pid))

        if not still_missing:
            write_jsonl(clean_flat, clean_path)
            return len(clean_flat), 0

        # shrink missing_map
        missing_map = {}
        for oid, pid in still_missing:
            missing_map.setdefault(oid, []).append(pid)

    # after delta retries: persist partial clean and missing list
    write_jsonl(clean_flat, clean_path)
    missing_items = []
    for oid, pids in missing_map.items():
        for pid in pids:
            missing_items.append({"opinion_id": oid, "persona_id": pid, "reason": "delta_exhausted"})
    write_jsonl(missing_items, missing_path)
    
    log(f"Final results for batch {batch_id}: {len(clean_flat)} successful pairs, {len(missing_items)} missing pairs")
    return len(clean_flat), len(missing_items)

# ---------- Consolidation ----------

def consolidate_embeddings(batches_dir: Path, out_path: Path) -> int:
    """
    Merge all batch_*_clean.jsonl in numeric order into embeddings.jsonl (PL first then EN,
    thanks to deterministic ordering from step 2). Returns number of lines written.
    """
    clean_files = sorted(batches_dir.glob("batch_*_clean.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for cf in clean_files:
            rows = read_jsonl(cf)
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
            print(f"  ↪ Added {len(rows)} pairs from {cf.name}")
    print(f"✅ embeddings.jsonl written: {written} rows -> {out_path}")
    return written

# ---------- Main async orchestration ----------

async def run_labeling(
    outdir: Path,
    model: str,
    base_url: str,
    api_key: str,
    concurrency: int = 4,
    retries: int = 4,
    backoff_base: float = 1.5,
    timeout: float = 180.0,
    max_tokens: int = 2048,
    resume: bool = True,
    from_batch: Optional[int] = None,
    to_batch: Optional[int] = None,
):
    log(f"Starting labeling pipeline with model: {model}")
    log(f"Configuration: concurrency={concurrency}, retries={retries}, timeout={timeout}s, max_tokens={max_tokens}")
    
    batches_dir = outdir / "labeling_batches"
    index_path = batches_dir / "batches_index.json"
    if not index_path.exists():
        log(f"Missing {index_path}. Run step 2 (labeling_build_batches.py) first.", level="ERROR")
        raise SystemExit(f"Missing {index_path}. Run step 2 (labeling_build_batches.py) first.")

    log("Loading batches index...")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    batches = index.get("batches", [])
    if not batches:
        log("No batches found in index.", level="WARN")
        return
    
    log(f"Found {len(batches)} batch(es) in index")
    if from_batch or to_batch:
        log(f"Processing range: batch #{from_batch or 1} to #{to_batch or 'end'}")

    log(f"Initializing OpenAI client with base URL: {base_url}")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # personas are per-batch snapshot files; we load lazily inside process_one_batch
    log(f"Setting up processing with concurrency level: {concurrency}")
    sem = asyncio.Semaphore(concurrency)
    ok_total = 0
    fail_total = 0

    async def worker(bmeta: Dict[str, Any]):
        nonlocal ok_total, fail_total
        bid = bmeta["batch_id"]
        lang = bmeta["lang"]
        req_path = Path(bmeta["request"])
        pers_path = Path(bmeta["personas_snapshot"])

        # range filter
        bnum = int(bid)
        if from_batch and bnum < from_batch:
            return
        if to_batch and bnum > to_batch:
            return

        # resume: if clean exists and has expected lines, skip
        clean_path = batches_dir / f"batch_{bid}_clean.jsonl"
        if resume and clean_path.exists():
            # expected lines = opinions_in_batch * personas_count
            opinions = read_jsonl(req_path)
            personas = json.loads(pers_path.read_text(encoding="utf-8"))
            expected = len(opinions) * len(personas)
            try:
                with clean_path.open("r", encoding="utf-8") as f:
                    count = sum(1 for _ in f if _.strip())
            except Exception:
                count = -1
            if count == expected:
                print(f"⏭️  Skipping batch #{bid} (already complete: {count}/{expected})")
                return

        # load personas for this batch
        personas = json.loads(pers_path.read_text(encoding="utf-8"))

        async with sem:
            t0 = time.time()
            print(f"📦 Processing batch #{bid} lang={lang} ...")
            try:
                ok_pairs, fail_pairs = await process_one_batch(
                    client=client,
                    model=model,
                    batches_dir=batches_dir,
                    batch_id=bid,
                    retries=retries,
                    backoff_base=backoff_base,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    personas=personas,
                )
                ok_total += ok_pairs
                fail_total += fail_pairs
                dt = time.time() - t0
                print(f"✅ Batch #{bid} DONE in {dt:.1f}s (ok={ok_pairs}, fail={fail_pairs})")
            except Exception as e:
                print(f"❌ Batch #{bid} crashed: {e}")
                # mark whole batch as failed in failures file
                failures_path = outdir / "labeling_failures.jsonl"
                write_jsonl([{
                    "batch_id": bid,
                    "error": repr(e)
                }], failures_path, mode="a")

    tasks = [asyncio.create_task(worker(b)) for b in batches]
    await asyncio.gather(*tasks)
    await client.close()

    # consolidate into embeddings.jsonl (PL first then EN via numeric order)
    embeddings_path = outdir / "embeddings.jsonl"
    consolidate_embeddings(batches_dir, embeddings_path)

    # audit
    audit = {
        "model": model,
        "base_url": base_url,
        "concurrency": concurrency,
        "retries": retries,
        "backoff_base": backoff_base,
        "timeout_sec": timeout,
        "max_tokens": max_tokens,
        "ok_pairs": ok_total,
        "failed_pairs": fail_total,
    }
    audit_file = outdir / "labeling_audit_run.json"
    write_json(audit, audit_file)
    success_rate = (ok_total / (ok_total + fail_total) * 100) if (ok_total + fail_total) > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"🏁 Labeling pipeline completed at {get_timestamp()}")
    print(f"📊 Summary: {ok_total} successful pairs, {fail_total} failed pairs ({success_rate:.1f}% success rate)")
    print(f"📝 Audit file: {audit_file}")
    print("=" * 60)

def main():
    load_dotenv()
    print(f"📋 Starting Medical Sentiment Analysis Labeling Pipeline - {get_timestamp()}")
    parser = argparse.ArgumentParser(description="Run labeling (step 3): async API, resume, delta-retry, consolidation.")
    parser.add_argument("--outdir", default="data/working", help="Base working dir (contains labeling_batches)")
    parser.add_argument("--model", default="deepseek-chat", help="Model name")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"), help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"), help="API key")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff-base", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--from-batch", type=int, default=None)
    parser.add_argument("--to-batch", type=int, default=None)
    parser.add_argument("--save-response", action="store_true", help="Save parsed model response per batch (debug).")
    args = parser.parse_args()

    print(f"🔧 Configuration: model={args.model}, concurrency={args.concurrency}, resume={args.resume}")
    if args.from_batch or args.to_batch:
        print(f"🔢 Processing batch range: {args.from_batch or 1} to {args.to_batch or 'end'}")

    if not args.api_key:
        print("❌ ERROR: Missing API key. Set OPENAI_API_KEY/DEEPSEEK_API_KEY in .env or pass --api-key.")
        raise SystemExit("Missing API key. Set OPENAI_API_KEY/DEEPSEEK_API_KEY in .env or pass --api-key.")

    try:
        print(f"\n🚀 Starting labeling process at {get_timestamp()}")
        start_time = time.time()
        
        asyncio.run(run_labeling(
            outdir=Path(args.outdir),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
            retries=args.retries,
            backoff_base=args.backoff_base,
            timeout=args.timeout,
            max_tokens=args["max_tokens"] if isinstance(args, dict) else args.max_tokens,
            resume=args.resume,
            from_batch=args.from_batch,
            to_batch=args.to_batch,
        ))
        
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"⏱️  Total execution time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Partial results will be persisted.")

if __name__ == "__main__":
    main()