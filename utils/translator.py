# src/translator.py
# Async translation + anonymization + batching against Deepseek Chat Completions API

import os
import re
import json
import orjson
import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import httpx
from dotenv import load_dotenv


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

EMOTIONS = [
    "Positive", "Negative", "Happiness", "Delight", "Inspiring",
    "Surprise", "Compassion", "Fear", "Sadness", "Disgust", "Anger",
    "Ironic", "Political", "Interesting", "Understandable", "Offensive", "Funny"
]

# ---------- IO utils ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
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
        "You are a careful medical-domain translator.\n"
        "TASK: Translate each Polish review to natural, fluent English while preserving the meaning, tone, and nuance.\n\n"
        "ANONYMIZATION RULES:\n"
        " - If a personal first/last name appears, replace it with the neutral placeholder: 'Jan Kowalski' (male) or 'Anna Kowalska' (female).\n"
        " - Do NOT invent names if none appear in the original text.\n"
        " - Keep clinic/hospital/brand names as-is unless they directly identify a private person; in doubt, keep them.\n"
        " - Do not add content that is not present in the source.\n\n"
        "OUTPUT:\n"
        "Return a STRICT JSON array (only the array, no commentary, no code fences). Each element:\n"
        "{ \"opinion_id\": str, \"lang\": \"en\", \"text\": str, \"redacted\": int }\n"
        " - \"redacted\" = number of personal name replacements performed.\n"
    )
    payload = [{"opinion_id": it["opinion_id"], "text": it["text"]} for it in batch_items]
    # Dajemy modelowi wejście w formie listy obiektów
    return instruction + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# ---------- Deepseek client ----------

class DeepseekClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, api_url: str = DEEPSEEK_API_URL, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=api_url.rsplit("/v1", 1)[0] if "/v1" in api_url else None,
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self._endpoint = api_url

    async def close(self):
        await self._client.aclose()

    async def translate_batch(self, items: List[Dict[str, Any]], retries: int = 4, backoff_base: float = 1.5) -> List[Dict[str, Any]]:
        """
        Wysyła jeden wsad jako pojedyncze wywołanie chat/completions.
        Zwraca listę obiektów {opinion_id, lang, text, redacted}.
        """
        prompt = build_prompt_for_translation(items)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that strictly follows output format instructions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }

        for attempt in range(retries + 1):
            try:
                resp = await self._client.post(self._endpoint, content=orjson.dumps(body))
                status = resp.status_code
                text = resp.text

                if status >= 500 or status == 429:
                    raise httpx.HTTPStatusError(f"Server/Rate error {status}", request=resp.request, response=resp)

                resp.raise_for_status()
                data = resp.json()

                # Deepseek (OpenAI-like) schema:
                # data["choices"][0]["message"]["content"]
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                parsed = _extract_json_array(content)
                if not isinstance(parsed, list):
                    raise ValueError("Model did not return a JSON array.")
                # quick schema check
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

            except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
                if attempt == retries:
                    raise
                sleep_s = backoff_base ** attempt + (0.25 * attempt)
                await asyncio.sleep(sleep_s)

        # Should not reach here
        return []


def _extract_json_array(s: str) -> Any:
    """
    Wyciąga pierwszą tablicę JSON ze stringa (usuwając ewentualne komentarze/markdown).
    """
    # Usuwamy code fences, jeśli model je dodał
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    # Szukamy pierwszego '[ ... ]'
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in response.")
    segment = s[start:end+1]
    return json.loads(segment)


# ---------- Batch runner ----------

async def process_batch(
    client: DeepseekClient,
    batch_items: List[Dict[str, Any]],
    batches_dir: Path,
    batch_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Przetwarza jeden batch: zapisuje snapshoty, woła API, waliduje,
    zwraca czyste EN rekordy + porażki + statystyki.
    """
    # request snapshot
    req_path = batches_dir / f"batch_{batch_idx:04d}_request.jsonl"
    write_jsonl(batch_items, req_path)

    # prompt snapshot
    prompt_path = batches_dir / f"batch_{batch_idx:04d}_prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(build_prompt_for_translation(batch_items))

    # API call
    try:
        responses = await client.translate_batch(batch_items)
    except Exception as e:
        failures = [{"opinion_id": it["opinion_id"], "error": str(e), "attempts": 1} for it in batch_items]
        return [], failures, {"batch": batch_idx, "ok": 0, "failed": len(batch_items)}

    # raw response snapshot
    resp_path = batches_dir / f"batch_{batch_idx:04d}_response.jsonl"
    write_jsonl(responses, resp_path)

    # validate/clean
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

    # clean snapshot
    clean_path = batches_dir / f"batch_{batch_idx:04d}_clean.jsonl"
    write_jsonl(clean, clean_path)

    # redaction flags (append)
    redaction_path = batches_dir.parent / "translation_redaction_flags.jsonl"
    flags = [{
        "opinion_id": r.get("opinion_id"),
        "names_detected": (r.get("redacted", 0) or 0) > 0,
        "redacted_count": r.get("redacted", 0) or 0
    } for r in responses]
    write_jsonl(flags, redaction_path, mode="a")

    stats = {"batch": batch_idx, "ok": len(clean), "failed": len(failures)}
    return clean, failures, stats


async def translate_file_async(
    in_path: Path,
    out_path: Path,
    batch_size: int = 50,
    concurrency: int = 5,
    model: str = DEFAULT_MODEL,
    api_url: str = DEEPSEEK_API_URL,
    api_key: Optional[str] = None,
):
    batches_dir = ensure_dirs(out_path)
    data = read_jsonl(in_path)

    # wybieramy tylko PL do tłumaczenia; ale całe wejście zachowujemy jako PL out
    pl_rows = [r for r in data if r.get("lang") == "pl"]
    # na wyjście zapisujemy najpierw PL
    write_jsonl(pl_rows, out_path)  # nadpisanie/utworzenie

    # tworzymy batch'e
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for r in pl_rows:
        current.append(r)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    client = DeepseekClient(api_key=api_key, model=model, api_url=api_url)

    sem = asyncio.Semaphore(concurrency)
    all_en: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []

    async def runner(idx: int, items: List[Dict[str, Any]]):
        async with sem:
            clean, fails, s = await process_batch(client, items, batches_dir, idx)
            all_en.extend(clean)
            all_failures.extend(fails)
            stats.append(s)

    tasks = [asyncio.create_task(runner(i, b)) for i, b in enumerate(batches, start=1)]
    t0 = time.time()
    try:
        await asyncio.gather(*tasks)
    finally:
        await client.close()
    t1 = time.time()

    # dopisz EN do pliku wynikowego (append)
    write_jsonl(all_en, out_path, mode="a")

    # failures
    failures_path = out_path.parent / "translation_failures.jsonl"
    if all_failures:
        write_jsonl(all_failures, failures_path)

    # audit
    audit = {
        "model": model,
        "api_url": api_url,
        "total_inputs": len(pl_rows),
        "translated": len(all_en),
        "failed": len(all_failures),
        "batches": len(batches),
        "batch_size": batch_size,
        "concurrency": concurrency,
        "elapsed_sec": round(t1 - t0, 3),
    }
    with open(out_path.parent / "translation_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


# ---------- CLI ----------

def main():
    load_dotenv()  # loads .env if present
    parser = argparse.ArgumentParser(description="Async translation + anonymization via Deepseek")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to data/working/raw_opinions.jsonl")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to data/working/translated_opinions.jsonl")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-url", type=str, default=DEEPSEEK_API_URL)
    parser.add_argument("--api-key", type=str, default=os.getenv("DEEPSEEK_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY. Put it in .env or pass --api-key.")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    asyncio.run(translate_file_async(
        in_path=in_path,
        out_path=out_path,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        model=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
    ))


if __name__ == "__main__":
    main()