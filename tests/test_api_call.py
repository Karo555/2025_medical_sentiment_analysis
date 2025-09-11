# scripts/diagnose_translator.py
# Diagnostic script using OpenAI SDK (sync) against an OpenAI-compatible endpoint (e.g., DeepSeek)

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# ====== helpers (minimal, zgodne z translator.py) ======

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_prompt_for_translation(batch_items: List[Dict[str, Any]]) -> str:
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
    return instruction + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

def extract_json_array(s: str) -> Any:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in response.")
    segment = s[start:end+1]
    return json.loads(segment)

# ====== main diagnose ======

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Diagnose translation step via OpenAI SDK (single call).")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to raw_opinions.jsonl")
    parser.add_argument("--outdir", dest="outdir", required=True, help="Directory for diagnostics artifacts")
    parser.add_argument("--n", type=int, default=5, help="How many opinions to send")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com/v1")
    # akceptujemy oba: DEEPSEEK_API_KEY lub OPENAI_API_KEY
    default_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    parser.add_argument("--api-key", type=str, default=default_key)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Put DEEPSEEK_API_KEY or OPENAI_API_KEY in .env or pass --api-key.")

    in_path = Path(args.in_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load first N PL rows
    all_rows = read_jsonl(in_path)
    pl_rows = [r for r in all_rows if r.get("lang") == "pl"]
    if not pl_rows:
        raise SystemExit("No PL rows found in input.")
    items = pl_rows[: args.n]

    # 2) build prompt and request body (snapshoty)
    prompt = build_prompt_for_translation(items)
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that strictly follows output format instructions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    write_text(prompt, outdir / "prompt.txt")
    write_json(body, outdir / "request_body.json")

    # 3) OpenAI SDK call (sync)
    try:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        completion = client.chat.completions.create(
            model=args.model,
            messages=body["messages"],
            temperature=body["temperature"],
        )
        # zapisz pełny completion (serializowalny dict)
        completion_dict = completion.model_dump() if hasattr(completion, "model_dump") else completion.__dict__
        write_json(completion_dict, outdir / "completion.json")

        content = (completion.choices[0].message.content or "").strip()
        write_text(content, outdir / "content.txt")
    except Exception as e:
        write_text(f"EXCEPTION:\n{repr(e)}", outdir / "exception.txt")
        print(f"[!] SDK error: {e}")
        return

    # 4) parse JSON array from content
    try:
        parsed = extract_json_array(content)
        write_json(parsed, outdir / "parsed.json")
        ok = [el for el in parsed if isinstance(el, dict) and "opinion_id" in el and "text" in el]
        print(f"[OK] Parsed JSON array, items: {len(parsed)}, valid: {len(ok)}")
    except Exception as e:
        write_text(f"PARSE_ERROR:\n{repr(e)}\n\nCONTENT_START\n{content[:5000]}\nCONTENT_END", outdir / "parse_error.txt")
        print(f"[!] Parse error: {e}")
        return

    print("[DONE] Diagnostics artifacts written to:", str(outdir))

if __name__ == "__main__":
    main()