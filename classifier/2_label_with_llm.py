"""
classifier/2_label_with_llm.py — Label headlines using GPT-4o as an expert annotator.

For each headline, asks gpt-4o whether it is likely to move a Kalshi prediction
market. Returns structured JSON with label, confidence, and reason via the
OpenAI structured outputs API.

Usage:
    export OPENAI_API_KEY=sk-...
    python classifier/2_label_with_llm.py

    # Resume a partial run (skips already-labeled rows):
    python classifier/2_label_with_llm.py --resume

Requirements:
    pip install openai
"""

import argparse
import asyncio
import csv
import json
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "data", "headlines_raw.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "headlines_labeled.csv")

MODEL           = "gpt-4o"
MAX_CONCURRENCY = 20
RETRY_ATTEMPTS  = 3
RETRY_DELAY_S   = 2.0

SYSTEM_PROMPT = """\
You are a prediction market analyst for Kalshi, a regulated US prediction market exchange.

Kalshi lists binary contracts on:
- US politics and elections (presidential approval, congressional votes, elections)
- Federal Reserve and macroeconomic outcomes (rate decisions, CPI, unemployment, GDP)
- Geopolitical events (wars, sanctions, diplomatic agreements)
- Weather and natural disasters (named storms, NOAA forecasts)
- Cryptocurrency prices and regulation
- Corporate earnings surprises and major M&A
- Sports championship outcomes
- Supreme Court decisions and major legislation

Your task: given a news headline, decide whether this headline is likely to cause a
meaningful probability shift on any currently-open Kalshi market.

Criteria for label = 1 (relevant):
- The headline describes a concrete, measurable outcome or a surprise deviation from
  expectations in one of the topic areas above.
- A trader who saw this headline would immediately think "this changes my estimate"
  for at least one binary market.

Criteria for label = 0 (not relevant):
- Evergreen, lifestyle, local, or celebrity content with no bearing on tradeable outcomes.
- Headlines about foreign countries with no US market linkage.
- Vague or generic headlines ("Officials discuss plans", "Market watch", etc.).
- Pure noise / malformed headlines (UUIDs, code artifacts, names with no context).
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label":      {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["label", "confidence", "reason"],
    "additionalProperties": False,
}


async def label_headline(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict,
) -> dict:
    headline = row["headline"].strip()

    for attempt in range(RETRY_ATTEMPTS):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Headline: {headline}"},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name":   "headline_label",
                            "strict": True,
                            "schema": RESPONSE_SCHEMA,
                        },
                    },
                    max_tokens=128,
                    temperature=0,
                )

                parsed = json.loads(response.choices[0].message.content)
                row["llm_label"]      = int(parsed["label"])
                row["llm_confidence"] = float(parsed["confidence"])
                row["llm_reason"]     = str(parsed.get("reason", ""))
                return row

            except (json.JSONDecodeError, KeyError) as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY_S)
                else:
                    print(f"  [WARN] parse failed: {headline[:60]} — {e}")
                    row["llm_label"]      = -1
                    row["llm_confidence"] = 0.0
                    row["llm_reason"]     = f"parse_error: {e}"
                    return row

            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY_S * (2 ** attempt))
                else:
                    print(f"  [ERROR] {e} for: {headline[:60]}")
                    row["llm_label"]      = -1
                    row["llm_confidence"] = 0.0
                    row["llm_reason"]     = f"error: {e}"
                    return row

    return row


async def main(resume: bool) -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}\nRun 1_export_data.py first.")
        return

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    already_labeled: set[str] = set()
    if resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("llm_label", "") not in ("", "-1"):
                    already_labeled.add(r["id"])
        print(f"Resuming — {len(already_labeled)} rows already labeled, skipping.")

    to_label = [r for r in rows if r["id"] not in already_labeled]
    print(f"Labeling {len(to_label)} headlines with {MODEL}...")

    client    = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    t0 = time.time()
    labeled = await asyncio.gather(
        *[label_headline(client, semaphore, row) for row in to_label]
    )
    elapsed = time.time() - t0

    if resume and already_labeled:
        existing: list[dict] = []
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        id_to_existing = {r["id"]: r for r in existing}
        for row in labeled:
            id_to_existing[row["id"]] = row
        all_rows = list(id_to_existing.values())
    else:
        all_rows = labeled

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = ["id", "headline", "source", "llm_label", "llm_confidence", "llm_reason"]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    valid    = [r for r in all_rows if str(r.get("llm_label", "")) not in ("", "-1")]
    positive = sum(1 for r in valid if str(r["llm_label"]) == "1")
    failed   = sum(1 for r in all_rows if str(r.get("llm_label", "")) == "-1")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {len(valid)} labeled: {positive} positive, {len(valid)-positive} negative")
    if failed:
        print(f"  {failed} failed — re-run with --resume to retry")
    print(f"  Output: {OUTPUT_PATH}")
    print("Next: run 3_train_classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the output file")
    args = parser.parse_args()
    asyncio.run(main(resume=args.resume))
