"""
classifier/1_export_data.py — Export headlines from PostgreSQL for classifier training.

Pulls every row from news_events and writes a clean CSV ready for LLM labeling.
No labels are assigned here — all labeling is done in Step 2.

Usage:
    python classifier/1_export_data.py
"""

import csv
import os

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://kalshi:kalshi@localhost:5432/kalshiquant",
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "headlines_raw.csv")

QUERY = """
SELECT
    ne.id::text  AS id,
    ne.headline,
    ne.source
FROM news_events ne
ORDER BY ne.fetched_at DESC;
"""


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(QUERY)
    cols = [desc[0] for desc in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "headline", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} headlines → {OUTPUT_PATH}")
    print("Next: run 2_label_with_llm.py")


if __name__ == "__main__":
    main()
