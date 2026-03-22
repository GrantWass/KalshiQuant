# Kalshi Headline Relevance Classifier

A binary classifier that predicts whether a news headline is likely to move a Kalshi prediction market.
Replaces the current rule-based `event_score` filter with a learned model that is faster and more accurate.

---

## Why

The current pipeline uses a keyword + NLP score threshold to decide whether to forward events to the
FAISS market matcher. It misses relevant headlines that don't contain the right keywords, and
passes through irrelevant ones that happen to match. A trained binary classifier fixes both problems.

## Architecture

```
headline (string)
    ↓
sentence-transformer embedding   ← same model already in the pipeline (all-MiniLM-L6-v2)
    ↓
logistic regression / MLP        ← trained on LLM-labeled examples
    ↓
p(relevant)  [0..1]
```

**Why this stack?**
- Embeddings are already computed in the pipeline — no extra model load
- Logistic regression inference is ~0.1ms per sample
- Embeddings generalize: a new headline about "Fed rate decision" will be close to similar
  training examples even if those exact words never appeared

---

## Step-by-Step

### Step 1 — Export training data

Run `1_export_data.py`. This pulls every headline from `news_events` along with weak labels
derived from the existing pipeline signals (event_score, market matches, filter outcome).

Output: `data/headlines_raw.csv`

```
headline | source 
```

See: [1_export_data.py](1_export_data.py)

---

### Step 2 — Label with Claude

Run `2_label_with_llm.py`. This sends each headline to `gpt-4o` and asks it to judge
whether the headline is likely to move a Kalshi prediction market.

The model returns structured JSON:
```json
{
  "label": 1,
  "confidence": 0.92,
  "reason": "References a Fed rate decision — directly tradeable on Kalshi"
}
```

Output: `data/headlines_labeled.csv` (adds `llm_label`, `llm_confidence`, `llm_reason` columns)

**Cost estimate:** ~1,061 headlines × ~200 tokens = ~212K tokens ≈ $0.13 at gpt-4o pricing.
Batch processing with `asyncio` (20 concurrent) keeps this under 1 minute.

See: [2_label_with_llm.py](2_label_with_llm.py)

---

### Step 3 — Train the classifier

Run `3_train_classifier.py`. This:
1. Embeds all headlines using `all-MiniLM-L6-v2`
2. Trains a logistic regression on the LLM labels
3. Evaluates on a held-out split (prints precision, recall, F1, AUC)
4. Saves the model to `model/classifier.pkl`

Output: `model/classifier.pkl`, `model/eval_report.txt`

See: [3_train_classifier.py](3_train_classifier.py)

---

## Integrating into the pipeline

Once `model/classifier.pkl` exists, swap it into `pipeline/event_detector.py` in place of
(or alongside) the current `event_score` threshold. The classifier produces a probability;
set a threshold (e.g. 0.5) or tune it on the eval set to optimize precision/recall tradeoff.

---

## Re-training

As the pipeline accumulates more data, re-run all three steps periodically. The LLM labeling
cost is low and the model retrains in seconds. A good cadence is monthly or after 5K new events.

---

## Files

```
classifier/
├── README.md                ← this file
├── 1_export_data.py         ← pull headlines from DB
├── 2_label_with_llm.py      ← label with Claude API
├── 3_train_classifier.py    ← train + evaluate + save model
├── data/                    ← generated CSVs (gitignored)
│   ├── headlines_raw.csv
│   └── headlines_labeled.csv
└── model/                   ← saved model artifacts (gitignored)
    ├── classifier.pkl
    └── eval_report.txt
```
