"""
classifier/3_train_classifier.py — Train and evaluate the binary headline classifier.

Pipeline:
  1. Load LLM-labeled headlines from data/headlines_labeled.csv
  2. Filter to high-confidence labels (llm_confidence >= 0.7 by default)
  3. Embed headlines using all-MiniLM-L6-v2 (same model as the trading pipeline)
  4. Train a logistic regression on the embeddings
  5. Evaluate on a stratified held-out split
  6. Save model to model/classifier.pkl

The saved artifact is a sklearn Pipeline: (StandardScaler → LogisticRegression).
Call classifier.predict_proba([headline_embedding])[0][1] for a relevance probability.

Usage:
    python classifier/3_train_classifier.py
    python classifier/3_train_classifier.py --min-confidence 0.8
    python classifier/3_train_classifier.py --model-type mlp   # small MLP alternative

Requirements:
    pip install scikit-learn sentence-transformers numpy
"""

import argparse
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import csv

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "data", "headlines_labeled.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "classifier.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "eval_report.txt")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_data(min_confidence: float) -> tuple[list[str], list[int]]:
    """Load headlines and LLM labels, filtering by confidence threshold."""
    if HAS_PANDAS:
        import pandas as pd
        df = pd.read_csv(INPUT_PATH)
        # Keep only confident, non-failed labels
        df = df[df["llm_label"].isin([0, 1])]
        df = df[df["llm_confidence"].astype(float) >= min_confidence]
        headlines = df["headline"].tolist()
        labels    = df["llm_label"].astype(int).tolist()
    else:
        with open(INPUT_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        headlines, labels = [], []
        for r in rows:
            if r.get("llm_label", "") not in ("0", "1"):
                continue
            if float(r.get("llm_confidence", 0)) < min_confidence:
                continue
            headlines.append(r["headline"])
            labels.append(int(r["llm_label"]))

    return headlines, labels


def embed(headlines: list[str]) -> np.ndarray:
    """Embed headlines using the same model as the trading pipeline."""
    print(f"Embedding {len(headlines)} headlines with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(headlines, show_progress_bar=True, batch_size=64)
    return embeddings.astype(np.float32)


def build_pipeline(model_type: str) -> Pipeline:
    if model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            max_iter=300,
            random_state=42,
        )
    else:  # logistic regression (default)
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",  # handles label imbalance automatically
            random_state=42,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> str:
    """Cross-validate and return a formatted report string."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # AUC via cross-validation
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    # Train/test split for full classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    lines = [
        "=" * 60,
        "CLASSIFIER EVALUATION REPORT",
        "=" * 60,
        f"Training examples : {len(X_train)}",
        f"Test examples     : {len(X_test)}",
        f"Positive rate     : {np.mean(y):.1%}",
        "",
        f"5-fold CV AUC     : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}",
        f"Test AUC          : {roc_auc_score(y_test, y_proba):.4f}",
        "",
        "Classification report (test set):",
        classification_report(y_test, y_pred, target_names=["noise", "relevant"]),
        "Confusion matrix (rows=actual, cols=predicted):",
        "             noise  relevant",
        f"  noise      {confusion_matrix(y_test, y_pred)[0][0]:5d}  {confusion_matrix(y_test, y_pred)[0][1]:8d}",
        f"  relevant   {confusion_matrix(y_test, y_pred)[1][0]:5d}  {confusion_matrix(y_test, y_pred)[1][1]:8d}",
        "=" * 60,
    ]
    return "\n".join(lines)


def main(min_confidence: float, model_type: str) -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"Input file not found: {INPUT_PATH}")
        print("Run 2_label_with_llm.py first.")
        return

    headlines, labels = load_data(min_confidence)
    print(f"Loaded {len(headlines)} examples (confidence >= {min_confidence})")
    print(f"  Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Negative: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    if len(headlines) < 50:
        print("\n[WARN] Fewer than 50 examples — consider lowering --min-confidence or collecting more data.")

    X = embed(headlines)
    y = np.array(labels)

    pipeline = build_pipeline(model_type)
    report   = evaluate(pipeline, X, y)
    print("\n" + report)

    # Retrain on full dataset before saving
    pipeline.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Report saved → {REPORT_PATH}")
    print("\nTo use the classifier:")
    print("  from sentence_transformers import SentenceTransformer")
    print("  import pickle")
    print("  model = SentenceTransformer('all-MiniLM-L6-v2')")
    print("  clf   = pickle.load(open('classifier/model/classifier.pkl', 'rb'))")
    print("  emb   = model.encode(['Fed raises rates 25 basis points'])")
    print("  prob  = clf.predict_proba(emb)[0][1]   # probability of being relevant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum LLM confidence to include a label (default: 0.7)")
    parser.add_argument("--model-type", choices=["logistic", "mlp"], default="logistic",
                        help="Classifier architecture (default: logistic)")
    args = parser.parse_args()
    main(min_confidence=args.min_confidence, model_type=args.model_type)
