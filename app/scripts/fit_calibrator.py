"""
app/scripts/fit_calibrator.py
------------------------------
Phase 5 Stage 5 — Fit Probability Calibrator.

Reads ``data/calibration_set.jsonl`` produced by ``build_calibration_set.py``
and fits a one-feature logistic regression:

    P(correct=1 | margin) = sigmoid(a * margin + b)

Saves the fitted model to ``data/calibrator.joblib`` using joblib.

Also prints:
  - Number of training samples
  - Train accuracy
  - Coefficient (a) and intercept (b)
  - Brier score (a sanity-check for probabilistic accuracy)

Usage
-----
    python -m app.scripts.fit_calibrator
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fit_calibrator")

_INPUT_PATH = Path("data/calibration_set.jsonl")
_OUTPUT_PATH = Path("data/calibrator.joblib")

# Cap margin at a reasonable ceiling to prevent extreme leverage from inf values
_MARGIN_CAP = 20.0


def fit_calibrator() -> LogisticRegression:
    """
    Load calibration pairs, fit a LogisticRegression, and save with joblib.

    Returns
    -------
    LogisticRegression
        The fitted calibrator.

    Raises
    ------
    FileNotFoundError
        If ``data/calibration_set.jsonl`` does not exist.
    ValueError
        If the file contains fewer than 10 usable pairs.
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Calibration set not found: {_INPUT_PATH}\n"
            "Run: python -m app.scripts.build_calibration_set"
        )

    margins: list[float] = []
    labels: list[int] = []

    with _INPUT_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                margin = float(rec["margin"])
                correct = int(rec["correct"])
                # Cap extreme margins
                margin = min(margin, _MARGIN_CAP)
                margins.append(margin)
                labels.append(correct)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping malformed record: %s", exc)

    n = len(margins)
    logger.info("Loaded %d calibration pairs from %s", n, _INPUT_PATH)

    if n < 10:
        raise ValueError(
            f"Too few calibration pairs ({n}). Need at least 10.\n"
            "Run build_calibration_set with more examples."
        )

    X = np.array(margins, dtype=np.float64).reshape(-1, 1)
    y = np.array(labels, dtype=np.int32)

    # Log class distribution
    n_correct = int(y.sum())
    logger.info(
        "Class distribution: correct=%d (%.1f%%), incorrect=%d (%.1f%%)",
        n_correct, 100.0 * n_correct / n,
        n - n_correct, 100.0 * (n - n_correct) / n,
    )

    # Fit logistic regression
    logger.info("Fitting LogisticRegression(max_iter=1000) …")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y)

    # Report metrics
    train_acc = float(clf.score(X, y))
    probs = clf.predict_proba(X)[:, 1]
    brier = float(brier_score_loss(y, probs))
    coef = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])

    logger.info("=" * 50)
    logger.info("Calibrator fitted successfully")
    logger.info("  Train samples  : %d", n)
    logger.info("  Train accuracy : %.4f", train_acc)
    logger.info("  Brier score    : %.4f  (0=perfect, 0.25=uninformed)", brier)
    logger.info("  Coefficient    : %.6f  (higher margin -> higher confidence)", coef)
    logger.info("  Intercept      : %.6f", intercept)
    logger.info("=" * 50)

    # Save model
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, _OUTPUT_PATH)
    logger.info("Calibrator saved → %s", _OUTPUT_PATH)

    return clf


if __name__ == "__main__":
    fit_calibrator()
