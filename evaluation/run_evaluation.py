"""Run all evaluations, persist real metrics, and refresh the README table.

From the repository root:
    python -m evaluation.run_evaluation
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sklearn
import tensorflow as tf

from evaluation.cross_validation import evaluate_classification as cv_classification
from evaluation.cross_validation import evaluate_regression as cv_regression
from evaluation.held_out_test import evaluate_classification as test_classification
from evaluation.held_out_test import evaluate_regression as test_regression
from evaluation.models import (
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    RANDOM_STATE,
    VALIDATION_SPLIT,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "evaluation" / "results.json"
README_PATH = ROOT / "README.md"
START_MARKER = "<!-- EVALUATION-METRICS:START -->"
END_MARKER = "<!-- EVALUATION-METRICS:END -->"


def format_value(value, digits=4):
    return f"{value:.{digits}f}"


def build_readme_table(results):
    cv = results["cross_validation"]
    held = results["held_out_test"]
    rows = [
        "### Regression — Diabetes dataset",
        "",
        "| Metric | 5-fold CV (mean ± std) | Held-out test |",
        "|---|---:|---:|",
    ]
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")):
        mean_std = cv["regression"][metric]
        rows.append(
            f"| {label} | {format_value(mean_std['mean'])} ± "
            f"{format_value(mean_std['std'])} | {format_value(held['regression'][metric])} |"
        )

    rows += [
        "",
        "### Binary classification — Breast Cancer dataset",
        "",
        "| Metric | 5-fold CV (mean ± std) | Held-out test |",
        "|---|---:|---:|",
    ]
    for metric, label in (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ):
        mean_std = cv["classification"][metric]
        rows.append(
            f"| {label} | {format_value(mean_std['mean'])} ± "
            f"{format_value(mean_std['std'])} | {format_value(held['classification'][metric])} |"
        )

    rows += [
        "",
        f"_Generated automatically on {results['generated_at']} (UTC) with TensorFlow/Keras._",
    ]
    return "\n".join(rows)


def update_readme(table):
    readme = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in readme or END_MARKER not in readme:
        raise RuntimeError("README is missing evaluation table markers.")
    before, remainder = readme.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    README_PATH.write_text(
        before + START_MARKER + "\n" + table + "\n" + END_MARKER + after,
        encoding="utf-8",
    )


def main():
    print("Running 5-fold cross-validation...")
    cv_results = {
        "regression": cv_regression(),
        "classification": cv_classification(),
    }

    print("Running held-out evaluation...")
    held_out_results = {
        "regression": test_regression(),
        "classification": test_classification(),
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "intro-deep-learning-starter",
        "framework": "TensorFlow/Keras",
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": {
            "random_state": RANDOM_STATE,
            "cross_validation_folds": 5,
            "held_out_test_size": 0.2,
            "epochs_max": EPOCHS,
            "batch_size": BATCH_SIZE,
            "early_stopping_patience": PATIENCE,
            "validation_split": VALIDATION_SPLIT,
        },
        "datasets": {
            "regression": "sklearn.datasets.load_diabetes",
            "classification": "sklearn.datasets.load_breast_cancer",
        },
        "cross_validation": cv_results,
        "held_out_test": held_out_results,
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    update_readme(build_readme_table(results))
    print(f"Saved real metrics to {RESULTS_PATH.relative_to(ROOT)}")
    print("Updated the evaluation tables in README.md")


if __name__ == "__main__":
    main()
