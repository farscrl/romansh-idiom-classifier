"""Shared utilities for comparison adapters (step 8)."""

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

matplotlib.use("Agg")

COMPARE_DIR = Path("data/05_comparison")
SPLITS_DIR  = Path("data/03_splits")

LABEL_SHORT = {
    "rm-sursilv": "Sursilv",
    "rm-sutsilv": "Sutsilv",
    "rm-surmiran": "Surmiran",
    "rm-puter":    "Puter",
    "rm-vallader": "Vallader",
    "rm-rumgr":    "RumGr",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "en": "English",
}

_BASE_TEST_SETS = {
    "test_a": ("Test A — news (FMR)",                SPLITS_DIR / "test/test_a.tsv"),
    "test_b": ("Test B — speech transcripts (RTR)",  SPLITS_DIR / "test/test_b.tsv"),
    "test_c": ("Test C — schoolbooks (Textbooks)",   SPLITS_DIR / "test/test_c.tsv"),
    "test_d": ("Test D — proprietary (out-of-domain)", SPLITS_DIR / "test/test_d.tsv"),
}
_MULTILINGUAL_TEST_SETS = {
    "test_e": ("Test E — Wikipedia (de/fr/it/en)", SPLITS_DIR / "test/test_e.tsv"),
}


def load_test_sets(meta: dict) -> dict:
    test_sets = dict(_BASE_TEST_SETS)
    if meta.get("multilingual"):
        test_sets.update(_MULTILINGUAL_TEST_SETS)
    return test_sets


def load_split(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    if not path.exists():
        return texts, labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    return texts, labels


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_confusion_matrix(y_true, y_pred, classes: list[str]) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with np.errstate(invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short = [LABEL_SHORT.get(c, c) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            norm_val = cm_norm[i, j]
            color = "white" if norm_val > 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def compute_metrics(labels, preds, classes: list[str], include_cm: bool = True) -> dict:
    """Compute accuracy, macro F1, per-class metrics, and optionally a confusion matrix PNG."""
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="macro", labels=classes, zero_division=0)
    report = classification_report(labels, preds, labels=classes, output_dict=True, zero_division=0)

    per_class = {
        cls: {
            "f1":        round(report[cls]["f1-score"],  6),
            "precision": round(report[cls]["precision"],  6),
            "recall":    round(report[cls]["recall"],     6),
            "support":   int(report[cls]["support"]),
        }
        for cls in classes
    }

    result = {
        "accuracy":  round(acc, 6),
        "macro_f1":  round(f1,  6),
        "per_class": per_class,
    }
    if include_cm:
        result["cm_png_b64"] = _plot_confusion_matrix(labels, preds, classes)
    return result


def write_result(out_path: Path, name: str, source: str, test_sets_data: dict) -> None:
    """Write the standard comparison JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name":      name,
        "source":    source,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_sets": test_sets_data,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
