"""Adapter: SVM-lite — writes data/05_comparison/02_svm_lite.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
from compare.lib import COMPARE_DIR, compute_metrics, load_split, load_test_sets, write_result
from src.splits import load_splits_meta

NAME   = "SVM-lite"
SOURCE = "python"
MODEL  = Path("models/svm_lite.joblib")
OUT    = COMPARE_DIR / "02_svm_lite.json"


def main():
    meta  = load_splits_meta()
    model = joblib.load(MODEL)
    results = {}
    for tkey, (label, path) in load_test_sets(meta).items():
        texts, labels = load_split(path)
        if not texts:
            continue
        preds   = list(model.predict(texts))
        classes = [c for c in meta["languages"] if c in set(labels)]
        metrics = compute_metrics(labels, preds, classes, include_cm=True)
        results[tkey] = {"label": label, "n_samples": len(texts), **metrics}
    write_result(OUT, NAME, SOURCE, results)
    print(f"[{NAME}] → {OUT}")


if __name__ == "__main__":
    main()
