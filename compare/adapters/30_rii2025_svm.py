"""Adapter: RII-2025 SVM (train, NE masked) — writes data/05_comparison/30_rii2025_svm.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
from compare.lib import COMPARE_DIR, compute_metrics, load_split, load_test_sets, write_result

NAME   = "RII-2025 SVM"
SOURCE = "python"
MODEL  = Path("compare/models/rii_2025/svm_char_word.joblib")
OUT    = COMPARE_DIR / "30_rii2025_svm.json"


def main():
    model = joblib.load(MODEL)
    results = {}
    for tkey, (label, path) in load_test_sets({"multilingual": False}).items():
        texts, labels = load_split(path)
        if not texts:
            continue
        preds   = list(model.predict(texts))
        classes = [c for c in model.classes_ if c in set(labels)]
        metrics = compute_metrics(labels, preds, classes, include_cm=True)
        results[tkey] = {"label": label, "n_samples": len(texts), **metrics}
    write_result(OUT, NAME, SOURCE, results)
    print(f"[{NAME}] → {OUT}")


if __name__ == "__main__":
    main()
