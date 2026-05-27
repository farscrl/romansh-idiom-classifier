"""
Verify Python classifier against the test sets.

Runs predictions on one or more test TSV files, prints accuracy and macro-F1,
and optionally saves results to JSON — same format as the TypeScript verify script
so results can be compared with tools/compare_results.py.

Usage:
  python scripts/verify.py --model lr
  python scripts/verify.py --model lr --output ../../data/04_evaluation/results_LR-py.json
  python scripts/verify.py --model lr --tests test_a test_b
  python scripts/verify.py --model ../../models/lr_export.json  # custom path
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from romansh_idiom_classifier import RomanshIdiomClassifier

TEST_SET_LABELS = {
    "test_a": "Test A — news (FMR)",
    "test_b": "Test B — speech transcripts (RTR)",
    "test_c": "Test C — schoolbooks (Textbooks)",
    "test_d": "Test D — proprietary (out-of-domain)",
}

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def macro_f1(true_labels: list[str], pred_labels: list[str]) -> tuple[float, dict[str, float]]:
    classes = sorted(set(true_labels))
    per_class: dict[str, float] = {}
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(true_labels, pred_labels))
        fp = sum(t != cls and p == cls for t, p in zip(true_labels, pred_labels))
        fn = sum(t == cls and p != cls for t, p in zip(true_labels, pred_labels))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        per_class[cls] = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    avg = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return avg, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Python classifier against test sets")
    parser.add_argument("--model", default="lr",
                        help="Bundled model name (lr, lr-lite, svm, svm-lite) or path to JSON export")
    parser.add_argument("--tests", nargs="+", default=["test_a", "test_b", "test_c", "test_d"])
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/03_splits/test"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    classifier = RomanshIdiomClassifier(model=args.model)
    model_name = args.model if args.model in {"lr", "lr-lite", "svm", "svm-lite"} \
                 else Path(args.model).stem.replace("_export", "")
    print(f"Model loaded: {args.model}\n")

    splits_dir = Path(args.splits)
    json_test_sets: dict = {}

    for test_name in args.tests:
        tsv_path = splits_dir / f"{test_name}.tsv"
        if not tsv_path.exists():
            print(f"{test_name}: file not found — skipping\n")
            continue

        true_labels, pred_labels = [], []
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            true_labels.append(label)
            pred_labels.append(classifier.predict(text))

        accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
        mf1, per_class = macro_f1(true_labels, pred_labels)

        print("=" * 60)
        print(f"Test set: {test_name}  ({len(true_labels):,} samples)")
        print(f"  accuracy  = {accuracy:.4f}")
        print(f"  macro-F1  = {mf1:.4f}")
        print("  per-class F1:")
        for cls, f1 in sorted(per_class.items()):
            print(f"    {cls:<16} {f1:.3f}")
        print()

        json_test_sets[test_name] = {
            "label":     TEST_SET_LABELS.get(test_name, test_name),
            "n_samples": len(true_labels),
            "accuracy":  round(accuracy, 6),
            "macro_f1":  round(mf1, 6),
            "per_class": {cls: {"f1": round(f1, 6)} for cls, f1 in per_class.items()},
        }

    if args.output:
        payload = {
            "model":     model_name,
            "source":    "python",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "test_sets": json_test_sets,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
