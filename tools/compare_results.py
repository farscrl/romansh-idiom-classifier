#!/usr/bin/env python3
"""Compare evaluation results across two or more JSON result files.

Each JSON file is produced by step7_evaluate.py (source: python) or
packages/romansh-idiom-classifier/scripts/verify.ts (source: typescript).

Usage:
  python compare_results.py data/04_evaluation/results_SVM.json data/04_evaluation/results_SVM-lite.json
  python compare_results.py results_SVM.json ts_results_svm_lite.json --threshold 0.005
"""

import json
import sys
from pathlib import Path


DIFF_THRESHOLD_DEFAULT = 0.001  # differences below this are shown as "≈"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_diff(delta: float, threshold: float) -> str:
    if abs(delta) < threshold:
        return "  ≈"
    sign = "+" if delta > 0 else ""
    return f" {sign}{delta:+.4f}"


def compare(files: list[str], threshold: float) -> None:
    datasets = [load(p) for p in files]
    names = [f"{d['model']} ({d['source']})" for d in datasets]
    col_w = max(len(n) for n in names) + 2

    all_test_keys = []
    for d in datasets:
        for k in d["test_sets"]:
            if k not in all_test_keys:
                all_test_keys.append(k)

    for tkey in all_test_keys:
        entries = [d["test_sets"].get(tkey) for d in datasets]
        if all(e is None for e in entries):
            continue

        label = next(e["label"] for e in entries if e)
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")

        header = f"  {'Metric':<18}" + "".join(f"  {n:<{col_w}}" for n in names)
        print(header)
        print(f"  {'-'*18}" + "".join(f"  {'-'*col_w}" for _ in names))

        for metric_key, metric_label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]:
            values = [e[metric_key] if e else None for e in entries]
            row = f"  {metric_label:<18}"
            for v in values:
                row += f"  {fmt(v) if v is not None else 'n/a':<{col_w}}"
            # Append diff columns if exactly 2 files
            if len(values) == 2 and all(v is not None for v in values):
                row += fmt_diff(values[1] - values[0], threshold)
            print(row)

        # Per-class F1
        all_classes = []
        for e in entries:
            if e:
                for cls in e.get("per_class", {}):
                    if cls not in all_classes:
                        all_classes.append(cls)

        if all_classes:
            print(f"  {'Per-class F1':<18}")
            for cls in sorted(all_classes):
                values = []
                for e in entries:
                    if e and cls in e.get("per_class", {}):
                        values.append(e["per_class"][cls]["f1"])
                    else:
                        values.append(None)
                row = f"    {cls:<16}"
                for v in values:
                    row += f"  {fmt(v) if v is not None else 'n/a':<{col_w}}"
                if len(values) == 2 and all(v is not None for v in values):
                    row += fmt_diff(values[1] - values[0], threshold)
                print(row)

    print()


def main() -> None:
    args = sys.argv[1:]
    threshold = DIFF_THRESHOLD_DEFAULT

    if "--threshold" in args:
        i = args.index("--threshold")
        threshold = float(args[i + 1])
        args = args[:i] + args[i + 2:]

    if len(args) < 2:
        print("Usage: python compare_results.py <file1.json> <file2.json> [--threshold 0.001]")
        sys.exit(1)

    for p in args:
        if not Path(p).exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    compare(args, threshold)


if __name__ == "__main__":
    main()