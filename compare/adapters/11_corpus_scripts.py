"""Adapter: corpus-scripts — writes data/05_comparison/11_corpus_scripts.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compare.lib import COMPARE_DIR, compute_metrics, load_test_sets, write_result
from compare.models.corpus_scripts.predict import predict_tsv

NAME   = "corpus-scripts"
SOURCE = "python"
OUT    = COMPARE_DIR / "11_corpus_scripts.json"

ROMANSH_IDIOMS = [
    "rm-sursilv", "rm-sutsilv", "rm-surmiran",
    "rm-puter", "rm-vallader", "rm-rumgr",
]


def main():
    results = {}
    for tkey, (label, path) in load_test_sets({"multilingual": False}).items():
        if not path.exists():
            continue
        labels, preds = predict_tsv(path)
        if not labels:
            continue
        classes = [c for c in ROMANSH_IDIOMS if c in set(labels)]
        metrics = compute_metrics(labels, preds, classes, include_cm=True)
        results[tkey] = {"label": label, "n_samples": len(labels), **metrics}
    write_result(OUT, NAME, SOURCE, results)
    print(f"[{NAME}] → {OUT}")


if __name__ == "__main__":
    main()
