"""Adapter: hunspell — writes data/05_comparison/20_hunspell.json"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compare.lib import COMPARE_DIR, compute_metrics, load_test_sets, write_result
from compare.models.hunspell.predict import predict_tsv
from src.splits import load_splits_meta

NAME   = "hunspell"
SOURCE = "python"
OUT    = COMPARE_DIR / "20_hunspell.json"

ROMANSH_IDIOMS = [
    "rm-sursilv", "rm-sutsilv", "rm-surmiran",
    "rm-puter", "rm-vallader", "rm-rumgr",
]
OTHER_LANGS = ["de", "fr", "it", "en"]


def main():
    meta = load_splits_meta()

    results = {}
    for tkey, (label, path) in load_test_sets(meta).items():
        if not path.exists():
            continue
        is_multilingual_set = tkey == "test_e"
        candidates = None if is_multilingual_set else ROMANSH_IDIOMS
        labels, preds = predict_tsv(path, candidates=candidates)
        if not labels:
            continue
        candidate_classes = ROMANSH_IDIOMS + (OTHER_LANGS if is_multilingual_set else [])
        classes = [c for c in candidate_classes if c in set(labels)]
        metrics = compute_metrics(labels, preds, classes, include_cm=True)
        results[tkey] = {"label": label, "n_samples": len(labels), **metrics}
    write_result(OUT, NAME, SOURCE, results)
    print(f"[{NAME}] → {OUT}")


if __name__ == "__main__":
    main()
