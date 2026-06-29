"""
Hunspell prediction wrapper.

Can be used as an importable module:

    from compare.models.hunspell.predict import predict_tsv, predict_texts

    labels, preds = predict_tsv(Path("data/03_splits/test/test_a.tsv"))
    labels, preds = predict_tsv(Path("data/03_splits/test/test_a.tsv"),
                                candidates=["rm-sursilv", "rm-sutsilv", ...])
    preds = predict_texts(["Igl isch bia.", "Das ist schön."])

Or as a CLI script (prints JSON to stdout):

    python compare/models/hunspell/predict.py <tsv_file> [--romansh-only]
    {"labels": [...], "predictions": [...]}

Detection works by measuring dictionary coverage: each word is weighted by
1 / (number of dictionaries that accept it), rewarding words unique to one
language. All dictionaries are always loaded for scoring — restricting to a
candidate subset only affects which language wins, not how words are scored.
When a text is below the confidence threshold ("mixed"), the top-scoring
language is still returned rather than abstaining.
"""

import json
import sys
from pathlib import Path

from compare.models.hunspell.language_detector import LanguageDetector

_DICTS_PATH = Path(__file__).parent
_ROMANSH = [
    "rm-sursilv", "rm-sutsilv", "rm-surmiran",
    "rm-puter", "rm-vallader", "rm-rumgr",
]

_detector: LanguageDetector | None = None


def _get_detector() -> LanguageDetector:
    global _detector
    if _detector is None:
        _detector = LanguageDetector(_DICTS_PATH)
    return _detector


def predict_text(text: str, candidates: list[str] | None = None) -> str:
    """Predict the language of a single text.

    If `candidates` is given, only those languages are considered when picking
    the winner. All dictionaries are still loaded for scoring.
    Returns a project label (e.g. 'rm-sursilv', 'de') or 'und' if no candidate
    scores above zero.
    """
    detector = _get_detector()
    scores = detector.score_text(text)  # sorted (lang, score) descending

    if candidates is not None:
        scores = [(lang, score) for lang, score in scores if lang in candidates]

    if not scores or scores[0][1] == 0.0:
        return "und"
    return scores[0][0]


def predict_texts(texts: list[str], candidates: list[str] | None = None) -> list[str]:
    """Predict languages for a list of texts. Returns predictions in project label format."""
    return [predict_text(t, candidates=candidates) for t in texts]


def predict_tsv(
    tsv_path: Path,
    candidates: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Run hunspell detection on a label-tab-text TSV file.

    Returns (labels, predictions). Predictions use project label format.
    Pass `candidates` to restrict predictions to a subset of languages.
    """
    labels, texts = [], []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    preds = predict_texts(texts, candidates=candidates)
    return labels, preds


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print("Usage: python predict.py <tsv_file> [--romansh-only]", file=sys.stderr)
        sys.exit(1)
    tsv = Path(args[0])
    candidates = _ROMANSH if "--romansh-only" in args else None
    labels, preds = predict_tsv(tsv, candidates=candidates)
    print(json.dumps({"labels": labels, "predictions": preds}))
