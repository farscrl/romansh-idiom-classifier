"""
Corpus-scripts prediction wrapper.

Python port of an old PHP heuristic used to sort articles from "La Quotidiana"
(2006) into idioms while building the Romansh corpus. It has no notion of
French/Italian/English and only distinguishes German from the six Romansh
idioms by thresholding the share of idiom-characteristic characters/markers
in the text.

Can be used as an importable module:

    from compare.models.corpus_scripts.predict import predict_tsv, predict_texts

    labels, preds = predict_tsv(Path("data/03_splits/test/test_a.tsv"))
    preds = predict_texts(["Igl isch bia.", "Das ist schön."])

Or as a CLI script (prints JSON to stdout):

    python compare/models/corpus_scripts/predict.py <tsv_file>
    {"labels": [...], "predictions": [...]}
"""

import json
import sys
from pathlib import Path

# Thresholds and marker lists, ported verbatim from the original PHP script.
_TUDESTG = ["ä"]
_TUDESTG_THRESHOLD = 0.4

_LADIN = ["ü", "ö", "s-ch", " pisser "]
_LADIN_THRESHOLD = 0.4

_PUTER = [" ho ", " fer ", " traunter ", "aunt ", " piglier ", "eau", "ted "]
_PUTER_THRESHOLD = 0.04

_SUTSILVAN = ["â", "ù"]
_SUTSILVAN_THRESHOLD = 0.25

_SURMIRAN = [
    "ò ", " angal ", " chegl ", " chest ", " mianc ", " tgossa ",
    " eneda ", " neir ", " ena ", " sen ",
]
_SURMIRAN_THRESHOLD = 0.02

_RG = ["à ", " è ", " èn ", "ì ", " da la ", " dal ", " cha", " qua ", "uai"]
_RG_THRESHOLD = 0.25

_IDIOM_TO_LABEL = {
    "td": "de",
    "pt": "rm-puter",
    "va": "rm-vallader",
    "st": "rm-sutsilv",
    "sm": "rm-surmiran",
    "rg": "rm-rumgr",
    "sr": "rm-sursilv",
}


def _count_signs(text: str, signs: list[str]) -> int:
    stripped = text
    for sign in signs:
        stripped = stripped.replace(sign, "")
    return len(text) - len(stripped)


def _percent_signs(text: str, signs: list[str]) -> float:
    if not text:
        return 0.0
    return _count_signs(text, signs) / (len(text) / 100)


def predict_text(text: str) -> str:
    if _percent_signs(text, _TUDESTG) > _TUDESTG_THRESHOLD:
        idiom = "td"
    elif _percent_signs(text, _LADIN) > _LADIN_THRESHOLD:
        idiom = "pt" if _percent_signs(text, _PUTER) > _PUTER_THRESHOLD else "va"
    elif _percent_signs(text, _SUTSILVAN) > _SUTSILVAN_THRESHOLD:
        idiom = "st"
    elif _percent_signs(text, _SURMIRAN) > _SURMIRAN_THRESHOLD:
        idiom = "sm"
    elif _percent_signs(text, _RG) > _RG_THRESHOLD:
        idiom = "rg"
    else:
        idiom = "sr"
    return _IDIOM_TO_LABEL[idiom]


def predict_texts(texts: list[str]) -> list[str]:
    return [predict_text(t) for t in texts]


def predict_tsv(tsv_path: Path) -> tuple[list[str], list[str]]:
    """Run corpus-scripts detection on a label-tab-text TSV file.

    Returns (labels, predictions). Predictions use project label format.
    """
    labels, texts = [], []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    preds = predict_texts(texts)
    return labels, preds


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python predict.py <tsv_file>", file=sys.stderr)
        sys.exit(1)
    tsv = Path(args[0])
    labels, preds = predict_tsv(tsv)
    print(json.dumps({"labels": labels, "predictions": preds}))
