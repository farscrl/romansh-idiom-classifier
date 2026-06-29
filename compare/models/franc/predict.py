"""
franc prediction wrapper.

Can be used as an importable module:

    from compare.models.franc.predict import predict_tsv, predict_texts

    labels, preds = predict_tsv(Path("data/03_splits/test/test_a.tsv"))
    labels, preds = predict_tsv(Path("data/03_splits/test/test_e.tsv"), multilingual=True)
    preds = predict_texts(["Igl isch bia.", "Das ist schön."], multilingual=True)

Or as a CLI script (prints JSON to stdout):

    python compare/models/franc/predict.py <tsv_file> [--multilingual]
    {"labels": [...], "predictions": [...]}

Multilingual mode expands the franc whitelist to include de/fr/it/en in addition to
the six Romansh idiom codes. Without --multilingual, only Romansh codes are candidates.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_FRANC_JS = Path(__file__).parent / "franc_predict.js"


def _run_franc(tsv_path: Path, multilingual: bool = False) -> dict:
    cmd = ["node", str(_FRANC_JS), str(tsv_path)]
    if multilingual:
        cmd.append("--multilingual")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def predict_tsv(tsv_path: Path, multilingual: bool = False) -> tuple[list[str], list[str]]:
    """Run franc on a label-tab-text TSV file. Returns (labels, predictions).

    Predictions use project label format (e.g. 'rm-sursilv', 'de'). Texts franc
    cannot identify are returned as 'und'.
    """
    data = _run_franc(tsv_path, multilingual=multilingual)
    labels      = data["labels"]
    predictions = [p if p is not None else "und" for p in data["predictions"]]
    return labels, predictions


def predict_texts(texts: list[str], multilingual: bool = False) -> list[str | None]:
    """Run franc on a list of plain texts. Returns predictions in project label format.

    None means franc returned 'und' (undetermined).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", encoding="utf-8", delete=False
    ) as f:
        for text in texts:
            f.write(f"_\t{text}\n")
        tmp = Path(f.name)
    try:
        data = _run_franc(tmp, multilingual=multilingual)
        return data["predictions"]
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print("Usage: python predict.py <tsv_file> [--multilingual]", file=sys.stderr)
        sys.exit(1)
    tsv = Path(args[0])
    multi = "--multilingual" in args
    labels, preds = predict_tsv(tsv, multilingual=multi)
    print(json.dumps({"labels": labels, "predictions": preds}))
