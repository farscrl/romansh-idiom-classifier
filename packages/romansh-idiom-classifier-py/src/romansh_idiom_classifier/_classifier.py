from __future__ import annotations

import importlib.resources as res
import json
import math
from pathlib import Path
from typing import Any

from ._tokenize import char_wb_ngrams, word_ngrams
from ._tfidf import compute_tfidf

BUNDLED_MODELS = {"lr", "lr-lite", "svm", "svm-lite"}
_MODEL_FILENAMES = {
    "lr":       "lr_export.json",
    "lr-lite":  "lr_lite_export.json",
    "svm":      "svm_export.json",
    "svm-lite": "svm_lite_export.json",
}


def _load_bundled(name: str) -> dict:
    filename = _MODEL_FILENAMES[name]
    data = res.files("romansh_idiom_classifier.models").joinpath(filename).read_text(encoding="utf-8")
    return json.loads(data)


class RomanshIdiomClassifier:
    """
    Romansh idiom classifier. Predicts one of: rm-sursilv, rm-sutsilv, rm-surmiran,
    rm-puter, rm-vallader, rm-rumgr.

    Args:
        model: Which model to use. One of:
            - ``"lr"`` (default) — full Logistic Regression, best overall accuracy
            - ``"lr-lite"`` — smaller LR (10k/5k vocab), fast and compact
            - ``"svm"`` — full LinearSVC, best on schoolbook text (Test C)
            - ``"svm-lite"`` — smaller SVM, smallest and fastest
            - A file path string to a custom JSON export
            - A pre-parsed dict
    """

    def __init__(self, model: str | dict | None = None) -> None:
        if model is None or model == "lr":
            data = _load_bundled("lr")
        elif isinstance(model, str) and model in BUNDLED_MODELS:
            data = _load_bundled(model)
        elif isinstance(model, str):
            data = json.loads(Path(model).read_text(encoding="utf-8"))
        elif isinstance(model, dict):
            data = model
        else:
            raise TypeError(f"model must be a name, path, or dict, got {type(model)}")

        self._classes: list[str] = data["classes"]
        self._char = data["char"]
        self._word = data["word"]
        self._char_coef: list[dict] = data["char_coef"]
        self._word_coef: list[dict] = data["word_coef"]
        self._intercept: list[float] = data["intercept"]

    def predict(self, text: str) -> str:
        """Return the single most likely idiom label for the given text."""
        scores = self.score(text)
        return max(scores, key=scores.__getitem__)

    def score(self, text: str) -> dict[str, float]:
        """
        Return a raw decision score per class (higher = more confident).
        Scores are unbounded reals — positive means evidence for that idiom,
        negative means evidence against. Use the gap between scores to judge confidence.
        """
        char_min, char_max = self._char["ngram_range"]
        word_min, word_max = self._word["ngram_range"]

        char_f = compute_tfidf(
            char_wb_ngrams(text, char_min, char_max),
            self._char["vocabulary"],
            self._char["idf"],
        )
        word_f = compute_tfidf(
            word_ngrams(text, word_min, word_max),
            self._word["vocabulary"],
            self._word["idf"],
        )

        result: dict[str, float] = {}
        for c, cls in enumerate(self._classes):
            s = self._intercept[c]
            for idx, val in zip(self._char_coef[c]["idx"], self._char_coef[c]["val"]):
                w = char_f.get(idx)
                if w is not None:
                    s += w * val
            for idx, val in zip(self._word_coef[c]["idx"], self._word_coef[c]["val"]):
                w = word_f.get(idx)
                if w is not None:
                    s += w * val
            result[cls] = s
        return result

    def soft_scores(self, text: str) -> dict[str, float]:
        """
        Return softmax-normalised scores (values between 0 and 1, summing to 1).
        Useful for confidence bars. Not calibrated probabilities — see score() for
        a more informative measure of model certainty.
        """
        raw = self.score(text)
        values = list(raw.values())
        max_v = max(values)
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps)
        return {cls: e / total for cls, e in zip(raw.keys(), exps)}
