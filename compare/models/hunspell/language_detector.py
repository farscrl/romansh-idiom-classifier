"""Core language detection via Hunspell spell-checker coverage."""

import re
from pathlib import Path
from typing import Optional

from spylls.hunspell import Dictionary

LANGUAGES: dict[str, str] = {
    "en":          "English",
    "de":          "German",
    "fr":          "French",
    "it":          "Italian",
    "rm-sursilv":  "Romansh Sursilvan",
    "rm-sutsilv":  "Romansh Sutsilvan",
    "rm-surmiran": "Romansh Surmiran",
    "rm-puter":    "Romansh Puter",
    "rm-vallader": "Romansh Vallader",
    "rm-rumgr":    "Rumantsch Grischun",
}

# (subfolder, stem) inside the dictionaries/ directory
_DICT_PATHS: dict[str, tuple[str, str]] = {
    "en":          ("en",          "en_US"),
    "de":          ("de",          "de_DE_frami"),
    "fr":          ("fr_FR",       "fr"),
    "it":          ("it_IT",       "it_IT"),
    "rm-sursilv":  ("rm-sursilv",  "rm-sursilv"),
    "rm-sutsilv":  ("rm-sutsilv",  "rm-sutsilv"),
    "rm-surmiran": ("rm-surmiran", "rm-surmiran"),
    "rm-puter":    ("rm-puter",    "rm-puter"),
    "rm-vallader": ("rm-vallader", "rm-vallader"),
    "rm-rumgr":    ("rm-rumgr",    "rm-rumgr"),
}

MIXED_THRESHOLD = 0.50


class LanguageDetector:
    def __init__(self, dictionaries_path: Path):
        self.dictionaries_path = dictionaries_path
        self._dicts: dict[str, Optional[Dictionary]] = {}
        # word-level cache: word -> {lang_code -> bool}
        self._cache: dict[str, dict[str, bool]] = {}

    def _load(self, lang: str) -> Optional[Dictionary]:
        if lang not in self._dicts:
            folder, stem = _DICT_PATHS[lang]
            path = self.dictionaries_path / folder / stem
            if path.with_suffix(".aff").exists():
                try:
                    self._dicts[lang] = Dictionary.from_files(str(path))
                except Exception as exc:
                    print(f"Warning: failed to load {lang} dictionary: {exc}")
                    self._dicts[lang] = None
            else:
                print(f"Warning: dictionary not found for {lang} at {path}.aff")
                self._dicts[lang] = None
        return self._dicts[lang]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        # Unicode letters; keep internal apostrophes (common in Romansh: ch'el, ün'altra)
        tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", text, re.UNICODE)
        return [t for t in tokens if len(t) >= 2]

    def _check(self, word: str, lang: str) -> bool:
        if word not in self._cache:
            self._cache[word] = {}
        if lang not in self._cache[word]:
            d = self._load(lang)
            try:
                self._cache[word][lang] = bool(d and d.lookup(word))
            except Exception:
                self._cache[word][lang] = False
        return self._cache[word][lang]

    def score_text(self, text: str) -> list[tuple[str, float]]:
        """Return list of (lang, score) sorted by score descending.

        Each word is weighted by 1 / (number of dictionaries that accept it),
        so words unique to one language contribute far more than common words
        that are valid across many dictionaries.  Words not found in any
        dictionary are excluded from both numerator and denominator.
        """
        words = self.tokenize(text)
        lang_weights: dict[str, float] = {lang: 0.0 for lang in LANGUAGES}
        total_weight = 0.0

        for word in words:
            accepted = {lang for lang in LANGUAGES if self._check(word, lang)}
            if not accepted:
                continue
            w = 1.0 / len(accepted)
            total_weight += w
            for lang in accepted:
                lang_weights[lang] += w

        if total_weight == 0:
            return [(lang, 0.0) for lang in LANGUAGES]

        scores = {lang: lang_weights[lang] / total_weight for lang in LANGUAGES}

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def word_breakdown(self, text: str) -> list[dict]:
        """Return per-token analysis: which dictionaries accept each word and its weight.

        Each entry: {"word": str, "accepted": [lang, ...], "weight": float, "not_found": bool}
        Words unique to one language have weight=1.0 and are the most diagnostic.
        Words not found in any dictionary have weight=0.0 and not_found=True.
        """
        lang_order = list(LANGUAGES.keys())
        result = []
        for word in self.tokenize(text):
            accepted = [lang for lang in lang_order if self._check(word, lang)]
            weight = 1.0 / len(accepted) if accepted else 0.0
            result.append({
                "word":      word,
                "accepted":  accepted,
                "weight":    weight,
                "not_found": len(accepted) == 0,
            })
        return result

    def detect(self, text: str, threshold: float = MIXED_THRESHOLD) -> dict:
        scores = self.score_text(text)
        best_lang, best_score = scores[0]
        is_mixed = best_score < threshold

        return {
            "best_lang":  best_lang if not is_mixed else "mixed",
            "best_score": best_score,
            "is_mixed":   is_mixed,
            "scores":     scores,   # sorted list of (lang, score)
        }

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in parts if s.strip()]