"""Export a fitted TF-IDF + linear classifier pipeline to JSON for browser inference.

Works with both LinearSVC (L1 → sparse coef_) and LogisticRegression (L2 → dense coef_).
Coefficients are stored as sparse {idx, val} rows (non-zero entries only).
For L2 models nearly all weights are non-zero, so the output is effectively dense,
but the schema and TypeScript inference code are identical for both model types.
"""

import json
from pathlib import Path

import numpy as np


def export_pipeline(pipeline, out_path: Path) -> None:
    """Serialize pipeline to JSON compatible with the TypeScript RomanshIdiomClassifier."""
    feat_union = pipeline.named_steps["features"]
    char_vec = feat_union.transformer_list[0][1]
    word_vec = feat_union.transformer_list[1][1]
    clf = pipeline.named_steps["clf"]

    char_vocab_size = len(char_vec.vocabulary_)
    char_coef = clf.coef_[:, :char_vocab_size]
    word_coef = clf.coef_[:, char_vocab_size:]

    def to_sparse(matrix):
        result = []
        for row in matrix:
            nz = np.nonzero(row)[0]
            result.append({"idx": nz.tolist(), "val": row[nz].tolist()})
        return result

    payload = {
        "classes": clf.classes_.tolist(),
        "char": {
            "ngram_range": list(char_vec.ngram_range),
            "vocabulary": {k: int(v) for k, v in char_vec.vocabulary_.items()},
            "idf": char_vec.idf_.tolist(),
        },
        "word": {
            "ngram_range": list(word_vec.ngram_range),
            "vocabulary": {k: int(v) for k, v in word_vec.vocabulary_.items()},
            "idf": word_vec.idf_.tolist(),
        },
        "char_coef": to_sparse(char_coef),
        "word_coef": to_sparse(word_coef),
        "intercept": clf.intercept_.tolist(),
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    mb = out_path.stat().st_size / 1_048_576
    nnz_pct = 100 * np.count_nonzero(clf.coef_) / clf.coef_.size
    print(f"  → {out_path}  ({mb:.1f} MB, {nnz_pct:.1f}% non-zero coef)")