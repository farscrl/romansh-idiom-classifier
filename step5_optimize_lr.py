"""
Step 5: Hyperparameter search for Logistic Regression classifier.

Methodology follows Charlotte Model's Bachelor's thesis:
  - Search on a stratified 20% subset of the training data
  - 5-fold stratified cross-validation
  - Macro-F1 scoring
  - Penalty: L2; solver fixed to lbfgs (fast convergence on dense L2 problems)
  - C range 0.01–2.0 (log-uniform)
  - Char n-grams (1,3)/(1,4), word n-grams (1,1)/(1,2), min_df 1/2

Best params written to models/lr_best_params.json.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.run_log import start_run

SPLITS_DIR = Path("data/03_splits")
MODELS_DIR = Path("models")
PARAMS_PATH = MODELS_DIR / "lr_best_params.json"

SEARCH_SUBSET = 0.20   # fraction of training data used for the search
N_ITER = 30
CV_FOLDS = 5
SEED = 42


def load_split(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    return texts, labels


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    start_run("step5_optimize_lr", artifacts=[PARAMS_PATH])

    print("Loading training set...")
    train_texts, train_labels = load_split(SPLITS_DIR / "train/train.tsv")
    print(f"  full train: {len(train_texts):,}")

    # Stratified 20% subset — same methodology as Charlotte Model's thesis
    search_texts, _, search_labels, _ = train_test_split(
        train_texts, train_labels,
        train_size=SEARCH_SUBSET,
        random_state=SEED,
        stratify=train_labels,
    )
    print(f"  search subset ({SEARCH_SUBSET:.0%}): {len(search_texts):,}")

    char_tfidf = TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, dtype=np.float32)
    word_tfidf = TfidfVectorizer(analyzer="word", sublinear_tf=True, dtype=np.float32)

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("char", char_tfidf),
            ("word", word_tfidf),
        ])),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED)),
    ])

    param_dist = {
        "features__char__ngram_range": [(1, 3), (1, 4), (1, 5), (1, 6)],
        "features__char__max_features": [100_000, 200_000],
        "features__char__min_df": [1, 2],
        "features__word__ngram_range": [(1, 1), (1, 2)],
        "features__word__max_features": [50_000, 100_000],
        "features__word__min_df": [1, 2],
        "clf__C": loguniform(1e-2, 2.0),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    param_samples = list(ParameterSampler(param_dist, n_iter=N_ITER, random_state=SEED))

    print(f"\nStarting search: {N_ITER} iterations × {CV_FOLDS}-fold CV = {N_ITER * CV_FOLDS} fits")
    print(f"Scoring: macro-F1\n")

    t0 = time.time()
    best_score = -1.0
    best_params = {}
    results = []

    for i, params in enumerate(param_samples, 1):
        t_iter = time.time()
        char_range = params["features__char__ngram_range"]
        word_range = params["features__word__ngram_range"]
        char_feats = params["features__char__max_features"] or "all"
        c_val = params["clf__C"]
        print(f"[{i:2d}/{N_ITER}] char={char_range} word={word_range} "
              f"char_feats={char_feats} C={c_val:.4f}")

        candidate = clone(pipeline)
        candidate.set_params(**params)
        fold_scores = cross_val_score(candidate, search_texts, search_labels,
                                      cv=cv, scoring="f1_macro")
        mean_f1 = fold_scores.mean()
        elapsed_iter = time.time() - t_iter
        marker = " ★ NEW BEST" if mean_f1 > best_score else ""
        print(f"         → F1: {fold_scores} | mean={mean_f1:.4f} ({elapsed_iter:.0f}s){marker}")

        results.append((mean_f1, params))
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    elapsed = time.time() - t0
    print(f"\nSearch completed in {elapsed:.1f}s")
    print(f"\nTop 5 configurations:")
    for rank, (score, p) in enumerate(sorted(results, reverse=True)[:5], 1):
        print(f"  {rank}. macro-F1={score:.4f}  char={p['features__char__ngram_range']} "
              f"word={p['features__word__ngram_range']} C={p['clf__C']:.4f}")
    print(f"\nBest macro-F1: {best_score:.4f}")
    print(f"Best params:   {best_params}")

    best = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in best_params.items()
    }
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved best params → {PARAMS_PATH}")


if __name__ == "__main__":
    main()