"""
Step 4: Train final SVM model using best hyperparameters from step 3.

Reads models/svm_best_params.json, trains a TF-IDF + LinearSVC pipeline on the
full training set, saves models/svm.joblib, and also trains a lite variant
(10k char / 5k word vocab) saved to models/svm_lite.joblib.
Both models are exported to JSON for browser inference.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.run_log import start_run
from src.model_export import export_pipeline
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion

SPLITS_DIR = Path("data/03_splits")
MODELS_DIR = Path("models")
PARAMS_PATH = MODELS_DIR / "svm_best_params.json"
MODEL_PATH = MODELS_DIR / "svm.joblib"
MODEL_PATH_LITE = MODELS_DIR / "svm_lite.joblib"

CHAR_LITE = 10_000
WORD_LITE = 5_000

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
    start_run("step4_train_svm", artifacts=[MODEL_PATH, MODEL_PATH_LITE])

    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"{PARAMS_PATH} not found — run step3_optimize_svm.py first.")

    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)
    print(f"Loaded params from {PARAMS_PATH}:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    print("\nLoading training set...")
    train_texts, train_labels = load_split(SPLITS_DIR / "train/train.tsv")
    print(f"  {len(train_texts):,} samples")

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        sublinear_tf=True,
        dtype=np.float32,
        ngram_range=tuple(params["features__char__ngram_range"]),
        max_features=params["features__char__max_features"] or None,
        min_df=params["features__char__min_df"],
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        sublinear_tf=True,
        dtype=np.float32,
        ngram_range=tuple(params["features__word__ngram_range"]),
        max_features=params["features__word__max_features"] or None,
        min_df=params["features__word__min_df"],
    )
    clf = LinearSVC(
        C=params["clf__C"],
        penalty="l1", loss="squared_hinge", dual=False,
        random_state=SEED, max_iter=10_000,
    )

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("char", char_tfidf),
            ("word", word_tfidf),
        ])),
        ("clf", clf),
    ])

    print("\nVectorizing + training SVM...")
    t0 = time.time()

    print("  Step 1/2: building TF-IDF feature matrices...")
    feat_union = pipeline.named_steps["features"]
    X = feat_union.fit_transform(train_texts, train_labels)
    t_vec = time.time() - t0
    char_vocab = len(feat_union.transformer_list[0][1].vocabulary_)
    word_vocab = len(feat_union.transformer_list[1][1].vocabulary_)
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  Vectorization done in {t_vec:.1f}s")
    print(f"    char features: {char_vocab:,}")
    print(f"    word features: {word_vocab:,}")
    print(f"    matrix shape:  {X.shape[0]:,} × {X.shape[1]:,}  (density {density:.4%})")

    print("  Step 2/2: fitting LinearSVC...")
    t_clf = time.time()
    pipeline.named_steps["clf"].fit(X, train_labels)
    elapsed_clf = time.time() - t_clf
    elapsed = time.time() - t0
    print(f"  Classifier done in {elapsed_clf:.1f}s  (total {elapsed:.1f}s)")

    joblib.dump(pipeline, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    print(f"\n  Model saved → {MODEL_PATH} ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    # Phase 2 — export full SVM to JSON                                   #
    # ------------------------------------------------------------------ #
    print("\nExporting full SVM...")
    export_pipeline(pipeline, MODELS_DIR / "svm_export.json")

    # ------------------------------------------------------------------ #
    # Phase 3 — train lite SVM (10k char / 5k word vocab)                #
    # ------------------------------------------------------------------ #
    print(f"\nTraining lite SVM (char={CHAR_LITE:,} / word={WORD_LITE:,} max features)...")
    t_lite = time.time()

    char_tfidf_lite = TfidfVectorizer(
        analyzer="char_wb", sublinear_tf=True, dtype=np.float32,
        ngram_range=tuple(params["features__char__ngram_range"]),
        max_features=CHAR_LITE,
        min_df=params["features__char__min_df"],
    )
    word_tfidf_lite = TfidfVectorizer(
        analyzer="word", sublinear_tf=True, dtype=np.float32,
        ngram_range=tuple(params["features__word__ngram_range"]),
        max_features=WORD_LITE,
        min_df=params["features__word__min_df"],
    )
    clf_lite = LinearSVC(
        C=params["clf__C"], penalty="l1", loss="squared_hinge", dual=False,
        random_state=SEED, max_iter=10_000,
    )
    pipeline_lite = Pipeline([
        ("features", FeatureUnion([
            ("char", char_tfidf_lite),
            ("word", word_tfidf_lite),
        ])),
        ("clf", clf_lite),
    ])

    print("  Step 1/2: building TF-IDF feature matrices...")
    feat_union_lite = pipeline_lite.named_steps["features"]
    X_lite = feat_union_lite.fit_transform(train_texts, train_labels)
    t_vec_lite = time.time() - t_lite
    char_vocab_lite = len(feat_union_lite.transformer_list[0][1].vocabulary_)
    word_vocab_lite = len(feat_union_lite.transformer_list[1][1].vocabulary_)
    density_lite = X_lite.nnz / (X_lite.shape[0] * X_lite.shape[1])
    print(f"  Vectorization done in {t_vec_lite:.1f}s")
    print(f"    char features: {char_vocab_lite:,}")
    print(f"    word features: {word_vocab_lite:,}")
    print(f"    matrix shape:  {X_lite.shape[0]:,} × {X_lite.shape[1]:,}  (density {density_lite:.4%})")

    print("  Step 2/2: fitting LinearSVC...")
    t_clf_lite = time.time()
    pipeline_lite.named_steps["clf"].fit(X_lite, train_labels)
    elapsed_clf_lite = time.time() - t_clf_lite
    elapsed_lite = time.time() - t_lite
    print(f"  Classifier done in {elapsed_clf_lite:.1f}s  (total {elapsed_lite:.1f}s)")

    joblib.dump(pipeline_lite, MODEL_PATH_LITE)
    size_lite_mb = MODEL_PATH_LITE.stat().st_size / 1_048_576
    print(f"\n  Model saved → {MODEL_PATH_LITE} ({size_lite_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    # Phase 4 — export lite SVM to JSON                                   #
    # ------------------------------------------------------------------ #
    print("\nExporting lite SVM...")
    export_pipeline(pipeline_lite, MODELS_DIR / "svm_lite_export.json")


if __name__ == "__main__":
    main()