# romansh-idiom-classifier

Python **Romansh idiom classifier**. Given a text in Romansh, it returns which of the six regional idioms it is written in.

| Label | Idiom |
|---|---|
| `rm-sursilv` | Sursilvan |
| `rm-sutsilv` | Sutsilvan |
| `rm-surmiran` | Surmiran |
| `rm-puter` | Puter |
| `rm-vallader` | Vallader |
| `rm-rumgr` | Rumantsch Grischun |

Pure Python inference — no scikit-learn required at runtime. All four trained models are bundled. Trained in [farscrl/romansh-idiom-identification](https://github.com/farscrl/romansh-idiom-identification).

## Installation

```bash
pip install romansh-idiom-classifier
```

## Usage

```python
from romansh_idiom_classifier import RomanshIdiomClassifier

classifier = RomanshIdiomClassifier()  # default: full LR model (best overall accuracy)

idiom = classifier.predict("L'uolp era puspei inagada fomentada. Cheu ha ella viu sin in pégn in tgaper che teneva in toc caschiel en siu bec.")
print(idiom)  # "rm-sursilv"
```

### Choosing a model

```python
classifier = RomanshIdiomClassifier()              # "lr" — full LR, best overall (default)
classifier = RomanshIdiomClassifier(model="svm")   # full SVM, best on schoolbook text
classifier = RomanshIdiomClassifier(model="lr-lite")   # smaller LR (10k/5k vocab)
classifier = RomanshIdiomClassifier(model="svm-lite")  # smallest and fastest
```

| Model | Avg accuracy | Avg macro-F1 | JSON size |
|---|---|---|---|
| `lr` (default) | 0.952 | **0.937** | 28 MB |
| `svm` | 0.952 | 0.928 | 11 MB |
| `lr-lite` | 0.946 | 0.933 | 2.6 MB |
| `svm-lite` | 0.947 | 0.924 | 1.8 MB |

All four models are bundled (~43 MB total, ~15 MB gzipped).

### With raw scores

`score()` returns unbounded real numbers — positive means evidence *for* that idiom, negative means evidence *against*. The gap between the highest and lowest score reflects confidence.

```python
scores = classifier.score("L'uolp era puspei inagada fomentada...")
# { "rm-sursilv": 14.57, "rm-surmiran": 1.31, "rm-sutsilv": -0.79, ... }
```

### With soft scores (for confidence bars)

`soft_scores()` applies softmax and returns values between 0 and 1 summing to 1. Useful for displaying a confidence bar chart. **Not calibrated probabilities** — use `score()` when you need to reason about model certainty.

```python
soft = classifier.soft_scores("L'uolp era puspei inagada fomentada...")
# { "rm-sursilv": ~1.0, "rm-surmiran": ~0.0, ... }
```

### From a custom model file

```python
classifier = RomanshIdiomClassifier(model="path/to/lr_export.json")
```

## API

### `RomanshIdiomClassifier(model=None)`

Creates a classifier. `model` can be:
- `None` or `"lr"` — bundled full LR model (default)
- `"lr-lite"`, `"svm"`, `"svm-lite"` — other bundled models
- A file path string to a JSON export
- A pre-parsed `dict`

### `classifier.predict(text: str) -> str`

Returns the single most likely idiom label.

### `classifier.score(text: str) -> dict[str, float]`

Returns a raw decision score per class. Unbounded reals — positive = evidence for, negative = evidence against.

### `classifier.soft_scores(text: str) -> dict[str, float]`

Returns softmax-normalised scores between 0 and 1, summing to 1. For display use. Not calibrated probabilities.

## Verification

The Python inference is a faithful reimplementation of the sklearn pipeline. Predictions should match the sklearn reference results within the float32/float64 precision gap (~0.001 threshold), and are **bit-for-bit identical** to the TypeScript package on every test set.

### Run the classifier on the test sets

```bash
cd packages/romansh-idiom-classifier-py

# Default model (LR) — save results to JSON
python scripts/verify.py --model lr \
  --output ../../data/04_evaluation/results_LR-py.json

# LR-lite
python scripts/verify.py --model lr-lite \
  --output ../../data/04_evaluation/results_LR-lite-py.json
```

You can restrict to specific test sets:

```bash
python scripts/verify.py --model lr --tests test_a test_b
```

### Compare Python package vs sklearn reference

```bash
# From the repository root
python tools/compare_results.py \
  data/04_evaluation/results_LR.json \
  data/04_evaluation/results_LR-py.json
```

All differences should show `≈` (below the 0.001 threshold). Any larger delta indicates a discrepancy in the inference implementation.

### Compare Python package vs TypeScript package

```bash
python tools/compare_results.py \
  data/04_evaluation/results_LR-lite-ts.json \
  data/04_evaluation/results_LR-lite-py.json
```

Results should be identical — both implementations share the same tokenisation logic.

## License

MIT
