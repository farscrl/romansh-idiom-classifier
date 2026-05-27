# @farscrl/romansh-idiom-classifier

Browser-side **Romansh idiom classifier**. Given a text in Romansh, it returns which of the six regional idioms it is written in.

| Label | Idiom |
|---|---|
| `rm-sursilv` | Sursilvan |
| `rm-sutsilv` | Sutsilvan |
| `rm-surmiran` | Surmiran |
| `rm-puter` | Puter |
| `rm-vallader` | Vallader |
| `rm-rumgr` | Rumantsch Grischun |

The classifier runs entirely in the browser — no server required. It is a pure TypeScript reimplementation of the linear TF-IDF + SVM/LR pipeline trained in [farscrl/romansh-idiom-identification](https://github.com/farscrl/romansh-idiom-identification).

## Installation

```bash
npm add @farscrl/romansh-idiom-classifier
```

## Usage

The model is bundled with the package — no separate model file needed.

```typescript
import { RomanshIdiomClassifier } from "@farscrl/romansh-idiom-classifier";

const classifier = new RomanshIdiomClassifier();

const idiom = classifier.predict("L’uolp era puspei inagada fomentada. Cheu ha ella viu sin in pégn in tgaper che teneva in toc caschiel en siu bec. Quei gustass a mi, ha ella tertgau, ed ha clamau al tgaper: «Tgei bi che ti eis! Sche tiu cant ei aschi bials sco tia cumparsa, lu eis ti il pli bi utschi da tuts.»");
console.log(idiom); // "rm-sursilv"
```

Works in browsers and Node.js >= 22 without any additional configuration.

### With raw scores

`score()` returns unbounded real numbers — positive means evidence *for* that idiom, negative means evidence *against*. The gap between the highest and lowest score reflects how confident the model is: a large spread (e.g. 14 vs −7) means a clear prediction; a small spread means the text is ambiguous or very short.

```typescript
const scores = classifier.score("L’uolp era puspei inagada fomentada. Cheu ha ella viu sin in pégn in tgaper che teneva in toc caschiel en siu bec. Quei gustass a mi, ha ella tertgau, ed ha clamau al tgaper: «Tgei bi che ti eis! Sche tiu cant ei aschi bials sco tia cumparsa, lu eis ti il pli bi utschi da tuts.»");
// { "rm-sursilv": 14.44, "rm-surmiran": 1.21, "rm-sutsilv": -0.88, "rm-vallader": -3.19, "rm-rumgr": -4.61, "rm-puter": -6.98 }
```

### With soft scores (for confidence bars)

`softScores()` applies softmax to the raw scores and returns values between 0 and 1 that sum to 1. This is useful for displaying a confidence bar chart. **These are not calibrated probabilities** — softmax on linear scores tends to look very confident even for borderline predictions. Use `score()` when you need to reason about model certainty.

```typescript
const soft = classifier.softScores("L’uolp era puspei inagada fomentada. Cheu ha ella viu sin in pégn in tgaper che teneva in toc caschiel en siu bec. Quei gustass a mi, ha ella tertgau, ed ha clamau al tgaper: «Tgei bi che ti eis! Sche tiu cant ei aschi bials sco tia cumparsa, lu eis ti il pli bi utschi da tuts.»");
// { "rm-sursilv": 1.0, "rm-surmiran": 1.8e-6, "rm-sutsilv": 2.2e-7, ... }
```

### From a pre-parsed model object

If you have already loaded the model JSON yourself (e.g. via a custom fetch or CDN):

```typescript
const classifier = new RomanshIdiomClassifier(modelJson);
```

## API

### `new RomanshIdiomClassifier(model?: ModelData): RomanshIdiomClassifier`

Creates a classifier. Without arguments, uses the bundled LR-lite model. Pass a parsed model object to use a custom model.

### `classifier.predict(text: string): string`

Returns the single most likely idiom label for the given text.

### `classifier.score(text: string): Record<string, number>`

Returns a raw decision score per class. Unbounded real numbers — positive means evidence for that idiom, negative means evidence against. Use the gap between scores to judge confidence.

### `classifier.softScores(text: string): Record<string, number>`

Returns softmax-normalised scores between 0 and 1, summing to 1. Convenient for rendering confidence bars. Not calibrated probabilities — see note above.

## Verification

The TypeScript inference is a faithful reimplementation of the Python pipeline, so predictions should be **bit-for-bit identical** to the Python model. The repository includes a verification script to confirm this.

### Prerequisites

Re-run the Python training pipeline to produce the model JSON files:

```bash
# From the repository root
python step4_train_svm.py   # produces models/svm_lite_export.json
python step6_train_lr.py    # produces models/lr_lite_export.json
python step7_evaluate.py    # produces data/04_evaluation/results_*.json
```

### Run the TypeScript classifier on the test sets

```bash
cd packages/romansh-idiom-classifier

# SVM lite — save results to JSON
pnpm verify ../../models/svm_lite_export.json \
  --output ../../data/04_evaluation/results_SVM-lite-ts.json

# LR lite
pnpm verify ../../models/lr_lite_export.json \
  --output ../../data/04_evaluation/results_LR-lite-ts.json
```

You can restrict to specific test sets:

```bash
pnpm verify ../../models/svm_lite_export.json --tests test_a test_b
```

### Compare TypeScript vs Python results

```bash
# From the repository root
python tools/compare_results.py \
  data/04_evaluation/results_SVM-lite.json \
  data/04_evaluation/results_SVM-lite-ts.json
```

All differences should show `≈` (below the 0.001 threshold). Any non-zero delta indicates a discrepancy in the inference implementation.

### Compare two Python models

```bash
python tools/compare_results.py \
  data/04_evaluation/results_SVM-lite.json \
  data/04_evaluation/results_LR-lite.json
```

---

## How it works

The classifier replicates scikit-learn's `TfidfVectorizer` + `LinearSVC` / `LogisticRegression` pipeline exactly:

1. **Character n-grams** (`char_wb`): each token is padded with spaces before extracting n-grams, so n-grams never cross word boundaries.
2. **Word n-grams**: lowercased word unigrams.
3. **Sublinear TF-IDF**: `(1 + log(tf)) × idf`, L2-normalised independently per vectorizer.
4. **Linear scoring**: `score = char_features · char_coef + word_features · word_coef + intercept`.

The model weights are stored in a compact JSON format and loaded once on startup.

## License

MIT