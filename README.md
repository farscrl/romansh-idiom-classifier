# Romansh Idiom Classifier

Automatic classification of Romansh text into one of its six regional idioms using classical machine learning (SVM and Logistic Regression) with TF-IDF character and word n-gram features.

## Academic Background

The ideas, methodology, and data pipeline in this project are heavily based on the following Bachelor's thesis:

> **Rumantsch Idiom Identification: Building an Automatic Language Identification System**
> Charlotte Model — Bachelor's Thesis in Informatics
> University of Zurich, Faculty of Business, Economics and Informatics
> Examiner: Prof. Dr. Martin Volk · Supervisor: Dr. Jannis Vamvas
> Submitted: 06.09.2025

The code and data pipeline in this repository were independently re-implemented from scratch to ensure the project can be published as open source, without any dependency on artefacts that are non-public.

## Idioms

| Label | Idiom |
|---|---|
| `rm-sursilv` | Sursilvan |
| `rm-sutsilv` | Sutsilvan |
| `rm-surmiran` | Surmiran |
| `rm-puter` | Puter |
| `rm-vallader` | Vallader |
| `rm-rumgr` | Rumantsch Grischun |

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation

Before running the pipeline, you need to obtain all data sources:

1. **Run step 0** to download the publicly available sources automatically:
   ```bash
   python step0_download_public_data.py
   ```

2. **Manually place** the following files in the corresponding folders:

   | Source | Folder | Instructions |
   |---|---|---|
   | Pledari Grond | `data/01_raw/pledari-grond/` | See [Data Sources → Pledari Grond](#pledari-grond) below |
   | RTR Transcripts | `data/01_raw/rtr-transcripts/` | See [Data Sources → RTR Transcripts](#rtr-transcripts) below |
   | Proprietary data | `data/01_raw/proprietary-data/` | Optional — skipped if empty |

3. **Optionally create an umlaut allowlist** at `data/01_raw/umlaut_allowlist.txt`:

   After running `step1_preprocess.py` for the first time, inspect
   `data/02_preprocessed/umlaut_triggers.tsv` to see which words triggered sentence removals.
   Add legitimate place names or proper nouns (one per line) to the allowlist so they survive
   as `$NE$` instead of causing the sentence to be dropped. Lines starting with `#` are comments.

   ```
   # Romansh place names and proper nouns containing German-script characters
   Val Müstair
   Zürich
   ```

## Pipeline

Once all data is in place, run the steps in order:

```bash
python step0_download_public_data.py   # download publicly available data from HuggingFace
python step1_preprocess.py             # clean and normalize text → data/02_preprocessed/
python step1z_validate.py             # inspect preprocessed data, generate HTML report
python step2_split_data.py             # create train/dev/test splits → data/03_splits/
python step3_optimize_svm.py           # find best SVM hyperparameters via randomized search
python step4_train_svm.py              # train SVM (full + lite), export JSON for browser
python step5_optimize_lr.py            # find best LR hyperparameters via randomized search
python step6_train_lr.py               # train LR (full + lite), export JSON for browser
python step7_evaluate.py               # evaluate all four models, generate HTML report
```

> **Tips:**
> - After step 1, inspect `data/02_preprocessed/umlaut_report.html` to review removed sentences, and `data/02_preprocessed/umlaut_triggers.tsv` to find candidates for the allowlist.
> - After step 1z, open `data/02_preprocessed/validation_report.html` to verify the data before committing to the splits.
> - Steps 3 and 5 (hyperparameter search) each take several hours. All steps write timestamped logs to `logs/<step>/<timestamp>/run.log`, with key output artifacts copied alongside.

---

## Packages

### npm (browser / Node.js)

```bash
npm add @farscrl/romansh-idiom-classifier
```

```typescript
import { RomanshIdiomClassifier } from "@farscrl/romansh-idiom-classifier";

const classifier = new RomanshIdiomClassifier();
const idiom = classifier.predict("Enstagl ha el sbaglià la via e s'ha pers en la cità.");
// → "rm-sursilv"
```

The LR-lite model is bundled directly — no server or separate model file needed. See [`packages/romansh-idiom-classifier-npm/`](packages/romansh-idiom-classifier-npm/) for full documentation.

### pip (Python)

```bash
pip install romansh-idiom-classifier
```

```python
from romansh_idiom_classifier import RomanshIdiomClassifier

classifier = RomanshIdiomClassifier()          # full LR model (best overall accuracy)
idiom = classifier.predict("Enstagl ha el sbaglià la via e s'ha pers en la cità.")
# → "rm-sursilv"
```

All four trained models are bundled. No scikit-learn required at runtime — pure Python + numpy inference. See [`packages/romansh-idiom-classifier-py/`](packages/romansh-idiom-classifier-py/) for full documentation.

---

## Models

Both classifiers use the same TF-IDF feature pipeline:

- **Character n-grams** (`char_wb` analyzer): sequences of characters within word boundaries. Captures spelling patterns and morphological endings that differ between idioms.
- **Word n-grams**: whole-word unigrams and optionally bigrams. Captures vocabulary differences.
- Both vectorizers use sublinear TF scaling (`1 + log(tf)`) and 32-bit floats to reduce memory.

Four model variants are trained and evaluated:

| Model | File | Vocab | Key properties |
|---|---|---|---|
| **SVM** | `models/svm.joblib` | 100k char + 100k word | LinearSVC, L1 penalty. Sparse weights (~10–20% non-zero). |
| **SVM-lite** | `models/svm_lite.joblib` | 10k char + 5k word | Same as SVM, smaller vocab. Exported to `svm_lite_export.json` for browser use. |
| **LR** | `models/lr.joblib` | 100k char + 50k word | Logistic Regression, L2 penalty, lbfgs solver. |
| **LR-lite** | `models/lr_lite.joblib` | 10k char + 5k word | Same as LR, smaller vocab. Exported to `lr_lite_export.json` for browser use. |

### Hyperparameter search (steps 3 and 5)

Following the methodology of Charlotte Model's thesis, the search is performed on a **stratified 20% subset** of the training data (not the full set) to keep compute tractable. Each of the 30 candidate configurations is evaluated with **5-fold stratified cross-validation**, optimising **macro-F1**. The best configuration is written to `models/svm_best_params.json` / `models/lr_best_params.json`.

Search space for both models:

| Parameter | Values searched |
|---|---|
| Char n-gram range | (1,3), (1,4), (1,5), (1,6) |
| Char max features | 100k, 200k |
| Char min\_df | 1, 2 |
| Word n-gram range | (1,1), (1,2) |
| Word max features | 50k, 100k |
| Word min\_df | 1, 2 |
| SVM C | log-uniform 0.01–4.0 |
| LR C | log-uniform 0.01–2.0 |
| LR penalty | l1, l2, elasticnet |

### Training and evaluation (steps 4, 6, 7)

Steps 4 and 6 train the final models on the **full training set** using the best hyperparameters found in steps 3 and 5 respectively. Step 7 evaluates both models on all test sets and writes a self-contained HTML report to `data/04_evaluation/report.html` with accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix heatmaps.

---

## Data Sources

### FMR (Fundaziun Medias Rumantschas) — *publicly available*

News texts produced by *Fundaziun Medias Rumantschas (FMR)*, published 2021–2025. FMR is the author of most texts appearing in *La Quotidiana*; the HuggingFace dataset is named after the newspaper but the actual content originates from FMR.

- **Source:** [ZurichNLP/quotidiana](https://huggingface.co/datasets/ZurichNLP/quotidiana) (subset `2021_2025`)
- **License:** CC BY 4.0
- **Downloaded automatically** by `step0_download_public_data.py` into `data/01_raw/fmr/`
- **Idioms:** all 6

---

### Pledari Grond / dicziunaris ladins — *partially public, partially requires permission*

Dictionary entries from the Romansh dictionary *Pledari Grond* and form *dicziunaris ladins*. Export 6 JSON files (one per idiom) and place them in `data/01_raw/pledari-grond/` with filenames:

```
pledarigrond_export_json_sursilvan.json
pledarigrond_export_json_sutsilvan.json
pledarigrond_export_json_surmiran.json
pledarigrond_export_json_rumantschgrischun.json
pledarigrond_export_json_puter.json
pledarigrond_export_json_vallader.json
```

- **Sursilvan, Sutsilvan, Surmiran, Rumantsch Grischun:** export from [pledarigrond.ch](https://www.pledarigrond.ch) (free)
- **Puter, Vallader:** contact [Uniun dals Grischs](https://www.udg.ch) for permission
- **Idioms:** all 6

---

### RTR Transcripts — *requires registration*

Transcripts of radio and TV broadcasts from RTR (Radiotelevisiun Svizra Rumantscha).

1. Register at [developer.srgssr.ch](https://developer.srgssr.ch/apis/rtr-linguistic#documentation)
2. Download the transcripts
3. Extract and place the folders in `data/01_raw/rtr-transcripts/`

> **Tip:** Each extracted folder contains a `clips/` subdirectory with audio files. These are not needed for this project and can be safely deleted to save disk space.

- **Idioms:** all 6

---

### Proprietary Data — *not publicly available (optional)*

Any proprietary or restricted Romansh text data you have access to can be placed here and will be included in the pipeline.

Place one or more `.tsv` files in `data/01_raw/proprietary-data/`. Each file must have two tab-separated columns with no header row:

```
rm-sursilv	Questa notg entscheivi a plover...
rm-rumgr	Il Cussegl federal ha decidì oz che...
```

Column 1 is the idiom label (e.g. `rm-sursilv`), column 2 is the text. **If the directory contains no `.tsv` files, this source is silently skipped** — all other steps will complete normally.

- **Idioms:** any subset of the 6 idioms

---

### Theater Plays — *publicly available*

Texts from Romansh theater plays.

- **Source:** [ZurichNLP/romansh_theater_plays](https://huggingface.co/datasets/ZurichNLP/romansh_theater_plays)
- **License:** CC0 1.0
- **Downloaded automatically** by `step0_download_public_data.py` into `data/01_raw/theater-plays/`
- **Idioms:** all 6

---

### Canton Laws — *publicly available*

Cantonal laws of the Canton of Grisons (Graubünden), extracted from [gr-lex.gr.ch](https://www.gr-lex.gr.ch). The dataset contains parallel text in German, Romansh (Rumantsch Grischun), and Italian; only the Romansh column (`RM`) is used. 

- **Source:** [ZurichNLP/romansh-canton-laws](https://huggingface.co/datasets/ZurichNLP/romansh-canton-laws)
- **License:** CC0 1.0
- **Downloaded automatically** by `step0_download_public_data.py` into `data/01_raw/canton-laws/`
- **Idioms:** `rm-rumgr` only

---

### Textbooks — *publicly available*

Texts from Romansh schoolbooks (years 2–9), from the *Mediomatix* project.

- **Source:** [ZurichNLP/mediomatix-raw](https://huggingface.co/datasets/ZurichNLP/mediomatix-raw)
- **License:** CC BY-NC-SA 4.0
- **Downloaded automatically** by `step0_download_public_data.py`
- **Idioms:** 5 (no `rm-rumgr`)

The dataset comes with predefined splits by school year: `train` (years 2–3), `validation` (year 4), `test` (year 5), and `no_surmiran` (years 6–9). The `no_surmiran` split exists for all idioms except Surmiran, where those grade levels were not available. We ignore these predefined splits and pool all data together before creating our own train/dev/test split.

---

## Data Splits

Produced by `step2_split_data.py` from `data/02_preprocessed/` into `data/03_splits/`. All splits use the format `label\ttext` (tab-separated, one sample per line). The random seed is fixed at 42 for reproducibility.

### `train/train.tsv` — Training set

| Source | Allocation | Balance |
|---|---|---|
| FMR | 80% of source | Unbalanced (reflects natural distribution) |
| Pledari Grond | 100% of source | Unbalanced |
| RTR Transcripts | Remainder after dev + test-b carved out | Unbalanced |
| Textbooks | 80% of source | Unbalanced |
| Theater Plays | 100% of source | Unbalanced |
| Canton Laws | 100% of source | rm-rumgr only |

The training set is shuffled after assembly. It additionally receives the training-only text cleanup (lowercasing, punctuation removal, placeholder removal — see [Data Cleanup](#data-cleanup) below).

### `dev/dev.tsv` — Development set

| Source | Allocation | Balance |
|---|---|---|
| RTR Transcripts | 200 samples per idiom | **Balanced** (exactly 200 × 6 idioms) |

Used for hyperparameter tuning and model selection during development. Balanced across idioms to give equal weight to each class. Drawn from the same domain as test-b (speech transcripts).

### `test/test_a.tsv` — Test set A: news (FMR, in-domain)

| Source | Allocation | Balance |
|---|---|---|
| FMR | 20% of source | Unbalanced (reflects natural distribution) |

In-domain test set: same source and text style as the FMR portion of training data. Tests generalisation within a well-represented domain. Unbalanced — idioms with more FMR articles have more test samples.

### `test/test_b.tsv` — Test set B: speech transcripts (RTR, in-domain)

| Source | Allocation | Balance |
|---|---|---|
| RTR Transcripts | 200 samples per idiom | **Balanced** (exactly 200 × 6 idioms) |

In-domain test set from the same source as the dev set (speech transcripts), but a disjoint sample. Balanced across idioms. Tests performance on transcribed spoken language, which tends to be shorter and more colloquial than written text.

### `test/test_c.tsv` — Test set C: schoolbooks (Textbooks, in-domain)

| Source | Allocation | Balance |
|---|---|---|
| Textbooks | 20% of source | Unbalanced (reflects natural distribution) |

In-domain test set: same source as the textbook portion of training. Tests generalisation to educational text (years 2–9). Unbalanced; no Rumantsch Grischun samples (not present in the Mediomatix dataset).

### `test/test_d.tsv` — Test set D: proprietary data (out-of-domain)

| Source | Allocation | Balance |
|---|---|---|
| Proprietary data | 100% of source | Depends on available data |

Out-of-domain test set. None of this data is used for training, making it the most realistic measure of how the model performs on unseen text types. Only present if `data/01_raw/proprietary-data/` contains `.tsv` files; otherwise this file is empty.

---

## Data Cleanup

### Per-source steps (step 1)

**FMR**
- Editorial signatures at the end of articles (e.g. `(cdm/fmr)`, `(cdo/rtr)`) are stripped.

**Pledari Grond**
- Entries whose headword (`rmStichwort`) begins with `cf. ` (cross-references to other entries) are skipped entirely, including their inflection forms and examples.

**Textbooks**
- Exercise labels (`1)`, `2a)`, `c)` etc.) are removed.
- Fill-in-the-blank markers (`_____`, `.....`) are removed.

### Applied to all texts (step 1)

- HTML tags stripped; HTML entities decoded (`&amp;` → `&`, `&nbsp;` → space, etc.).
- URLs replaced with `$URL$`.
- Digit sequences replaced with `$NUM$`.
- Invisible characters removed: soft hyphen (U+00AD), zero-width characters (U+200B–U+200D), BOM (U+FEFF), line/paragraph separators (U+2028–U+2029).
- Non-breaking spaces (U+00A0) normalized to regular spaces.
- Unicode normalization: NFKD → combining dot-below (U+0323) removed → NFC.
- Whitespace collapsed to single spaces.
- Texts containing no alphabetic characters are discarded.

### Umlaut filtering (step 1, after all sources are written)

Texts are split into sentences at `.`, `?`, `!` boundaries. Sentences containing characters foreign to the idiom are removed; if the entire text is foreign, it is dropped:

| Idiom | Filtered characters |
|---|---|
| rm-sursilv, rm-sutsilv, rm-surmiran, rm-rumgr | ä ö ü Ä Ö Ü |
| rm-puter, rm-vallader | ä Ä (ö/ü are native to these idioms) |

Terms listed in `data/01_raw/umlaut_allowlist.txt` are replaced with `$NE$` before this filter runs, so sentences containing only allowlisted proper nouns are preserved.

### Cross-source deduplication (step 1, after all sources are written)

- Texts appearing in multiple sources with the **same** idiom: kept only in the highest-priority source. Priority order (highest → lowest): proprietary-data, fmr, rtr-transcripts, textbooks, theater-plays, canton-laws, pledari-grond.
- Texts appearing with **different** idiom labels across sources: removed from all sources (contradictory label).

### Training-only cleanup (step 2)

Applied only to `train.tsv`; dev and test sets retain the preprocessed form so evaluation reflects realistic input:

- Placeholders (`$NE$`, `$NUM$`, `$URL$`) removed.
- Punctuation removed (letters, digits, spaces, hyphens inside words, and apostrophes are kept).
- Lowercased.
- Texts that become empty after cleanup are discarded.