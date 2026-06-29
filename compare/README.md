# compare/

This directory contains the comparison infrastructure for step 8. Each adapter script
in `adapters/` evaluates one model and writes a JSON result file to `data/05_comparison/`.
The filename prefix determines display order in the report.

Run all adapters and generate the report:

```bash
python step8_compare.py
```

Regenerate report from existing results without re-running:

```bash
python step8_compare.py --report-only
```

---

## Adapter groups

### 01–04 · This work (SVM / LR, full / lite)

Models trained in steps 3–6 of this pipeline on the full Romansh training set.

| File | Name | Model |
|---|---|---|
| `01_svm.py` | SVM | `models/svm.joblib` |
| `02_svm_lite.py` | SVM-lite | `models/svm_lite.joblib` |
| `03_lr.py` | LR | `models/lr.joblib` |
| `04_lr_lite.py` | LR-lite | `models/lr_lite.joblib` |

The full models use the best hyperparameters found in steps 3/5. The lite variants cap
vocabulary to 10k char / 5k word features for browser inference. All four evaluate on
test sets A–D (and E when multilingual splits are active).

---

### 10 · franc

A Romansh-specialised build of the [franc](https://github.com/wooorm/franc) n-gram language
detector, trained by the ProSvizraRumantscha project to distinguish the six Romansh idioms.
The bundled `all.js` is sourced from:
https://github.com/ProSvizraRumantscha/pledarix/blob/master/webExtension/app/lib/franc-all.js

Files in `compare/models/franc/`. Evaluates on test sets A–D with a Romansh-only whitelist;
on test E (if active) it expands the whitelist to include German, French, Italian and English.

---

### 11 · corpus-scripts

A Python port of an old PHP heuristic used to sort articles from "La Quotidiana" (2006)
while building the Romansh corpus. It thresholds the share of idiom-characteristic
characters/markers (e.g. `ü`/`ö`/`s-ch` for Ladin, `â`/`ù` for Sutsilvan) in the text and
has no notion of French/Italian/English — only German vs. the six Romansh idioms.

Logic is in `compare/models/corpus_scripts/predict.py`. Evaluates on test sets A–D only
(Romansh only).

---

### 20 · hunspell

Dictionary-coverage based detection using [spylls](https://github.com/zverok/spylls)
(a Python Hunspell implementation). Each word is weighted by `1 / n` where `n` is the
number of dictionaries that accept it, rewarding words unique to one language.

Dictionaries and core logic are in `compare/models/hunspell/`. Supports all 10 languages
(6 Romansh idioms + de/fr/it/en). On test sets A–D predictions are restricted to Romansh
idioms; on test E all languages are candidates.


---

### 30–32 · RII-2025 (Charlotte Model's Bachelor's thesis)

> Charlotte Model. *Rumantsch Idiom Identification: Building an Automatic Language
> Identification System.* Bachelor's thesis, University of Zurich, 06.09.2025.
> Examiner: Prof. Dr. Martin Volk. Supervisor: Dr. Jannis Vamvas.

Models located in `compare/models/rii_2025/`. All three are LinearSVC with TF-IDF
char + word features. Named entities were masked during preprocessing (except the
unmasked variant). Evaluates on test sets A–D only (Romansh only).

| File | Name | Training data | NE masking |
|---|---|---|---|
| `30_rii2025_svm.py` | RII-2025 SVM | train | yes |
| `31_rii2025_svm_train_dev.py` | RII-2025 SVM (train+dev) | train + dev | yes |
| `32_rii2025_svm_unmasked.py` | RII-2025 SVM (unmasked) | train | no |

---

### 40–42 · LID-2026 (Model, Ahmadi & Vamvas)

> Charlotte Model, Sina Ahmadi, and Jannis Vamvas. 2026. *Robust Language Identification
> for Romansh Varieties.* In Proceedings of the 11th Swiss Text Analytics Conference,
> pages 101–110, Zurich. ACL.

Models located in `compare/models/lid-2026/`. Same architecture and naming convention
as RII-2025. Evaluates on test sets A–D only (Romansh only).

| File | Name | Training data | NE masking |
|---|---|---|---|
| `40_lid2026_svm.py` | LID-2026 SVM | train | yes |
| `41_lid2026_svm_train_dev.py` | LID-2026 SVM (train+dev) | train + dev | yes |
| `42_lid2026_svm_unmasked.py` | LID-2026 SVM (unmasked) | train | no |

---

## Adding a new adapter

1. Create `adapters/NN_name.py` with the numeric prefix that gives the desired column order.
2. The script must write a JSON file to `data/05_comparison/NN_name.json` using
   `compare.lib.write_result(out_path, name, source, test_sets_data)`.
3. Run `python step8_compare.py` — the new model appears automatically in the report.
