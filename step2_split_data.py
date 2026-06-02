"""
Step 2: Read preprocessed data and create train/dev/test splits.

Split allocation (Romansh):
  FMR             → train (80%) + test-a (20%, unbalanced)
  Pledari Grond   → train (all)
  RTR Transcripts → dev (balanced) + test-b (balanced) + train (remainder)
  Textbooks       → train (80%) + test-c (20%, unbalanced)
  Theater Plays   → train (all)
  Canton Laws     → train (all)
  Proprietary     → test-d (all, out-of-domain)

With --multilingual, Wikipedia de/fr/it/en data is added:
  Wikipedia       → train (80%) + dev (10%) + test-e (10%), per language

A splits-meta.json file is written to data/03_splits/ indicating whether the
splits include multilingual data, so downstream steps can adapt accordingly.

Output:
  data/03_splits/train/train.tsv
  data/03_splits/dev/dev.tsv
  data/03_splits/test/test_a.tsv  (Romansh, in-domain)
  data/03_splits/test/test_b.tsv  (Romansh, in-domain)
  data/03_splits/test/test_c.tsv  (Romansh, in-domain)
  data/03_splits/test/test_d.tsv  (Romansh, out-of-domain)
  data/03_splits/test/test_e.tsv  (Wikipedia de/fr/it/en — only with --multilingual)
  data/03_splits/splits-meta.json
"""

import argparse
import json
import re
import random
from pathlib import Path

from src.preprocessing import WIKIPEDIA_LANGS
from src.run_log import start_run

SEED = 42
FMR_TEST_RATIO = 0.2
TEXTBOOKS_TEST_RATIO = 0.2
DEV_PER_IDIOM = 200
TEST_B_PER_IDIOM = 200
WIKI_DEV_RATIO = 0.10
WIKI_TEST_E_RATIO = 0.10

ROMANSH_IDIOMS = ["rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader", "rm-rumgr"]

PREPROCESSED_DIR = Path("data/02_preprocessed")
SPLITS_DIR = Path("data/03_splits")

random.seed(SEED)

_PLACEHOLDER_RE = re.compile(r"\$(?:NE|NUM|URL)\$")
_PUNCT_RE = re.compile(r"[^\w \-'']")
_STANDALONE_HYPHEN_RE = re.compile(r"(?<!\S)-|-(?!\S)")


def clean_for_training(text: str) -> str:
    """Remove placeholders, punctuation, and lowercase. Applied to train split only."""
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _STANDALONE_HYPHEN_RE.sub(" ", text)
    text = text.lower()
    return " ".join(text.split())


def load_preprocessed(source: str) -> list[tuple[str, str]]:
    """Read all rm-*.tsv files from a preprocessed source folder."""
    samples = []
    source_dir = PREPROCESSED_DIR / source
    if not source_dir.exists():
        print(f"  [{source}] Not found in preprocessed — skipping.")
        return []
    for path in sorted(source_dir.glob("rm-*.tsv")):
        idiom = path.stem
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if text:
                    samples.append((idiom, text))
    return samples


def save_tsv(samples: list[tuple[str, str]], path: Path, clean: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for label, text in samples:
            if clean:
                text = clean_for_training(text)
            if text:
                f.write(f"{label}\t{text}\n")
                written += 1
    skipped = len(samples) - written
    note = f"  ({skipped:,} empty after cleaning)" if skipped else ""
    print(f"  Saved {written:>8,} samples → {path}{note}")


def split_by_ratio(samples: list[tuple[str, str]], test_ratio: float):
    """Per-idiom ratio-preserving split into (train, test)."""
    by_idiom: dict[str, list] = {}
    for label, text in samples:
        by_idiom.setdefault(label, []).append((label, text))
    train, test = [], []
    for idiom_samples in by_idiom.values():
        random.shuffle(idiom_samples)
        n_test = max(1, int(len(idiom_samples) * test_ratio))
        test.extend(idiom_samples[:n_test])
        train.extend(idiom_samples[n_test:])
    return train, test


def load_preprocessed_wiki(langs: list[str]) -> list[tuple[str, str]]:
    """Read {lang}.tsv files from data/02_preprocessed/wikipedia/."""
    samples = []
    source_dir = PREPROCESSED_DIR / "wikipedia"
    if not source_dir.exists():
        print("  [Wikipedia] Not found in preprocessed — skipping.")
        return []
    for lang in langs:
        path = source_dir / f"{lang}.tsv"
        if not path.exists():
            print(f"  [Wikipedia/{lang}] Not found — skipping.")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if text:
                    samples.append((lang, text))
                    count += 1
        print(f"  [Wikipedia/{lang}] {count:,} samples")
    return samples


def split_wikipedia(samples: list[tuple[str, str]], dev_ratio: float, test_ratio: float):
    """Per-language random 3-way split into (train, dev, test)."""
    by_lang: dict[str, list] = {}
    for label, text in samples:
        by_lang.setdefault(label, []).append((label, text))
    train, dev, test = [], [], []
    for lang_samples in by_lang.values():
        random.shuffle(lang_samples)
        n = len(lang_samples)
        n_test = int(n * test_ratio)
        n_dev = int(n * dev_ratio)
        test.extend(lang_samples[:n_test])
        dev.extend(lang_samples[n_test:n_test + n_dev])
        train.extend(lang_samples[n_test + n_dev:])
    return train, dev, test


def split_rtr_balanced(samples: list[tuple[str, str]], dev_per_idiom: int, test_b_per_idiom: int):
    """Carve out balanced dev and test-b sets; remainder goes to train."""
    by_idiom: dict[str, list] = {}
    for label, text in samples:
        by_idiom.setdefault(label, []).append((label, text))
    dev, test_b, train = [], [], []
    for idiom, idiom_samples in by_idiom.items():
        random.shuffle(idiom_samples)
        n_needed = dev_per_idiom + test_b_per_idiom
        if len(idiom_samples) < n_needed:
            print(f"  [RTR] Warning: {idiom} has only {len(idiom_samples)} samples, "
                  f"need {n_needed} for dev+test-b. Using all for train.")
            train.extend(idiom_samples)
            continue
        dev.extend(idiom_samples[:dev_per_idiom])
        test_b.extend(idiom_samples[dev_per_idiom:dev_per_idiom + test_b_per_idiom])
        train.extend(idiom_samples[dev_per_idiom + test_b_per_idiom:])
    return dev, test_b, train


def write_metadata(multilingual: bool, languages: list[str]):
    meta = {"multilingual": multilingual, "languages": languages}
    path = SPLITS_DIR / "splits-meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {path}")


def main():
    parser = argparse.ArgumentParser(description="Create train/dev/test splits.")
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Include Wikipedia de/fr/it/en data and produce test-e.",
    )
    args = parser.parse_args()

    start_run("step2_split_data")

    train, dev, test_a, test_b, test_c, test_d, test_e = [], [], [], [], [], [], []

    # --- FMR ---
    print("\nLoading FMR from preprocessed...")
    fmr = load_preprocessed("fmr")
    fmr_train, fmr_test_a = split_by_ratio(fmr, FMR_TEST_RATIO)
    train.extend(fmr_train)
    test_a.extend(fmr_test_a)
    print(f"  → {len(fmr_train):,} train / {len(fmr_test_a):,} test-a")

    # --- Pledari Grond ---
    print("\nLoading Pledari Grond from preprocessed...")
    pg = load_preprocessed("pledari-grond")
    train.extend(pg)
    print(f"  → {len(pg):,} train")

    # --- RTR Transcripts ---
    print("\nLoading RTR Transcripts from preprocessed...")
    rtr = load_preprocessed("rtr-transcripts")
    if rtr:
        rtr_dev, rtr_test_b, rtr_train = split_rtr_balanced(rtr, DEV_PER_IDIOM, TEST_B_PER_IDIOM)
        dev.extend(rtr_dev)
        test_b.extend(rtr_test_b)
        train.extend(rtr_train)
        print(f"  → {len(rtr_train):,} train / {len(rtr_dev):,} dev / {len(rtr_test_b):,} test-b")
    else:
        print("  No RTR data — dev and test-b will be empty.")

    # --- Theater Plays ---
    print("\nLoading Theater Plays from preprocessed...")
    tp = load_preprocessed("theater-plays")
    train.extend(tp)
    print(f"  → {len(tp):,} train")

    # --- Canton Laws ---
    print("\nLoading Canton Laws from preprocessed...")
    cl = load_preprocessed("canton-laws")
    train.extend(cl)
    print(f"  → {len(cl):,} train")

    # --- Textbooks ---
    print("\nLoading Textbooks from preprocessed...")
    tb = load_preprocessed("textbooks")
    tb_train, tb_test_c = split_by_ratio(tb, TEXTBOOKS_TEST_RATIO)
    train.extend(tb_train)
    test_c.extend(tb_test_c)
    print(f"  → {len(tb_train):,} train / {len(tb_test_c):,} test-c")

    # --- Proprietary ---
    print("\nLoading Proprietary data from preprocessed...")
    prop = load_preprocessed("proprietary-data")
    test_d.extend(prop)
    print(f"  → {len(prop):,} test-d")

    # --- Wikipedia (multilingual only) ---
    if args.multilingual:
        print("\nLoading Wikipedia (multilingual) from preprocessed...")
        wiki = load_preprocessed_wiki(WIKIPEDIA_LANGS)
        wiki_train, wiki_dev, wiki_test_e = split_wikipedia(wiki, WIKI_DEV_RATIO, WIKI_TEST_E_RATIO)
        train.extend(wiki_train)
        dev.extend(wiki_dev)
        test_e.extend(wiki_test_e)
        print(f"  → {len(wiki_train):,} train / {len(wiki_dev):,} dev / {len(wiki_test_e):,} test-e")

    # --- Shuffle train ---
    random.shuffle(train)

    # --- Save ---
    print("\nSaving splits...")
    save_tsv(train,  SPLITS_DIR / "train/train.tsv", clean=True)
    save_tsv(dev,    SPLITS_DIR / "dev/dev.tsv")
    save_tsv(test_a, SPLITS_DIR / "test/test_a.tsv")
    save_tsv(test_b, SPLITS_DIR / "test/test_b.tsv")
    save_tsv(test_c, SPLITS_DIR / "test/test_c.tsv")
    save_tsv(test_d, SPLITS_DIR / "test/test_d.tsv")
    if args.multilingual:
        save_tsv(test_e, SPLITS_DIR / "test/test_e.tsv")

    languages = list(ROMANSH_IDIOMS)
    if args.multilingual:
        languages.extend(WIKIPEDIA_LANGS)
    write_metadata(args.multilingual, languages)

    print("\nDone.")
    print(f"  train:  {len(train):>8,}")
    print(f"  dev:    {len(dev):>8,}")
    print(f"  test-a: {len(test_a):>8,}")
    print(f"  test-b: {len(test_b):>8,}")
    print(f"  test-c: {len(test_c):>8,}")
    print(f"  test-d: {len(test_d):>8,}")
    if args.multilingual:
        print(f"  test-e: {len(test_e):>8,}  (Wikipedia de/fr/it/en)")
    mode = "multilingual" if args.multilingual else "Romansh-only"
    print(f"\n  Mode: {mode}")


if __name__ == "__main__":
    main()