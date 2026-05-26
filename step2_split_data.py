"""
Step 2: Read preprocessed data and create train/dev/test splits.

Split allocation:
  FMR             → train (80%) + test-a (20%, unbalanced)
  Pledari Grond   → train (all)
  RTR Transcripts → dev (balanced) + test-b (balanced) + train (remainder)
  Textbooks       → train (80%) + test-c (20%, unbalanced)
  Theater Plays   → train (all)
  Canton Laws     → train (all)
  Proprietary     → test-d (all, out-of-domain)

Output:
  data/03_splits/train/train.tsv
  data/03_splits/dev/dev.tsv
  data/03_splits/test/test_a.tsv
  data/03_splits/test/test_b.tsv
  data/03_splits/test/test_c.tsv
  data/03_splits/test/test_d.tsv
"""

import re
import random
from pathlib import Path

from src.run_log import start_run

SEED = 42
FMR_TEST_RATIO = 0.2
TEXTBOOKS_TEST_RATIO = 0.2
DEV_PER_IDIOM = 200
TEST_B_PER_IDIOM = 200

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


def main():
    start_run("step2_split_data")

    train, dev, test_a, test_b, test_c, test_d = [], [], [], [], [], []

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

    print("\nDone.")
    print(f"  train:  {len(train):>8,}")
    print(f"  dev:    {len(dev):>8,}")
    print(f"  test-a: {len(test_a):>8,}")
    print(f"  test-b: {len(test_b):>8,}")
    print(f"  test-c: {len(test_c):>8,}")
    print(f"  test-d: {len(test_d):>8,}")


if __name__ == "__main__":
    main()