"""
Downloads all publicly available data sources from HuggingFace.

Run this once before step1_preprocess.py:
    python step0_download_public_data.py

Sources downloaded:
  - FMR (2021-2025):           ZurichNLP/quotidiana, subset 2021_2025
  - Textbooks:                 ZurichNLP/mediomatix-raw, all 5 idiom subsets
  - Theater Plays:             ZurichNLP/romansh_theater_plays
  - Canton Laws:               ZurichNLP/romansh-canton-laws (rm-rumgr only)
  - Wikipedia (de/fr/it/en):   wikimedia/wikipedia, streamed paragraphs

Sources that must be obtained manually (see README.md):
  - Pledari Grond:     export JSON files from pledarigrond.ch (+ permission for Puter/Vallader)
  - RTR Transcripts:   download from developer.srgssr.ch
  - Proprietary data:  not publicly available (optional — ignored if missing)
"""

import json
from pathlib import Path
from datasets import load_dataset

from src.run_log import start_run

FMR_DIR = Path("data/01_raw/fmr")
TEXTBOOKS_DIR = Path("data/01_raw/textbooks")
THEATER_PLAYS_DIR = Path("data/01_raw/theater-plays")
CANTON_LAWS_DIR = Path("data/01_raw/canton-laws")
WIKIPEDIA_DIR = Path("data/01_raw/wikipedia")

WIKIPEDIA_LANGS = ["de", "fr", "it", "en"]
WIKIPEDIA_DUMP = "20231101"          # stable HuggingFace dump date
WIKIPEDIA_SAMPLES_PER_LANG = 100_000 # paragraphs per language
WIKIPEDIA_MIN_PARA_LEN = 50          # skip section headers / stubs

TEXTBOOK_IDIOMS = [
    "rm-sursilv",
    "rm-sutsilv",
    "rm-surmiran",
    "rm-puter",
    "rm-vallader",
]

TEXTBOOK_SPLITS = ["train", "validation", "test", "no_surmiran"]


def download_fmr():
    print("Downloading FMR (2021-2025)...")
    dataset = load_dataset("ZurichNLP/quotidiana", "2021_2025", split="train")
    out = FMR_DIR / "data.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"  Saved {len(dataset)} articles to {out}")


def download_textbooks():
    print("Downloading Textbooks (mediomatix-raw)...")
    for idiom in TEXTBOOK_IDIOMS:
        print(f"  Idiom: {idiom}")
        idiom_dir = TEXTBOOKS_DIR / idiom
        idiom_dir.mkdir(exist_ok=True)
        dataset = load_dataset("ZurichNLP/mediomatix-raw", idiom)
        for split in TEXTBOOK_SPLITS:
            if split not in dataset:
                continue
            out = idiom_dir / f"{split}.jsonl"
            rows = dataset[split]
            with open(out, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(f"    {split}: {len(rows)} rows → {out}")


def download_theater_plays():
    print("Downloading Theater Plays (romansh_theater_plays)...")
    dataset = load_dataset("ZurichNLP/romansh_theater_plays", split="train")
    out = THEATER_PLAYS_DIR / "data.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"  Saved {len(dataset)} rows to {out}")


def download_canton_laws():
    print("Downloading Canton Laws (romansh-canton-laws)...")
    CANTON_LAWS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ZurichNLP/romansh-canton-laws", split="train")
    out = CANTON_LAWS_DIR / "data.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"  Saved {len(dataset)} rows to {out}")


def download_wikipedia():
    print(f"Downloading Wikipedia ({', '.join(WIKIPEDIA_LANGS)}, dump {WIKIPEDIA_DUMP})...")
    for lang in WIKIPEDIA_LANGS:
        lang_dir = WIKIPEDIA_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        out = lang_dir / "data.jsonl"
        config = f"{WIKIPEDIA_DUMP}.{lang}"
        print(f"  {lang}: streaming {config}...")
        dataset = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for article in dataset:
                for para in article.get("text", "").split("\n"):
                    para = para.strip()
                    if len(para) >= WIKIPEDIA_MIN_PARA_LEN:
                        f.write(json.dumps({"text": para, "lang": lang}, ensure_ascii=False) + "\n")
                        count += 1
                        if count >= WIKIPEDIA_SAMPLES_PER_LANG:
                            break
                if count >= WIKIPEDIA_SAMPLES_PER_LANG:
                    break
        print(f"    Saved {count:,} paragraphs → {out}")


if __name__ == "__main__":
    start_run("step0_download_public_data")
    download_fmr()
    download_textbooks()
    download_theater_plays()
    download_canton_laws()
    download_wikipedia()
    print("\nDone. Add the remaining data sources manually (see README.md).")