#!/usr/bin/env python3
"""
Compute data statistics tables from preprocessed files and write them to JSON.

Table A: per-source, per-idiom sample and token counts
Table B: per-idiom totals across training sources (pivot of Table A)
Table C: per-split, per-idiom sample and token counts
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED = REPO_ROOT / "data" / "02_preprocessed"
SPLITS = REPO_ROOT / "data" / "03_splits"
OUTPUT = REPO_ROOT / "data" / "04_evaluation" / "data_stats.json"

SOURCES = {
    "fmr":            ("FMR (La Quotidiana)",          False),
    "rtr-transcripts": ("RTR Transcripts",              False),
    "pledari-grond":  ("PG / dicziunaris ladins",       False),
    "textbooks":      ("Mediomatix Textbooks",           False),
    "theater-plays":  ("Theater Plays",                  False),
    "canton-laws":    ("Canton Laws",                    False),
    "proprietary-data": ("RTR Telesguard Notes",         True),
}

IDIOMS = {
    "rm-sursilv":  "Sursilvan",
    "rm-sutsilv":  "Sutsilvan",
    "rm-surmiran": "Surmiran",
    "rm-puter":    "Puter",
    "rm-vallader": "Vallader",
    "rm-rumgr":    "Rumantsch Grischun",
}

SPLITS_ORDER = {
    "train":   "Train",
    "dev":     "Dev",
    "test_a":  "Test A",
    "test_b":  "Test B",
    "test_c":  "Test C",
    "test_d":  "Test D",
}


def count_file(path: Path) -> tuple[int, int]:
    """Return (samples, tokens) for a one-column text file."""
    samples = tokens = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                samples += 1
                tokens += len(line.split())
    return samples, tokens


def avg(tokens: int, samples: int) -> float:
    return round(tokens / samples, 1) if samples else 0.0


def build_table_a() -> list[dict]:
    rows = []
    for source_key, (source_name, evaluation_only) in SOURCES.items():
        source_dir = PREPROCESSED / source_key
        if not source_dir.is_dir():
            print(f"  warning: {source_dir} not found, skipping", file=sys.stderr)
            continue
        idiom_data: dict[str, dict] = {}
        for tsv in sorted(source_dir.glob("rm-*.tsv")):
            idiom_key = tsv.stem
            if idiom_key not in IDIOMS:
                continue
            s, t = count_file(tsv)
            idiom_data[idiom_key] = {
                "name": IDIOMS[idiom_key],
                "samples": s,
                "tokens": t,
                "avg_tokens": avg(t, s),
            }
        total_s = sum(v["samples"] for v in idiom_data.values())
        total_t = sum(v["tokens"] for v in idiom_data.values())
        rows.append({
            "key": source_key,
            "name": source_name,
            "evaluation_only": evaluation_only,
            "idioms": idiom_data,
            "total": {
                "samples": total_s,
                "tokens": total_t,
                "avg_tokens": avg(total_t, total_s),
            },
        })
    return rows


def build_table_b(table_a: list[dict]) -> dict:
    training_sources = [r for r in table_a if not r["evaluation_only"]]
    by_idiom: dict[str, dict] = {}
    for idiom_key, idiom_name in IDIOMS.items():
        by_source: dict[str, int] = {}
        total_s = total_t = 0
        for row in training_sources:
            entry = row["idioms"].get(idiom_key)
            s = entry["samples"] if entry else 0
            t = entry["tokens"] if entry else 0
            by_source[row["key"]] = s
            total_s += s
            total_t += t
        by_idiom[idiom_key] = {
            "name": idiom_name,
            "by_source": by_source,
            "total_samples": total_s,
            "total_tokens": total_t,
            "avg_tokens": avg(total_t, total_s),
        }
    grand_s = sum(v["total_samples"] for v in by_idiom.values())
    grand_t = sum(v["total_tokens"] for v in by_idiom.values())
    return {
        "idioms": by_idiom,
        "total": {
            "by_source": {
                src["key"]: sum(
                    src["idioms"].get(ik, {}).get("samples", 0)
                    for ik in IDIOMS
                )
                for src in training_sources
            },
            "total_samples": grand_s,
            "total_tokens": grand_t,
            "avg_tokens": avg(grand_t, grand_s),
        },
    }


def build_table_c() -> list[dict]:
    rows = []
    for split_stem, split_name in SPLITS_ORDER.items():
        tsv_path = None
        for subdir in SPLITS.iterdir():
            candidate = subdir / f"{split_stem}.tsv"
            if candidate.exists():
                tsv_path = candidate
                break
        if tsv_path is None:
            print(f"  warning: no file for split '{split_stem}', skipping", file=sys.stderr)
            continue
        idiom_data: dict[str, dict] = {k: {"name": v, "samples": 0, "tokens": 0} for k, v in IDIOMS.items()}
        with tsv_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                idiom_key, text = parts
                if idiom_key not in idiom_data:
                    continue
                idiom_data[idiom_key]["samples"] += 1
                idiom_data[idiom_key]["tokens"] += len(text.split())
        for entry in idiom_data.values():
            entry["avg_tokens"] = avg(entry["tokens"], entry["samples"])
        total_s = sum(v["samples"] for v in idiom_data.values())
        total_t = sum(v["tokens"] for v in idiom_data.values())
        rows.append({
            "key": split_stem,
            "name": split_name,
            "idioms": idiom_data,
            "total": {
                "samples": total_s,
                "tokens": total_t,
                "avg_tokens": avg(total_t, total_s),
            },
        })
    return rows


def main() -> None:
    print("Computing Table A (per source)...")
    table_a = build_table_a()
    print("Computing Table B (per idiom)...")
    table_b = build_table_b(table_a)
    print("Computing Table C (per split)...")
    table_c = build_table_c()

    result = {
        "table_a_sources": table_a,
        "table_b_by_idiom": table_b,
        "table_c_splits": table_c,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
