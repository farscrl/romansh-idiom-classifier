"""
Step 1: Load all data sources, clean and normalize text, write to data/02_preprocessed/.

One subfolder per source, one file per idiom within each subfolder.
Each file contains one text per line (no label — idiom is implicit in the filename).

Output:
  data/02_preprocessed/fmr/rm-{idiom}.tsv
  data/02_preprocessed/pledari-grond/rm-{idiom}.tsv
  data/02_preprocessed/rtr-transcripts/rm-{idiom}.tsv
  data/02_preprocessed/textbooks/rm-{idiom}.tsv
  data/02_preprocessed/theater-plays/rm-{idiom}.tsv
  data/02_preprocessed/canton-laws/rm-{idiom}.tsv
  data/02_preprocessed/proprietary-data/rm-{idiom}.tsv
"""

import re
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from src.run_log import start_run
from src.preprocessing import (
    load_fmr,
    load_pledari_grond,
    load_rtr_transcripts,
    load_textbooks,
    load_theater_plays,
    load_canton_laws,
    load_proprietary,
)

PREPROCESSED_DIR = Path("data/02_preprocessed")

# ── umlaut filter ──────────────────────────────────────────────────────────────

# Per-idiom: characters whose presence in a sentence triggers removal.
# ä/ö/ü do not exist natively in sursilvan/sutsilvan/surmiran/rumgr.
# ö/ü ARE native to puter and vallader (Engadinese); only ä is foreign there.
FOREIGN_CHARS: dict[str, set[str]] = {
    "rm-sursilv":  set("äöüÄÖÜ"),
    "rm-sutsilv":  set("äöüÄÖÜ"),
    "rm-surmiran": set("äöüÄÖÜ"),
    "rm-rumgr":    set("äöüÄÖÜ"),
    "rm-puter":    set("äÄ"),
    "rm-vallader": set("äÄ"),
}

ALLOWLIST_PATH = Path("data/01_raw/umlaut_allowlist.txt")
TRIGGER_LOG_PATH = Path("data/02_preprocessed/umlaut_triggers.tsv")
UMLAUT_REPORT_PATH = Path("data/02_preprocessed/umlaut_report.html")

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# ── cross-source dedup ─────────────────────────────────────────────────────────

# Sources in descending priority order for cross-source deduplication.
# When the same text appears in multiple sources with the same idiom, the
# higher-priority source keeps it; the others lose it.
# If the same text appears with *different* idioms across sources, it is
# removed from all sources (contradictory label — cannot be trusted).
DEDUP_PRIORITY = [
    "proprietary-data",
    "fmr",
    "rtr-transcripts",
    "textbooks",
    "theater-plays",
    "canton-laws",
    "pledari-grond",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def save_by_idiom(samples: list[tuple[str, str]], source_dir: Path):
    source_dir.mkdir(parents=True, exist_ok=True)
    by_idiom: dict[str, list[str]] = defaultdict(list)
    for label, text in samples:
        by_idiom[label].append(text)
    total = 0
    for idiom in sorted(by_idiom):
        path = source_dir / f"{idiom}.tsv"
        unique = list(dict.fromkeys(by_idiom[idiom]))   # deduplicate, preserve order
        n_dupes = len(by_idiom[idiom]) - len(unique)
        with open(path, "w", encoding="utf-8") as f:
            for text in unique:
                f.write(text + "\n")
        total += len(unique)
        dupe_note = f"  ({n_dupes} duplicates removed)" if n_dupes else ""
        print(f"    {idiom}: {len(unique):>7,} samples → {path}{dupe_note}")
    print(f"  Total: {total:,}")


def _load_allowlist() -> list[str]:
    if not ALLOWLIST_PATH.exists():
        return []
    terms = []
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    return terms


def apply_allowlist(text: str, allowlist: list[str]) -> str:
    for term in allowlist:
        text = text.replace(term, "$NE$")
    return text


def _render_diff_block(sentences: list[str], dropped_idx: set[int], fully_dropped: bool) -> str:
    if fully_dropped:
        inner = " ".join(f"<span class='dropped'>{escape(s)}</span>" for s in sentences)
        return f"<div class='diff fully-dropped'><span class='label'>fully dropped</span> {inner}</div>"
    parts = []
    for i, s in enumerate(sentences):
        if i in dropped_idx:
            parts.append(f"<span class='dropped'>{escape(s)}</span>")
        else:
            parts.append(f"<span class='kept'>{escape(s)}</span>")
    return "<div class='diff'>" + " ".join(parts) + "</div>"


def _section_anchor(key: str) -> str:
    return key.replace("/", "-").replace(" ", "")


def _build_report_html(report_sections: dict[str, list[str]]) -> str:
    css = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto;
       padding: 24px 32px; background: #f2f4f8; color: #222; line-height: 1.55; }
h1 { font-size: 1.5em; border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; margin-bottom: 20px; }
h2 { font-size: 1.05em; color: #1a1a2e; margin: 32px 0 8px;
     border-left: 4px solid #457b9d; padding-left: 12px; }
.count { font-weight: normal; color: #666; font-size: 0.9em; }
nav ol { margin: 8px 0 24px; padding-left: 22px; line-height: 2; }
nav a { color: #457b9d; text-decoration: none; }
nav a:hover { text-decoration: underline; }
.diff { background: white; border: 1px solid #ddd; border-radius: 6px;
        padding: 10px 14px; margin-bottom: 5px; font-size: 0.88em; word-break: break-word; }
.diff.fully-dropped { border-color: #e74c3c; background: #fff5f5; }
.label { font-size: 0.78em; font-weight: bold; color: #c0392b;
         background: #fdd; padding: 1px 6px; border-radius: 3px; margin-right: 6px; }
.kept { color: #222; }
.dropped { background: #ffdddd; color: #c0392b; text-decoration: line-through;
           padding: 1px 3px; border-radius: 2px; margin: 0 2px; }
"""
    toc_items = "".join(
        "<li><a href='#s-" + escape(_section_anchor(k)) + "'>" + escape(k) + "</a></li>"
        for k in report_sections
    )
    toc = "<nav><ol>" + toc_items + "</ol></nav>"

    sections_html = ""
    for key, blocks in report_sections.items():
        anchor = _section_anchor(key)
        sections_html += (
            "<h2 id='s-" + escape(anchor) + "'>" + escape(key)
            + " <span class='count'>(" + str(len(blocks)) + " texts modified)</span></h2>"
            + "".join(blocks)
        )

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>Umlaut filter report</title><style>" + css + "</style></head>"
        "<body><h1>Umlaut filter report</h1>" + toc + sections_html + "</body></html>"
    )


# ── pipeline steps ─────────────────────────────────────────────────────────────

def filter_umlaut_sentences():
    """
    For each idiom, split texts into sentences and remove any sentence that
    contains a character foreign to that idiom (defined in FOREIGN_CHARS).
    Applies the allowlist first so known valid terms survive as $NE$.

    Writes:
      umlaut_report.html  — diff view of every change for manual review
      umlaut_triggers.tsv — frequency-sorted words that triggered removals
    """
    print("\nFiltering umlaut sentences...")
    allowlist = _load_allowlist()
    if allowlist:
        print(f"  Allowlist: {len(allowlist)} terms")

    trigger_counts: Counter = Counter()
    report_sections: dict[str, list[str]] = {}
    total_cleaned = 0
    total_dropped = 0

    for source_dir in sorted(list(PREPROCESSED_DIR.iterdir())):
        if not source_dir.is_dir():
            continue
        for path in sorted(list(source_dir.glob("rm-*.tsv"))):
            idiom = path.stem
            foreign = FOREIGN_CHARS.get(idiom)
            if foreign is None:
                continue

            original: list[str] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    t = line.rstrip("\n")
                    if t:
                        original.append(t)

            filtered: list[str] = []
            changed = False
            diff_blocks: list[str] = []

            for text in original:
                cleaned = apply_allowlist(text, allowlist)

                if not any(c in foreign for c in cleaned):
                    filtered.append(text)
                    continue

                sentences = _SENT_SPLIT_RE.split(cleaned)
                kept_idx: set[int] = set()
                dropped_idx: set[int] = set()
                for i, sent in enumerate(sentences):
                    if any(c in foreign for c in sent):
                        dropped_idx.add(i)
                    else:
                        kept_idx.add(i)

                for i in dropped_idx:
                    for word in sentences[i].split():
                        if any(c in foreign for c in word):
                            trigger_counts[word] += 1

                kept_sentences = [sentences[i] for i in sorted(kept_idx)]
                result = " ".join(kept_sentences).strip()

                if result:
                    filtered.append(result)
                    total_cleaned += 1
                    diff_blocks.append(_render_diff_block(sentences, dropped_idx, False))
                else:
                    total_dropped += 1
                    diff_blocks.append(_render_diff_block(sentences, dropped_idx, True))
                changed = True

            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    for t in filtered:
                        f.write(t + "\n")
                n_removed = len(original) - len(filtered)
                section_key = f"{source_dir.name} / {idiom}"
                report_sections[section_key] = diff_blocks
                print(f"    {source_dir.name}/{idiom}: "
                      f"{len(diff_blocks) - n_removed} partially cleaned, "
                      f"{n_removed} fully dropped")

    html = _build_report_html(report_sections)
    UMLAUT_REPORT_PATH.write_text(html, encoding="utf-8")
    size_kb = UMLAUT_REPORT_PATH.stat().st_size // 1024
    print(f"  Report → {UMLAUT_REPORT_PATH} ({size_kb} KB)")

    with open(TRIGGER_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("count\tword\n")
        for word, count in trigger_counts.most_common():
            f.write(f"{count}\t{word}\n")
    print(f"  Trigger log → {TRIGGER_LOG_PATH} ({len(trigger_counts):,} unique triggering words)")
    print(f"  Total: {total_cleaned:,} texts partially cleaned, {total_dropped:,} texts fully dropped")


def cross_source_dedup():
    """
    Remove texts that appear in more than one source.

    - Same text, same idiom in multiple sources → keep only in the
      highest-priority source (DEDUP_PRIORITY), remove from the rest.
    - Same text, different idioms across sources → remove from all sources
      (contradictory label).

    Only rewrites files that actually changed.
    """
    print("\nCross-source deduplication...")

    text_locs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source in DEDUP_PRIORITY:
        source_dir = PREPROCESSED_DIR / source
        if not source_dir.exists():
            continue
        for path in sorted(list(source_dir.glob("rm-*.tsv"))):
            idiom = path.stem
            with open(path, encoding="utf-8") as f:
                for line in f:
                    text = line.rstrip("\n")
                    if text:
                        text_locs[text].add((source, idiom))

    keep_in: dict[str, str | None] = {}
    contradictory = 0
    redundant = 0
    for text, locs in text_locs.items():
        if len(locs) == 1:
            continue
        idioms = {idiom for _, idiom in locs}
        if len(idioms) > 1:
            keep_in[text] = None
            contradictory += 1
        else:
            sources_present = {src for src, _ in locs}
            for src in DEDUP_PRIORITY:
                if src in sources_present:
                    keep_in[text] = src
                    break
            redundant += 1

    print(f"  {contradictory:,} texts with contradictory idiom labels → removed from all sources")
    print(f"  {redundant:,} texts duplicated across sources (same idiom) → kept in highest-priority source")

    total_removed = 0
    for source in DEDUP_PRIORITY:
        source_dir = PREPROCESSED_DIR / source
        if not source_dir.exists():
            continue
        for path in sorted(list(source_dir.glob("rm-*.tsv"))):
            idiom = path.stem
            original: list[str] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    text = line.rstrip("\n")
                    if text:
                        original.append(text)

            filtered = [
                t for t in original
                if t not in keep_in or keep_in[t] == source
            ]
            removed = len(original) - len(filtered)
            if removed:
                with open(path, "w", encoding="utf-8") as f:
                    for t in filtered:
                        f.write(t + "\n")
                print(f"    {source}/{idiom}: removed {removed:,} duplicates")
                total_removed += removed

    print(f"  Total removed: {total_removed:,}")


def main():
    start_run("step1_preprocess", artifacts=[UMLAUT_REPORT_PATH, TRIGGER_LOG_PATH])

    print("\nPreprocessing FMR...")
    save_by_idiom(load_fmr(), PREPROCESSED_DIR / "fmr")

    print("\nPreprocessing Pledari Grond...")
    save_by_idiom(load_pledari_grond(), PREPROCESSED_DIR / "pledari-grond")

    print("\nPreprocessing RTR Transcripts...")
    save_by_idiom(load_rtr_transcripts(), PREPROCESSED_DIR / "rtr-transcripts")

    print("\nPreprocessing Textbooks...")
    save_by_idiom(load_textbooks(), PREPROCESSED_DIR / "textbooks")

    print("\nPreprocessing Theater Plays...")
    save_by_idiom(load_theater_plays(), PREPROCESSED_DIR / "theater-plays")

    print("\nPreprocessing Canton Laws...")
    save_by_idiom(load_canton_laws(), PREPROCESSED_DIR / "canton-laws")

    print("\nPreprocessing Proprietary data...")
    save_by_idiom(load_proprietary(), PREPROCESSED_DIR / "proprietary-data")

    filter_umlaut_sentences()
    cross_source_dedup()

    print("\nDone. Inspect data/02_preprocessed/ before running step2_split_data.py.")


if __name__ == "__main__":
    main()