"""
Step 1z: Validate preprocessed data and generate an HTML report.

Run after step1_preprocess.py, before step2_split_data.py:
    python step1z_validate.py

Report saved to: data/02_preprocessed/validation_report.html
"""

import base64
import io
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.run_log import start_run

# ── constants ──────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)

PREPROCESSED_DIR = Path("data/02_preprocessed")
REPORT_PATH = PREPROCESSED_DIR / "validation_report.html"

SOURCES = ["fmr", "pledari-grond", "rtr-transcripts", "textbooks", "theater-plays", "canton-laws", "proprietary-data"]
IDIOMS = ["rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader", "rm-rumgr"]
DEDUP_SOURCES = ["fmr", "rtr-transcripts", "textbooks", "theater-plays", "canton-laws", "proprietary-data"]

SHORT_THRESHOLD = 10
DEDUP_MIN_LEN = 40
SAMPLE_N = 5
MAX_EXAMPLES = 10
MAX_DEDUP_DISPLAY = 50
MAX_SHORT_DISPLAY = 100

HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{1,6}|#x[0-9a-fA-F]{1,6});")

SUSPICIOUS_CHARS = {
    0x00AD: "Soft Hyphen",
    0x00A0: "Non-Breaking Space",
    0x200B: "Zero-Width Space",
    0x200C: "Zero-Width Non-Joiner",
    0x200D: "Zero-Width Joiner",
    0xFEFF: "BOM / ZWNBSP",
    0x2028: "Line Separator",
    0x2029: "Paragraph Separator",
}

UCAT_NAMES = {
    "L": "Letters",
    "N": "Numbers",
    "Z": "Separators",
    "P": "Punctuation",
    "S": "Symbols",
    "C": "Other / Control",
}

# ── data collection ────────────────────────────────────────────────────────────

def collect():
    counts = {s: Counter() for s in SOURCES}
    lengths = {s: [] for s in SOURCES}
    idiom_lengths = {i: [] for i in IDIOMS}
    entity_hits = defaultdict(list)
    susp_hits = defaultdict(list)
    sample_pools = {s: {i: [] for i in IDIOMS} for s in SOURCES}
    char_counts = {i: Counter() for i in IDIOMS}
    char_examples = {i: defaultdict(list) for i in IDIOMS}
    # hash → list of (source, idiom) across all sources
    hash_locs: dict[int, list] = {}
    hash_snippet: dict[int, str] = {}

    for source in SOURCES:
        print(f"  Scanning {source}...", flush=True)
        for idiom in IDIOMS:
            path = PREPROCESSED_DIR / source / f"{idiom}.tsv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    text = line.rstrip("\n")
                    if not text:
                        continue

                    counts[source][idiom] += 1
                    n = len(text)
                    lengths[source].append(n)
                    idiom_lengths[idiom].append(n)
                    snippet = text[:100]

                    for m in HTML_ENTITY_RE.finditer(text):
                        ent = m.group()
                        if len(entity_hits[ent]) < MAX_EXAMPLES:
                            entity_hits[ent].append((source, idiom, snippet))

                    for c in set(text):
                        cp = ord(c)
                        if cp in SUSPICIOUS_CHARS and len(susp_hits[cp]) < MAX_EXAMPLES:
                            susp_hits[cp].append((source, idiom, snippet))

                    pool = sample_pools[source][idiom]
                    if len(pool) < 50:
                        pool.append(text)

                    c_ctr = Counter(text)
                    char_counts[idiom].update(c_ctr)
                    for c in c_ctr:
                        if len(char_examples[idiom][c]) < MAX_EXAMPLES:
                            char_examples[idiom][c].append((source, text))

                    if source in DEDUP_SOURCES and len(text) >= DEDUP_MIN_LEN:
                        h = hash(text)
                        if h not in hash_locs:
                            hash_locs[h] = []
                            hash_snippet[h] = snippet
                        hash_locs[h].append((source, idiom))

    # Only flag texts that appear in 2+ distinct (source, idiom) pairs
    duplicates = [
        (hash_snippet[h], locs)
        for h, locs in hash_locs.items()
        if len(set(locs)) > 1
    ][:MAX_DEDUP_DISPLAY]

    samples = {
        s: {i: random.sample(pool, min(SAMPLE_N, len(pool)))
            for i, pool in sample_pools[s].items()}
        for s in SOURCES
    }

    return counts, lengths, idiom_lengths, entity_hits, susp_hits, samples, char_counts, char_examples, duplicates

# ── chart helpers ──────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_counts_heatmap(counts) -> str:
    active = [s for s in SOURCES if sum(counts[s].values()) > 0]
    data = [[counts[s].get(i, 0) for i in IDIOMS] for s in active]
    max_val = max(v for row in data for v in row if v > 0) if any(v for row in data for v in row) else 1

    fig, ax = plt.subplots(figsize=(10, max(3, len(active) * 0.7 + 1.5)))
    im = ax.imshow(data, cmap="YlOrRd", norm=mcolors.LogNorm(vmin=1, vmax=max_val), aspect="auto")
    ax.set_xticks(range(len(IDIOMS)))
    ax.set_xticklabels(IDIOMS, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels(active, fontsize=9)
    fig.colorbar(im, ax=ax, label="Count (log scale)", shrink=0.8)
    for i, src in enumerate(active):
        for j, idiom in enumerate(IDIOMS):
            val = counts[src].get(idiom, 0)
            if val > 0:
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        fontsize=7, color="black" if val < max_val * 0.4 else "white")
    ax.set_title("Sample counts per source × idiom (log colour scale)")
    fig.tight_layout()
    return fig_to_b64(fig)


def _bxp_stats(data: list) -> dict | None:
    if not data:
        return None
    s = sorted(data)
    n = len(s)
    q1, q3 = s[n // 4], s[3 * n // 4]
    med = statistics.median(s)
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    non_out = [x for x in s if lo_fence <= x <= hi_fence]
    all_fliers = [x for x in s if x < lo_fence or x > hi_fence]
    return dict(
        med=med, q1=q1, q3=q3,
        whislo=min(non_out) if non_out else q1,
        whishi=max(non_out) if non_out else q3,
        fliers=random.sample(all_fliers, min(300, len(all_fliers))),
        label="",
    )


def chart_lengths(lengths) -> str:
    active = [(s, lengths[s]) for s in SOURCES if lengths[s]]
    labels = [a[0] for a in active]
    bxp_data = [_bxp_stats(a[1]) for a in active]
    bxp_data = [b for b in bxp_data if b is not None]

    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#6d6875"]
    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.7 + 1.5)))
    bp = ax.bxp(bxp_data, vert=False, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Text length (characters)")
    ax.set_title("Text length distribution per source (x clipped at 2 000)")
    ax.set_xlim(0, 2000)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_lengths_by_idiom(idiom_lengths) -> str:
    active = [(i, idiom_lengths[i]) for i in IDIOMS if idiom_lengths[i]]
    labels = [a[0] for a in active]
    bxp_data = [b for b in (_bxp_stats(a[1]) for a in active) if b is not None]

    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#264653"]
    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.7 + 1.5)))
    bp = ax.bxp(bxp_data, vert=False, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Text length (characters)")
    ax.set_title("Text length distribution per idiom — all sources combined (x clipped at 2 000)")
    ax.set_xlim(0, 2000)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── HTML helpers ───────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; max-width: 1500px; margin: 0 auto;
       padding: 24px 32px; color: #222; background: #f2f4f8; line-height: 1.5; }
h1 { color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; margin-bottom: 20px; font-size: 1.6em; }
h2 { color: #1a1a2e; margin: 40px 0 12px; padding-left: 14px; border-left: 4px solid #457b9d; font-size: 1.2em; }
h3 { color: #457b9d; margin: 16px 0 8px; font-size: 1em; }
.card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px 24px; margin-bottom: 12px; }
nav ol { margin: 8px 0 0; padding-left: 22px; line-height: 2; }
nav a { color: #457b9d; text-decoration: none; }
nav a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 10px 0; }
th { background: #1a1a2e; color: white; padding: 8px 14px; text-align: left; white-space: nowrap; }
td { padding: 5px 14px; border-bottom: 1px solid #eee; vertical-align: top; }
tr:hover td { background: #f0f4ff; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.grand { font-weight: bold; background: #eef2ff; }
details { margin: 3px 0; }
summary { cursor: pointer; padding: 4px 10px; background: #f0f0f0; border-radius: 4px;
          list-style: none; font-size: 0.9em; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 0.75em; color: #666; }
details[open] > summary::before { content: "▼ "; }
summary:hover { background: #e0e0e0; }
mark { background: #fff176; padding: 0 2px; border-radius: 2px; font-weight: bold; }
.ok   { color: #27ae60; font-weight: bold; }
.warn { color: #e67e22; font-weight: bold; }
.bad  { color: #c0392b; font-weight: bold; }
.mono { font-family: monospace; background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
.char-cell { font-family: monospace; font-size: 1.3em; text-align: center; min-width: 2em; }
.cp   { font-family: monospace; font-size: 0.8em; color: #555; white-space: nowrap; }
.snip { color: #555; font-size: 0.85em; }
img   { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin-top: 12px; }
ol.examples { margin: 6px 0 4px 18px; font-size: 0.85em; }
ol.examples li { margin-bottom: 4px; word-break: break-word; }
"""

def img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{escape(alt)}">'


def excerpt_around(text: str, c: str, window: int = 70) -> str:
    idx = text.find(c)
    if idx < 0:
        return text[:140]
    start = max(0, idx - window)
    end = min(len(text), idx + window + 1)
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


def highlight(text: str, c: str) -> str:
    ec = escape(c)
    return escape(text).replace(ec, f"<mark>{ec}</mark>", 1)

# ── sections ───────────────────────────────────────────────────────────────────

def section_counts(counts) -> str:
    active = [s for s in SOURCES if sum(counts[s].values()) > 0]
    header = "".join(f"<th>{escape(i)}</th>" for i in IDIOMS)
    rows = []
    col_totals = Counter()
    grand = 0
    for s in active:
        row_total = sum(counts[s].values())
        grand += row_total
        cells = ""
        for i in IDIOMS:
            v = counts[s].get(i, 0)
            col_totals[i] += v
            cells += f"<td class='num'>{v:,}</td>"
        rows.append(f"<tr><td><b>{escape(s)}</b></td>{cells}<td class='num'><b>{row_total:,}</b></td></tr>")
    total_cells = "".join(f"<td class='num grand'>{col_totals[i]:,}</td>" for i in IDIOMS)
    rows.append(f"<tr class='grand'><td><b>Total</b></td>{total_cells}<td class='num grand'><b>{grand:,}</b></td></tr>")
    table = f"<table><thead><tr><th>Source</th>{header}<th>Total</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    chart = img_tag(chart_counts_heatmap(counts), "counts heatmap")
    return f"<h2 id='s-counts'>1. Sample Counts</h2><div class='card'>{table}{chart}</div>"


def _len_stats_row(label: str, data: list) -> str:
    sd = sorted(data)
    n = len(sd)
    p75 = sd[min(int(n * 0.75), n - 1)]
    p90 = sd[min(int(n * 0.90), n - 1)]
    return (
        f"<tr><td>{escape(label)}</td>"
        f"<td class='num'>{n:,}</td>"
        f"<td class='num'>{sd[0]:,}</td>"
        f"<td class='num'>{round(statistics.median(sd)):,}</td>"
        f"<td class='num'>{p75:,}</td>"
        f"<td class='num'>{p90:,}</td>"
        f"<td class='num'>{round(statistics.mean(sd)):,}</td>"
        f"<td class='num'>{sd[-1]:,}</td></tr>"
    )

_LEN_TABLE_HEADER = (
    "<table><thead><tr><th>Source / Idiom</th><th>Count</th>"
    "<th>Min</th><th>Median</th><th>p75</th><th>p90</th><th>Mean</th><th>Max</th>"
    "</tr></thead><tbody>"
)
_BOXPLOT_DESC = (
    "<p style='margin-bottom:8px'>"
    "How to read: the vertical line is the <b>median</b>, the box spans Q1–Q3 "
    "(interquartile range), whiskers extend to 1.5× IQR beyond the box, and dots "
    "are outliers. The x-axis is clipped at 2,000 characters; longer texts exist "
    "but are not shown to keep the scale readable."
    "</p>"
)


def section_lengths(lengths, idiom_lengths) -> str:
    src_rows = "".join(
        _len_stats_row(s, lengths[s]) for s in SOURCES if lengths[s]
    )
    src_table = _LEN_TABLE_HEADER + src_rows + "</tbody></table>"

    idiom_rows = "".join(
        _len_stats_row(i, idiom_lengths[i]) for i in IDIOMS if idiom_lengths[i]
    )
    idiom_table = _LEN_TABLE_HEADER + idiom_rows + "</tbody></table>"

    chart_src = img_tag(chart_lengths(lengths), "length distribution per source")
    chart_idiom = img_tag(chart_lengths_by_idiom(idiom_lengths), "length distribution per idiom")

    idiom_desc = (
        "<p style='margin-bottom:8px'>"
        "Aggregates all sources for each idiom. Similar distributions across idioms "
        "indicate balanced coverage. A noticeably lower median for one idiom usually "
        "means a short-entry source (e.g. a dictionary) dominates that idiom's data."
        "</p>"
    )

    return (
        f"<h2 id='s-lengths'>2. Text Length Distribution</h2>"
        f"<div class='card'>"
        f"<h3>Per source</h3>"
        f"{_BOXPLOT_DESC}{src_table}{chart_src}"
        f"<h3 style='margin-top:28px'>Per idiom — all sources combined</h3>"
        f"{idiom_desc}{idiom_table}{chart_idiom}"
        f"</div>"
    )


def section_duplicates(duplicates) -> str:
    if not duplicates:
        body = "<p class='ok'>✓ No cross-source duplicates found.</p>"
    else:
        note = (f"<p class='warn'>⚠ {len(duplicates)} duplicate text(s) found across sources "
                f"(texts ≥ {DEDUP_MIN_LEN} chars; Pledari Grond excluded). "
                f"Up to {MAX_DEDUP_DISPLAY} shown.</p>")
        rows = []
        for snip, locs in duplicates:
            loc_str = " · ".join(f"<span class='mono'>{escape(s)}/{escape(i)}</span>" for s, i in locs)
            rows.append(f"<tr><td>{loc_str}</td><td class='snip'>{escape(snip)}</td></tr>")
        table = (
            "<table><thead><tr><th>Locations</th><th>Text snippet</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        body = note + table
    return f"<h2 id='s-dupes'>3. Cross-Source Duplicates</h2><div class='card'>{body}</div>"


def section_artifacts(entity_hits, susp_hits) -> str:
    parts = []

    if entity_hits:
        rows = []
        for ent in sorted(entity_hits):
            hits = entity_hits[ent]
            items = "".join(
                f"<li>{escape(s)}/{escape(i)}: <span class='snip'>{escape(snip)}</span></li>"
                for s, i, snip in hits
            )
            rows.append(
                f"<tr><td><span class='mono'>{escape(ent)}</span></td>"
                f"<td class='num'>{len(hits)}</td>"
                f"<td><details><summary>{len(hits)} example(s)</summary>"
                f"<ol class='examples'>{items}</ol></details></td></tr>"
            )
        parts.append(
            "<h3>HTML Entities</h3>"
            "<table><thead><tr><th>Entity</th><th>Occurrences</th><th>Examples</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        parts.append("<p class='ok'>✓ No HTML entities found.</p>")

    if susp_hits:
        rows = []
        for cp in sorted(susp_hits):
            hits = susp_hits[cp]
            items = "".join(
                f"<li>{escape(s)}/{escape(i)}: <span class='snip'>{escape(snip)}</span></li>"
                for s, i, snip in hits
            )
            rows.append(
                f"<tr><td class='mono'>U+{cp:04X}</td>"
                f"<td>{escape(SUSPICIOUS_CHARS[cp])}</td>"
                f"<td class='num'>{len(hits)}</td>"
                f"<td><details><summary>{len(hits)} example(s)</summary>"
                f"<ol class='examples'>{items}</ol></details></td></tr>"
            )
        parts.append(
            "<h3>Suspicious Unicode Characters</h3>"
            "<table><thead><tr><th>Codepoint</th><th>Name</th><th>Occurrences</th><th>Examples</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        parts.append("<p class='ok'>✓ No suspicious Unicode characters found.</p>")

    return f"<h2 id='s-artifacts'>4. Encoding Artifacts</h2><div class='card'>{''.join(parts)}</div>"


def section_short(short_texts) -> str:
    if not short_texts:
        body = f"<p class='ok'>✓ No texts shorter than {SHORT_THRESHOLD} characters found.</p>"
    else:
        note = (
            f"<p>Texts with fewer than {SHORT_THRESHOLD} characters "
            f"({len(short_texts)} shown). Short entries from <b>pledari-grond</b> "
            f"(dictionary headwords) are expected.</p>"
        )
        rows = "".join(
            f"<tr><td>{escape(s)}</td><td>{escape(i)}</td>"
            f"<td class='num'>{len(t)}</td>"
            f"<td><span class='mono'>{escape(t)}</span></td></tr>"
            for s, i, t in short_texts
        )
        table = (
            "<table><thead><tr><th>Source</th><th>Idiom</th>"
            f"<th>Len</th><th>Text</th></tr></thead><tbody>{rows}</tbody></table>"
        )
        body = note + table
    return f"<h2 id='s-short'>5. Very Short Texts</h2><div class='card'>{body}</div>"


def section_samples(samples) -> str:  # noqa: section 5
    blocks = []
    for source in SOURCES:
        idiom_blocks = []
        for idiom in IDIOMS:
            pool = samples[source].get(idiom, [])
            if not pool:
                continue
            items = "".join(f"<li>{escape(t)}</li>" for t in pool)
            idiom_blocks.append(
                f"<details><summary>{escape(idiom)}</summary>"
                f"<ol class='examples'>{items}</ol></details>"
            )
        if idiom_blocks:
            blocks.append(
                f"<details><summary><b>{escape(source)}</b></summary>"
                f"<div style='margin:4px 0 4px 16px'>{''.join(idiom_blocks)}</div></details>"
            )
    note = f"<p style='margin-bottom:10px'>{SAMPLE_N} random texts per source/idiom (sampled from the first 50 texts per file).</p>"
    return f"<h2 id='s-samples'>5. Random Samples</h2><div class='card'>{note}{''.join(blocks)}</div>"


def section_char_inventory(char_counts, char_examples) -> str:
    idiom_blocks = []
    for idiom in IDIOMS:
        ctr = char_counts[idiom]
        if not ctr:
            continue

        by_group = defaultdict(list)
        for c in ctr:
            if c == "\n":
                continue
            by_group[unicodedata.category(c)[0]].append(c)

        group_blocks = []
        for gkey in ["L", "N", "Z", "P", "S", "C"]:
            chars = sorted(by_group.get(gkey, []), key=ord)
            if not chars:
                continue
            rows = []
            for c in chars:
                cp = ord(c)
                uname = unicodedata.name(c, f"U+{cp:04X}")
                ucat = unicodedata.category(c)
                count = ctr[c]
                examples = char_examples[idiom].get(c, [])
                ex_items = "".join(
                    f"<li><span class='mono' style='font-size:0.8em;color:#888'>{escape(src)}</span>"
                    f" — {highlight(t, c)}</li>"
                    for src, t in examples
                )
                ex_block = (
                    f"<details><summary>{len(examples)} example(s)</summary>"
                    f"<ol class='examples'>{ex_items}</ol></details>"
                ) if ex_items else ""
                if c.strip() or cp == 0x20:
                    disp = f"<td class='char-cell'>{escape(c)}</td>"
                else:
                    disp = f"<td class='char-cell' title='U+{cp:04X}' style='color:#aaa'>␣</td>"
                rows.append(
                    f"<tr>{disp}<td class='cp'>U+{cp:04X}</td>"
                    f"<td class='cp'>{escape(uname)}</td>"
                    f"<td class='cp'>{escape(ucat)}</td>"
                    f"<td class='num'>{count:,}</td>"
                    f"<td>{ex_block}</td></tr>"
                )
            gname = UCAT_NAMES.get(gkey, gkey)
            table = (
                "<table style='font-size:0.85em'>"
                "<thead><tr><th>Char</th><th>Codepoint</th><th>Unicode Name</th>"
                "<th>Cat</th><th>Count</th><th>Examples</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )
            group_blocks.append(
                f"<details><summary>{escape(gname)} — {len(chars)} unique</summary>{table}</details>"
            )

        idiom_blocks.append(
            f"<details id='ci-{escape(idiom)}'>"
            f"<summary><b>{escape(idiom)}</b> — {len(ctr)} unique characters</summary>"
            f"<div style='margin:4px 0 4px 16px'>{''.join(group_blocks)}</div>"
            f"</details>"
        )

    note = ("<p style='margin-bottom:10px'>Every character found across all sources for each idiom, "
            "grouped by Unicode category. Click a character's examples to see it in context.</p>")
    return (
        f"<h2 id='s-chars'>6. Character Inventory per Idiom</h2>"
        f"<div class='card'>{note}{''.join(idiom_blocks)}</div>"
    )

# ── assembly ───────────────────────────────────────────────────────────────────

def build_html(counts, lengths, idiom_lengths, entity_hits, susp_hits, samples, char_counts, char_examples, duplicates) -> str:
    toc = """<div class='card' style='display:inline-block;min-width:260px;margin-bottom:24px'>
<b>Contents</b>
<ol style='margin:8px 0 0;padding-left:22px;line-height:2'>
<li><a href='#s-counts'>Sample Counts</a></li>
<li><a href='#s-lengths'>Text Length Distribution</a></li>
<li><a href='#s-dupes'>Cross-Source Duplicates</a></li>
<li><a href='#s-artifacts'>Encoding Artifacts</a></li>
<li><a href='#s-samples'>Random Samples</a></li>
<li><a href='#s-chars'>Character Inventory per Idiom</a></li>
</ol></div>"""

    body = "".join([
        section_counts(counts),
        section_lengths(lengths, idiom_lengths),
        section_duplicates(duplicates),
        section_artifacts(entity_hits, susp_hits),
        section_samples(samples),
        section_char_inventory(char_counts, char_examples),
    ])

    return (
        f"<!DOCTYPE html><html lang='en'><head>"
        f"<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Preprocessed Data — Validation Report</title>"
        f"<style>{CSS}</style></head>"
        f"<body><h1>Preprocessed Data — Validation Report</h1>{toc}{body}</body></html>"
    )

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    start_run("step1z_validate", artifacts=[REPORT_PATH])

    print("Collecting data...")
    counts, lengths, idiom_lengths, entity_hits, susp_hits, samples, char_counts, char_examples, duplicates = collect()

    print("Building report...")
    html = build_html(counts, lengths, idiom_lengths, entity_hits, susp_hits, samples, char_counts, char_examples, duplicates)

    REPORT_PATH.write_text(html, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size // 1024
    print(f"\nReport saved to {REPORT_PATH} ({size_kb} KB)")
    print(f"Open: file://{REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()