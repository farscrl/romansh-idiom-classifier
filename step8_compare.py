"""
Step 8: Compare all models (trained + external) across all test sets.

Discovers adapter scripts in compare/adapters/[0-9]*.py, runs each via subprocess,
reads whatever JSON result files exist in data/05_comparison/, and generates a
comparison HTML report. The report is driven solely by the JSON files present —
adapters that fail simply produce no column.

usage: python step8_compare.py [--report-only]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from src.run_log import start_run
from src.splits import load_splits_meta

ADAPTERS_DIR = Path("compare/adapters")
COMPARE_DIR  = Path("data/05_comparison")
REPORT_PATH  = COMPARE_DIR / "report.html"

LABEL_SHORT = {
    "rm-sursilv": "Sursilv",
    "rm-sutsilv": "Sutsilv",
    "rm-surmiran": "Surmiran",
    "rm-puter":    "Puter",
    "rm-vallader": "Vallader",
    "rm-rumgr":    "RumGr",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "en": "English",
}

TEST_SET_LABELS = {
    "test_a": "Test A — news (FMR)",
    "test_b": "Test B — speech transcripts (RTR)",
    "test_c": "Test C — schoolbooks (Textbooks)",
    "test_d": "Test D — proprietary (out-of-domain)",
    "test_e": "Test E — Wikipedia (de/fr/it/en)",
}

CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: .4rem; }
h2 { margin-top: 2.5rem; color: #1a4f8a; }
h3 { margin-top: 1.5rem; color: #444; }
table.summary { border-collapse: collapse; width: 100%; margin: 1rem 0; }
table.summary th, table.summary td { padding: .45rem .9rem; border: 1px solid #ddd; text-align: right; white-space: nowrap; }
table.summary th { background: #e8eef6; text-align: center; }
table.summary td:first-child, table.summary td:nth-child(2) { text-align: left; }
table.summary th:first-child, table.summary th:nth-child(2) { text-align: left; }
table.summary tr.group-sep td { border-top: 2px solid #aac; }
table.summary td.best  { background: #d4edda; font-weight: 600; }
table.summary td.worst { background: #f8d7da; }
table.summary td.na    { color: #aaa; text-align: center; }
table.compare { border-collapse: collapse; width: 100%; margin: .8rem 0; }
table.compare th, table.compare td { padding: .4rem .8rem; border: 1px solid #ddd; text-align: right; }
table.compare th { background: #e8eef6; }
table.compare td:first-child, table.compare th:first-child { text-align: left; }
.cm-grid { display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 1rem 0; }
.cm-grid figure { margin: 0; text-align: center; }
.cm-grid img { max-width: 420px; border: 1px solid #ddd; border-radius: 4px; display: block; }
.cm-grid figcaption { margin-top: .4rem; font-size: .85rem; color: #555; }
footer { margin-top: 3rem; color: #aaa; font-size: .8rem; }
"""


def run_adapters(adapter_paths: list[Path]) -> None:
    repo_root = str(Path(__file__).resolve().parent)
    for adapter in adapter_paths:
        print(f"\n{'─' * 60}")
        print(f"Running: {adapter.name}")
        t0 = time.time()
        result = subprocess.run([sys.executable, str(adapter)], cwd=repo_root)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        else:
            print(f"  done ({elapsed:.1f}s)")


def load_results(json_paths: list[Path]) -> dict:
    """Load JSON files. Returns {stem: parsed_data} in filename order."""
    loaded = {}
    for path in json_paths:
        try:
            loaded[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Warning: could not load {path.name}: {e}")
    return loaded


def build_summary_table(loaded: dict, adapter_order: list[str], tkeys: list[str]) -> str:
    col_headers = "".join(f"<th>{escape(loaded[s]['name'])}</th>" for s in adapter_order)
    header = f"<thead><tr><th>Test set</th><th>Metric</th>{col_headers}</tr></thead>"

    rows = []
    for i, tkey in enumerate(tkeys):
        tlabel = TEST_SET_LABELS.get(tkey, tkey)
        for metric_key, metric_label, fmt_fn in [
            ("accuracy", "Accuracy", lambda v: f"{v:.2%}"),
            ("macro_f1", "Macro F1", lambda v: f"{v:.4f}"),
        ]:
            values = {
                stem: loaded[stem]["test_sets"][tkey][metric_key]
                for stem in adapter_order
                if tkey in loaded[stem].get("test_sets", {})
                and metric_key in loaded[stem]["test_sets"][tkey]
            }

            defined = list(values.values())
            best_val  = max(defined) if defined else None
            worst_val = min(defined) if defined else None
            spread = (best_val - worst_val) if best_val is not None and worst_val is not None else 0

            cells = []
            for stem in adapter_order:
                if stem not in values:
                    cells.append("<td class='na'>—</td>")
                else:
                    v = values[stem]
                    css = ""
                    if spread > 0:
                        if v == best_val:
                            css = " class='best'"
                        elif v == worst_val:
                            css = " class='worst'"
                    cells.append(f"<td{css}>{fmt_fn(v)}</td>")

            sep = " class='group-sep'" if metric_key == "accuracy" and i > 0 else ""
            test_cell = f"<td rowspan='2'>{escape(tlabel)}</td>" if metric_key == "accuracy" else ""
            rows.append(f"<tr{sep}>{test_cell}<td>{metric_label}</td>{''.join(cells)}</tr>")

    return (
        "<table class='summary'>"
        + header
        + "<tbody>" + "".join(rows) + "</tbody>"
        + "</table>"
    )


def build_per_class_table(loaded: dict, adapter_order: list[str], tkey: str, classes: list[str]) -> str:
    col_headers = "".join(f"<th>{escape(loaded[s]['name'])}</th>" for s in adapter_order)
    header = f"<thead><tr><th>Class</th>{col_headers}</tr></thead>"

    rows = []
    for cls in classes:
        cells = [f"<td>{escape(LABEL_SHORT.get(cls, cls))}</td>"]
        for stem in adapter_order:
            pc = loaded[stem].get("test_sets", {}).get(tkey, {}).get("per_class", {}).get(cls)
            cells.append(f"<td>{pc['f1']:.3f}</td>" if pc else "<td class='na'>—</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table class='compare'>"
        + header
        + "<tbody>" + "".join(rows) + "</tbody>"
        + "</table>"
    )


def build_report(loaded: dict, adapter_order: list[str], meta: dict) -> str:
    tkey_order = ["test_a", "test_b", "test_c", "test_d"]
    if meta.get("multilingual"):
        tkey_order.append("test_e")
    active_tkeys = [
        tk for tk in tkey_order
        if any(tk in loaded[s].get("test_sets", {}) for s in adapter_order)
    ]

    body = [f"<style>{CSS}</style>", "<h1>Romansh Idiom Classifier — Comparison Report</h1>"]
    body.append("<h2>Summary</h2>")
    body.append(build_summary_table(loaded, adapter_order, active_tkeys))

    for tkey in active_tkeys:
        tlabel = TEST_SET_LABELS.get(tkey, tkey)
        body.append(f"<h2>{escape(tlabel)}</h2>")

        seen: dict[str, bool] = {}
        for stem in adapter_order:
            for cls in loaded[stem].get("test_sets", {}).get(tkey, {}).get("per_class", {}):
                seen[cls] = True

        canonical = list(LABEL_SHORT.keys())
        classes = [c for c in canonical if c in seen] + [c for c in seen if c not in canonical]

        if classes:
            body.append("<h3>Per-class F1</h3>")
            body.append(build_per_class_table(loaded, adapter_order, tkey, classes))

        cms = [
            (loaded[stem]["name"], loaded[stem]["test_sets"][tkey]["cm_png_b64"])
            for stem in adapter_order
            if "cm_png_b64" in loaded[stem].get("test_sets", {}).get(tkey, {})
        ]
        if cms:
            body.append("<h3>Confusion Matrices</h3><div class='cm-grid'>")
            for display, b64 in cms:
                body.append(
                    f"<figure>"
                    f"<img src='data:image/png;base64,{b64}' alt='confusion matrix {escape(display)}'>"
                    f"<figcaption>{escape(display)}</figcaption>"
                    f"</figure>"
                )
            body.append("</div>")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body.append(f"<footer>Generated: {timestamp}</footer>")

    return (
        "<!DOCTYPE html><html>"
        "<head><meta charset='utf-8'><title>Romansh Idiom Classifier — Comparison</title></head>"
        "<body>" + "".join(body) + "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description="Step 8: compare all models")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip running adapters; regenerate report from existing JSONs")
    args = parser.parse_args()

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    start_run("step8_compare", artifacts=[REPORT_PATH])

    meta = load_splits_meta()
    mode = "multilingual" if meta["multilingual"] else "Romansh-only"
    print(f"Mode: {mode} ({len(meta['languages'])} classes: {', '.join(meta['languages'])})")

    if not args.report_only:
        adapter_paths = sorted(ADAPTERS_DIR.glob("[0-9]*.py"))
        print(f"\nDiscovered {len(adapter_paths)} adapter(s):")
        for p in adapter_paths:
            print(f"  {p.name}")
        run_adapters(adapter_paths)
    else:
        print("\n--report-only: skipping adapter execution")

    json_paths = sorted(COMPARE_DIR.glob("[0-9]*.json"))
    print(f"\nLoading {len(json_paths)} result file(s)...")
    loaded = load_results(json_paths)

    if not loaded:
        print("No result files found. Run adapters first.")
        return

    adapter_order = list(loaded.keys())  # already sorted by filename
    print(f"Models in report: {', '.join(loaded[s]['name'] for s in adapter_order)}")

    print("\nBuilding report...")
    html = build_report(loaded, adapter_order, meta)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
