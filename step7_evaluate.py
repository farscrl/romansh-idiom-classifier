"""
Step 7: Evaluate SVM and Logistic Regression models on all test sets.

Loads models/svm.joblib and models/lr.joblib, runs predictions on test_a through test_d
(skipping any that are empty or missing), and generates an HTML report at
data/04_evaluation/report.html.

Metrics reported per model × test set:
  - Accuracy
  - Macro F1
  - Per-class precision / recall / F1 / support
  - Confusion matrix (heatmap)
"""

import base64
import io
import time
from collections import Counter
from html import escape
from pathlib import Path

from src.run_log import start_run
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

matplotlib.use("Agg")

MODELS_DIR = Path("models")
SPLITS_DIR = Path("data/03_splits")
EVAL_DIR = Path("data/04_evaluation")
REPORT_PATH = EVAL_DIR / "report.html"

IDIOMS = ["rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader", "rm-rumgr"]
IDIOM_SHORT = {
    "rm-sursilv": "Sursilv",
    "rm-sutsilv": "Sutsilv",
    "rm-surmiran": "Surmiran",
    "rm-puter": "Puter",
    "rm-vallader": "Vallader",
    "rm-rumgr": "RumGr",
}

TEST_SETS = {
    "test_a": ("Test A — news (FMR)", SPLITS_DIR / "test/test_a.tsv"),
    "test_b": ("Test B — speech transcripts (RTR)", SPLITS_DIR / "test/test_b.tsv"),
    "test_c": ("Test C — schoolbooks (Textbooks)", SPLITS_DIR / "test/test_c.tsv"),
    "test_d": ("Test D — proprietary (out-of-domain)", SPLITS_DIR / "test/test_d.tsv"),
}

MODELS = {
    "SVM":      MODELS_DIR / "svm.joblib",
    "SVM-lite": MODELS_DIR / "svm_lite.joblib",
    "LR":       MODELS_DIR / "lr.joblib",
    "LR-lite":  MODELS_DIR / "lr_lite.joblib",
}


def load_split(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    if not path.exists():
        return texts, labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    return texts, labels


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_confusion_matrix(y_true, y_pred, classes: list[str]) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short = [IDIOM_SHORT.get(c, c) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            norm_val = cm_norm[i, j]
            color = "white" if norm_val > 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def per_class_table(y_true, y_pred, classes: list[str]) -> str:
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    rows = []
    for cls in classes:
        d = report.get(cls, {})
        short = IDIOM_SHORT.get(cls, cls)
        rows.append(
            f"<tr>"
            f"<td>{escape(short)}</td>"
            f"<td>{d.get('precision', 0):.3f}</td>"
            f"<td>{d.get('recall', 0):.3f}</td>"
            f"<td>{d.get('f1-score', 0):.3f}</td>"
            f"<td>{int(d.get('support', 0))}</td>"
            f"</tr>"
        )
    macro = report.get("macro avg", {})
    rows.append(
        f"<tr class='macro'>"
        f"<td><strong>macro avg</strong></td>"
        f"<td>{macro.get('precision', 0):.3f}</td>"
        f"<td>{macro.get('recall', 0):.3f}</td>"
        f"<td>{macro.get('f1-score', 0):.3f}</td>"
        f"<td>{int(macro.get('support', 0))}</td>"
        f"</tr>"
    )
    return (
        "<table class='metrics'>"
        "<thead><tr><th>Idiom</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def top_features_table(model, n: int = 10) -> str:
    """HTML table of top-n positive-weight features per class."""
    try:
        feat_union = model.named_steps["features"]
        clf = model.named_steps["clf"]
    except (AttributeError, KeyError):
        return "<p class='skipped'>Feature names not available for this model type.</p>"

    feature_names = feat_union.get_feature_names_out()
    classes = clf.classes_

    per_class = []
    for coef_row in clf.coef_:
        top_idx = np.argsort(coef_row)[::-1][:n]
        per_class.append([(feature_names[i], coef_row[i]) for i in top_idx if coef_row[i] > 0])

    def fmt_feature(name: str, weight: float) -> str:
        tag, feat = name.split("__", 1)
        feat_display = escape(feat.replace(" ", "·"))
        return (
            f'<span class="feat-tag">{escape(tag)}</span>'
            f'<code>{feat_display}</code>'
            f' <span class="feat-w">({weight:.3f})</span>'
        )

    short_names = [IDIOM_SHORT.get(c, c) for c in classes]
    header = "<tr><th>#</th>" + "".join(f"<th>{escape(s)}</th>" for s in short_names) + "</tr>"
    rows = []
    for rank in range(n):
        cells = [f"<td class='rank'>{rank + 1}</td>"]
        for class_feats in per_class:
            if rank < len(class_feats):
                cells.append(f"<td>{fmt_feature(*class_feats[rank])}</td>")
            else:
                cells.append("<td>—</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table class='features'>"
        f"<thead>{header}</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: .4rem; }
h2 { margin-top: 2.5rem; color: #1a4f8a; }
h3 { margin-top: 1.5rem; color: #444; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; margin: 1rem 0; }
.summary-card { border: 1px solid #ccc; border-radius: 6px; padding: 1rem; background: #f9f9f9; }
.summary-card .model { font-size: .85rem; color: #666; }
.summary-card .score { font-size: 1.8rem; font-weight: bold; color: #1a4f8a; }
.summary-card .label { font-size: .9rem; }
table.metrics { border-collapse: collapse; width: 100%; margin: .8rem 0; }
table.metrics th, table.metrics td { padding: .4rem .8rem; border: 1px solid #ddd; text-align: right; }
table.metrics th { background: #e8eef6; }
table.metrics td:first-child, table.metrics th:first-child { text-align: left; }
tr.macro { background: #f0f4fa; font-weight: 500; }
.cm-wrap { display: flex; gap: 2rem; flex-wrap: wrap; align-items: flex-start; margin: 1rem 0; }
.cm-wrap img { max-width: 480px; border: 1px solid #ddd; border-radius: 4px; }
.skipped { color: #888; font-style: italic; }
table.features { border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .83rem; }
table.features th, table.features td { padding: .35rem .6rem; border: 1px solid #ddd; vertical-align: top; }
table.features thead th { background: #e8eef6; text-align: center; }
table.features td.rank { text-align: center; color: #999; width: 2rem; }
.feat-tag { display: inline-block; font-size: .7rem; background: #dde8f0; padding: 1px 4px; border-radius: 3px; color: #356; margin-right: 3px; }
.feat-w { color: #888; font-size: .8rem; }
code { font-family: monospace; background: #f4f4f4; padding: 1px 3px; border-radius: 2px; }
"""


def build_html(results: dict, top_features_html: dict) -> str:
    """results: {test_key: {model_name: {acc, f1, table_html, cm_b64, n_samples, classes}}}"""
    body_parts = [f"<style>{CSS}</style>", "<h1>Romansh Idiom Identification — Evaluation Report</h1>"]

    # Summary table: rows = test sets, cols = models
    model_names = list(MODELS.keys())
    body_parts.append("<h2>Summary</h2>")
    body_parts.append("<div class='summary-grid'>")
    for tkey, (tlabel, _) in TEST_SETS.items():
        if tkey not in results:
            continue
        for mname in model_names:
            if mname not in results[tkey]:
                continue
            r = results[tkey][mname]
            body_parts.append(
                f"<div class='summary-card'>"
                f"<div class='model'>{escape(mname)} · {escape(tlabel)}</div>"
                f"<div class='score'>{r['acc']:.1%}</div>"
                f"<div class='label'>accuracy &nbsp;·&nbsp; macro F1 {r['f1']:.3f} &nbsp;·&nbsp; n={r['n_samples']:,}</div>"
                f"</div>"
            )
    body_parts.append("</div>")

    # Top features per class (one section per model, independent of test sets)
    body_parts.append("<h2>Top 10 Features per Class</h2>")
    for mname in model_names:
        if mname not in top_features_html:
            continue
        body_parts.append(f"<h3>{escape(mname)}</h3>")
        body_parts.append(top_features_html[mname])

    # Per-test-set sections
    for tkey, (tlabel, _) in TEST_SETS.items():
        if tkey not in results:
            body_parts.append(f"<h2>{escape(tlabel)}</h2><p class='skipped'>Skipped (empty or missing).</p>")
            continue
        body_parts.append(f"<h2>{escape(tlabel)}</h2>")
        for mname in model_names:
            if mname not in results[tkey]:
                body_parts.append(f"<h3>{escape(mname)}</h3><p class='skipped'>Model not found.</p>")
                continue
            r = results[tkey][mname]
            body_parts.append(
                f"<h3>{escape(mname)} &mdash; accuracy {r['acc']:.1%} &nbsp;·&nbsp; macro F1 {r['f1']:.3f}</h3>"
            )
            body_parts.append("<div class='cm-wrap'>")
            body_parts.append(f"<img src='data:image/png;base64,{r['cm_b64']}' alt='confusion matrix'>")
            body_parts.append(r["table_html"])
            body_parts.append("</div>")

    return "<!DOCTYPE html><html><head><meta charset='utf-8'><title>RII Evaluation</title></head><body>" + "".join(body_parts) + "</body></html>"


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    start_run("step7_evaluate", artifacts=[REPORT_PATH])

    # Load models
    loaded_models = {}
    for mname, mpath in MODELS.items():
        if not mpath.exists():
            print(f"[{mname}] model not found at {mpath} — skipping.")
            continue
        print(f"Loading {mname} from {mpath}...")
        loaded_models[mname] = joblib.load(mpath)

    if not loaded_models:
        print("No models found. Run steps 4 and 6 first.")
        return

    results = {}

    for tkey, (tlabel, tpath) in TEST_SETS.items():
        print(f"\n{'='*60}")
        print(f"Test set: {tlabel}")
        texts, labels = load_split(tpath)
        if not texts:
            print(f"  Empty or missing — skipping.")
            continue

        classes = [c for c in IDIOMS if c in set(labels)]
        dist = Counter(labels)
        print(f"  {len(texts):,} samples across {len(classes)} idioms:")
        for cls in classes:
            short = IDIOM_SHORT.get(cls, cls)
            print(f"    {short:<10} {dist[cls]:>5,}")
        results[tkey] = {}

        for mname, model in loaded_models.items():
            print(f"\n  [{mname}] predicting...")
            t_pred = time.time()
            preds = model.predict(texts)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            print(f"  [{mname}] done in {time.time()-t_pred:.1f}s")
            print(f"  [{mname}] accuracy={acc:.4f}  macro-F1={f1:.4f}")
            print(f"  [{mname}] per-class F1:")
            f1_per = f1_score(labels, preds, average=None, labels=classes, zero_division=0)
            for cls, score in zip(classes, f1_per):
                bar = "█" * int(score * 20)
                print(f"    {IDIOM_SHORT.get(cls, cls):<10} {score:.3f}  {bar}")

            cm_b64 = plot_confusion_matrix(labels, preds, classes)
            table_html = per_class_table(labels, preds, classes)

            results[tkey][mname] = {
                "acc": acc,
                "f1": f1,
                "n_samples": len(texts),
                "classes": classes,
                "cm_b64": cm_b64,
                "table_html": table_html,
            }

    print("\nExtracting top features per class...")
    top_features_html = {}
    for mname, model in loaded_models.items():
        top_features_html[mname] = top_features_table(model)
        print(f"  [{mname}] done")

    html = build_html(results, top_features_html)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()