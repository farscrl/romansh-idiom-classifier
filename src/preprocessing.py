"""
Text loading and cleaning for all data sources.
"""

import csv
import html as html_module
import json
import re
import unicodedata
from pathlib import Path

COMBINING_DOT_BELOW = "̣"

IDIOMS = ["rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader", "rm-rumgr"]

FMR_DIR = Path("data/01_raw/fmr")
PLEDARI_GROND_DIR = Path("data/01_raw/pledari-grond")
TEXTBOOKS_DIR = Path("data/01_raw/textbooks")
THEATER_PLAYS_DIR = Path("data/01_raw/theater-plays")
CANTON_LAWS_DIR = Path("data/01_raw/canton-laws")
PROPRIETARY_DIR = Path("data/01_raw/proprietary-data")
WIKIPEDIA_DIR = Path("data/01_raw/wikipedia")
WIKIPEDIA_LANGS = ["de", "fr", "it", "en"]

PLEDARI_GROND_FILES = {
    "pledarigrond_export_json_sursilvan.json": "rm-sursilv",
    "pledarigrond_export_json_sutsilvan.json": "rm-sutsilv",
    "pledarigrond_export_json_surmiran.json": "rm-surmiran",
    "pledarigrond_export_json_rumantschgrischun.json": "rm-rumgr",
    "pledarigrond_export_json_puter.json": "rm-puter",
    "pledarigrond_export_json_vallader.json": "rm-vallader",
}

TEXTBOOK_IDIOMS = ["rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader"]
TEXTBOOK_SPLITS = ["train", "validation", "test", "no_surmiran"]

RTR_DIR = Path("data/01_raw/rtr-transcripts")

# Map folder name prefix (before first '-cc-') to idiom label.
# The prefix 'rm' alone (no idiom suffix) = Rumantsch Grischun.
RTR_PREFIX_TO_IDIOM = {
    "rm":          "rm-rumgr",
    "rmputer":     "rm-puter",
    "rmsursilv":   "rm-sursilv",
    "rmsursiv":    "rm-sursilv",    # typo variant in some releases
    "rmsutsilv":   "rm-sutsilv",
    "rmsurmiran":  "rm-surmiran",
    "rmvallader":  "rm-vallader",
}


# Invisible / format characters to delete entirely (applied after NFKD decomposition)
_DELETE = str.maketrans("", "", "".join(chr(c) for c in (
    0x00AD,  # soft hyphen
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0xFEFF,  # BOM / zero-width no-break space
    0x2028,  # line separator
    0x2029,  # paragraph separator
)))
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE = re.compile(r"\d+")
_EDITORIAL_RE = re.compile(r"\s*\([a-z]+(?:/[a-z]+)+\)\s*$")  # strips (cdm/fmr), (cdo/rtr) etc.
_EXERCISE_ENUM_RE = re.compile(r"(?:^| )(?:\d+[a-z]?|[a-z])\) ")  # 1) 2a) c)
_FILL_BLANK_RE = re.compile(r"_{2,}|\.{3,}")

def normalize(text: str) -> str:
    text = html_module.unescape(text)               # &nbsp; -> U+00A0, &amp; -> & etc.
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = unicodedata.normalize("NFKD", text)
    text = text.replace(COMBINING_DOT_BELOW, "")
    text = text.translate(_DELETE)                  # soft hyphen, zero-width, BOM -> remove
    text = text.replace("\u00a0", " ")             # non-breaking space -> regular space
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub("$URL$", text)               # URLs -> $URL$
    text = _NUM_RE.sub("$NUM$", text)               # digit sequences -> $NUM$
    return " ".join(text.split())


def is_valid(text: str) -> bool:
    """Return True if text contains at least one letter."""
    return any(c.isalpha() for c in text)


def _collect_inflection_strings(obj) -> list[str]:
    """Recursively collect all non-empty string leaf values from an inflection dict."""
    results = []
    if isinstance(obj, dict):
        for v in obj.values():
            results.extend(_collect_inflection_strings(v))
    elif isinstance(obj, str):
        cleaned = normalize(obj)
        if cleaned and is_valid(cleaned):
            results.append(cleaned)
    return results


# ---------------------------------------------------------------------------
# Loaders — each returns a list of (idiom_label, text) tuples
# ---------------------------------------------------------------------------

def load_fmr() -> list[tuple[str, str]]:
    """Load FMR news texts from data/raw/fmr/data.jsonl."""
    path = FMR_DIR / "data.jsonl"
    if not path.exists():
        print("  [FMR] data.jsonl not found, skipping.")
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = row.get("variety", "").strip()
            text = normalize(row.get("text", ""))
            text = _EDITORIAL_RE.sub("", text).strip()
            if label in IDIOMS and is_valid(text):
                samples.append((label, text))
    print(f"  [FMR] Loaded {len(samples)} samples.")
    return samples


def load_pledari_grond() -> list[tuple[str, str]]:
    """
    Load dictionary entries from Pledari Grond JSON files.

    For each entry extracts:
      - rmStichwort (main headword)
      - all non-empty string leaf values from inflection forms
      - rm field from each example
    """
    samples = []
    for filename, label in PLEDARI_GROND_FILES.items():
        path = PLEDARI_GROND_DIR / filename
        if not path.exists():
            print(f"  [Pledari Grond] {filename} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        count = 0
        for entry in entries:
            stichwort_raw = entry.get("rmStichwort") or ""
            if stichwort_raw.startswith("cf. "):
                continue  # cross-reference entry — skip entirely

            # Main headword
            stichwort = normalize(stichwort_raw)
            if is_valid(stichwort):
                samples.append((label, stichwort))
                count += 1

            # Inflection forms — flatten all string leaf values into one sample
            inflection = entry.get("inflection")
            if isinstance(inflection, dict):
                forms = _collect_inflection_strings(inflection)
                if forms:
                    samples.append((label, " ".join(forms)))
                    count += 1

            # Examples
            for example in entry.get("examples") or []:
                rm = normalize(example.get("rm") or "")
                if is_valid(rm):
                    samples.append((label, rm))
                    count += 1

        print(f"  [Pledari Grond] {label}: loaded {count} samples from {filename}.")
    return samples


def load_textbooks() -> list[tuple[str, str]]:
    """Load all textbook texts from data/01_raw/textbooks/ (all HuggingFace splits pooled)."""
    samples = []
    for idiom in TEXTBOOK_IDIOMS:
        idiom_dir = TEXTBOOKS_DIR / idiom
        for split in TEXTBOOK_SPLITS:
            path = idiom_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    text = normalize(row.get("text") or "")
                    text = _EXERCISE_ENUM_RE.sub(" ", text).strip()
                    text = _FILL_BLANK_RE.sub("", text).strip()
                    if is_valid(text):
                        samples.append((idiom, text))
    print(f"  [Textbooks] Loaded {len(samples)} samples.")
    return samples


def load_theater_plays() -> list[tuple[str, str]]:
    """Load theater play texts from data/01_raw/theater-plays/data.jsonl."""
    path = THEATER_PLAYS_DIR / "data.jsonl"
    if not path.exists():
        print("  [Theater Plays] data.jsonl not found, skipping.")
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = row.get("idiom", "").strip()
            text = normalize(row.get("page_text", ""))
            if label in IDIOMS and is_valid(text):
                samples.append((label, text))
    print(f"  [Theater Plays] Loaded {len(samples)} samples.")
    return samples


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text


def load_rtr_transcripts() -> list[tuple[str, str]]:
    """
    Load RTR Common Voice transcripts from data/raw/rtr-transcripts/.

    Each subdirectory contains validated.tsv (Common Voice format).
    The idiom is derived from the folder name prefix before '-cc-'.
    Only validated.tsv is used (human-verified transcripts).
    Unextracted .tgz files are skipped with a warning.
    """
    samples = []
    for entry in sorted(RTR_DIR.iterdir()):
        if entry.suffix == ".tgz":
            print(f"  [RTR] {entry.name} is not extracted — skipping. Extract it first.")
            continue
        if not entry.is_dir():
            continue
        prefix = entry.name.split("-cc-")[0]
        idiom = RTR_PREFIX_TO_IDIOM.get(prefix)
        if idiom is None:
            print(f"  [RTR] Unknown folder prefix '{prefix}' in {entry.name}, skipping.")
            continue
        validated = entry / "validated.tsv"
        if not validated.exists():
            print(f"  [RTR] No validated.tsv in {entry.name}, skipping.")
            continue
        count = 0
        with open(validated, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                raw = row.get("sentence", "")
                text = normalize(_strip_html(raw))
                if is_valid(text):
                    samples.append((idiom, text))
                    count += 1
        print(f"  [RTR] {idiom} ({entry.name}): {count} samples.")
    return samples


def load_canton_laws() -> list[tuple[str, str]]:
    """Load Rumantsch Grischun canton laws from data/01_raw/canton-laws/data.jsonl."""
    path = CANTON_LAWS_DIR / "data.jsonl"
    if not path.exists():
        print("  [Canton Laws] data.jsonl not found, skipping.")
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = normalize(row.get("RM") or "")
            if is_valid(text):
                samples.append(("rm-rumgr", text))
    print(f"  [Canton Laws] Loaded {len(samples)} samples.")
    return samples


def load_wikipedia(lang: str) -> list[tuple[str, str]]:
    """Load preprocessed Wikipedia paragraphs for one language from data/01_raw/wikipedia/{lang}/data.jsonl."""
    path = WIKIPEDIA_DIR / lang / "data.jsonl"
    if not path.exists():
        print(f"  [Wikipedia/{lang}] data.jsonl not found, skipping.")
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = normalize(row.get("text", ""))
            if is_valid(text):
                samples.append((lang, text))
    print(f"  [Wikipedia/{lang}] Loaded {len(samples):,} samples.")
    return samples


def load_proprietary() -> list[tuple[str, str]]:
    """
    Load proprietary TSV files from data/raw/proprietary-data/.

    Expected format (no header): <idiom_label> TAB <text>
    All .tsv files in the directory are loaded. Returns [] if directory is empty.
    """
    tsv_files = sorted(PROPRIETARY_DIR.glob("*.tsv"))
    if not tsv_files:
        print("  [Proprietary] No .tsv files found, skipping.")
        return []
    samples = []
    for path in tsv_files:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                label, text = parts
                label = label.strip()
                text = normalize(text)
                if label in IDIOMS and is_valid(text):
                    samples.append((label, text))
                    count += 1
        print(f"  [Proprietary] {path.name}: {count} samples.")
    return samples