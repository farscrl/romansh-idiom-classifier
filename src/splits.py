import json
from pathlib import Path

SPLITS_DIR = Path("data/03_splits")

ROMANSH_LANGUAGES = ["rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader", "rm-rumgr"]


def load_splits_meta() -> dict:
    """Read data/03_splits/splits-meta.json. Falls back to Romansh-only defaults if not found."""
    path = SPLITS_DIR / "splits-meta.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"multilingual": False, "languages": ROMANSH_LANGUAGES}
