#!/usr/bin/env bash
# Copy model JSON exports from repo root into the package before building/publishing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="$(cd "$(dirname "$0")/.." && pwd)/src/romansh_idiom_classifier/models"

for name in lr lr_lite svm svm_lite; do
    src="$REPO_ROOT/models/${name}_export.json"
    if [[ -f "$src" ]]; then
        cp "$src" "$DEST/"
        echo "  copied ${name}_export.json"
    else
        echo "  WARNING: $src not found — skipping"
    fi
done
