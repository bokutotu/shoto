#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HS_FILES=$(find shoto isl -name "*.hs" -type f)

if [ -n "$HS_FILES" ]; then
    echo "$HS_FILES" | xargs stylish-haskell -i
    echo "$HS_FILES" | xargs fourmolu -i
fi
