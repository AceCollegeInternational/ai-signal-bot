#!/usr/bin/env bash
set -euo pipefail

# Fail fast if unresolved merge-conflict markers exist in source files.
if rg -n "^(<<<<<<<|=======|>>>>>>>)" --glob "*.py" --glob "*.yaml" --glob "*.yml" --glob "*.md" . >/tmp/conflicts.txt; then
  echo "❌ Unresolved merge conflict markers found:"
  cat /tmp/conflicts.txt
  exit 1
fi

# Install pandas-ta from PyPI to avoid outbound git clone restrictions in Render builds.
pip install "pandas-ta==0.4.71b0"

# Install vectorbt without dependency resolution so pip won't fail on pandas-ta metadata constraints.
pip install vectorbt==0.26.2 --no-deps

# Install the rest of the project dependencies.
pip install -r requirements.txt
