#!/usr/bin/env bash
set -euo pipefail

# Fail fast if unresolved merge-conflict markers exist in source files.
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) -print0 \
  | xargs -0 -r grep -nE "^(<<<<<<<|=======|>>>>>>>)" >/tmp/conflicts.txt || true

if [ -s /tmp/conflicts.txt ]; then
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
