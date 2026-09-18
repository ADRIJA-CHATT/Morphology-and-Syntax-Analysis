#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -file-line-error report.tex
else
  if ! command -v xelatex >/dev/null 2>&1; then
    echo "Error: latexmk and xelatex are both unavailable." >&2
    exit 1
  fi
  xelatex -interaction=nonstopmode -file-line-error report.tex >/dev/null
  xelatex -interaction=nonstopmode -file-line-error report.tex >/dev/null
fi
