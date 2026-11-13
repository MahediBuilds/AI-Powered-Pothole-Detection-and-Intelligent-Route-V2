#!/usr/bin/env bash
# Render every .dot in this folder to ../<name>.pdf and a preview PNG; print natural size in inches.
cd "$(dirname "$0")"
for f in ${@:-*.dot}; do
  n="${f%.dot}"
  dot -Tpdf "$f" -o "../$n.pdf" || { echo "FAIL $f"; continue; }
  dot -Tpng -Gdpi=110 "$f" -o "preview/$n.png"
  printf "%s  " "$n"; dot -Tplain "$f" | head -1
done
