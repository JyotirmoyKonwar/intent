#!/usr/bin/env bash
set -euo pipefail

dir="${1:-.}"

find "$dir" -type f -name '*.*' | while read -r f; do
  printf '%s %s\n' "${f##*.}" "$(wc -l < "$f")"
done | awk '{n[$1] += $2} END {for (e in n) printf "%-8s %d\n", e, n[e]}'

if ! grep -q "alias loc=" "$HOME/.bashrc" 2>/dev/null; then
  echo "alias loc='bash $(cd "$(dirname "$0")" && pwd)/loc.sh'" >> "$HOME/.bashrc"
fi
