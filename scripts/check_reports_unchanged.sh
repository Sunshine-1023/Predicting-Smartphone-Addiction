#!/usr/bin/env bash
# Fail if paths under reports/ changed relative to a before-snapshot file.
set -euo pipefail

usage() {
  echo "Usage: $0 snapshot <file> | check <before-file>" >&2
  exit 2
}

snapshot_reports() {
  {
    git diff --binary -- reports/ || true
    echo "---UNTRACKED---"
    while IFS= read -r path; do
      [ -z "$path" ] && continue
      echo "$path"
      # Hash contents so edits to already-dirty files are detected.
      if [ -f "$path" ]; then
        shasum -a 256 "$path"
      fi
    done < <(git ls-files --others --exclude-standard -- reports/)
    echo "---TRACKED-HASHES---"
    while IFS= read -r path; do
      [ -z "$path" ] && continue
      if [ -f "$path" ]; then
        echo "$path"
        shasum -a 256 "$path"
      elif [ ! -e "$path" ]; then
        echo "$path"
        echo "MISSING"
      fi
    done < <(git ls-files -- reports/)
  }
}

cmd="${1:-}"
file="${2:-}"
[ -n "$cmd" ] && [ -n "$file" ] || usage

case "$cmd" in
  snapshot)
    snapshot_reports >"$file"
    ;;
  check)
    after="$(mktemp)"
    snapshot_reports >"$after"
    if ! cmp -s "$file" "$after"; then
      echo "ERROR: tests mutated files under reports/" >&2
      diff -u "$file" "$after" >&2 || true
      rm -f "$after"
      exit 1
    fi
    rm -f "$after"
    echo "reports/ unchanged by tests"
    ;;
  *)
    usage
    ;;
esac
