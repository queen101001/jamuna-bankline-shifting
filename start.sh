#!/usr/bin/env bash
# Convenience wrapper — delegates to the cross-platform start.py
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/start.py" "$@"
