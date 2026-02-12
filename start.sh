#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Jamuna Bankline Prediction System ==="
echo ""

# --- Backend ---
echo "[backend] Starting FastAPI on http://localhost:8000 ..."
cd "$ROOT"
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# --- Frontend ---
echo "[frontend] Starting Next.js on http://localhost:3000 ..."
cd "$ROOT/frontend"
pnpm dev &
FRONTEND_PID=$!

# --- Cleanup on exit ---
cleanup() {
  echo ""
  echo "Stopping servers..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:3000"
echo "  API Docs → http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

wait
