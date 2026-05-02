# Repository Guidelines

## Project Structure & Module Organization

This is a full-stack ML system for Jamuna River bankline prediction. Backend Python code lives in `src/`: `serving/` exposes FastAPI endpoints, `models/` wraps TFT and 10 baselines, `training/` contains training/tuning/evaluation commands, `data/` handles Excel loading, and `anomaly/` contains changepoint/VAE logic. Configuration is in `configs/config.yaml`.

Frontend code is in `frontend/src/`: routes in `app/`, components in `components/`, helpers in `lib/`, and Zustand state in `store/`. Datasets live in `data/` and `dataset/`; checkpoints, joblib baselines, and caches live in `models/`.

## Build, Test, and Development Commands

- `python start.py` starts the full stack and installs missing dependencies.
- `uv sync --dev` installs Python dependencies and dev tools.
- `uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000` runs the API.
- `cd frontend && pnpm dev` runs the frontend on `http://localhost:3000`.
- `docker compose up --build` builds and runs both services in containers.
- `uv run python -m src.training.train_all_baselines` trains all baseline models.
- `uv run pytest tests/` runs backend tests.
- `uv run ruff check src/` and `uv run ruff format src/` lint and format.
- `uv run mypy src/` runs strict Python type checks.
- `cd frontend && pnpm build` verifies production frontend output.

## Coding Style & Naming Conventions

Use Python 3.12+ with 4-space indentation, public type hints, and descriptive `snake_case` names. Ruff uses line length 100 with E/F/I/UP/B rules; mypy is strict. Keep schemas in `src/serving/schemas.py`.

Frontend files use PascalCase React components, for example `ReachGrid.js`, and camelCase helpers. Use the `@/*` alias for `frontend/src/*`. Components fetch the backend through `src/lib/api.js`; avoid Next.js API routes unless architecture changes.

## Testing Guidelines

Backend tests use pytest under `tests/`, matching `test_*.py`. Add focused tests for preprocessing, schemas, baseline wrappers, and FastAPI endpoints. `pytest-asyncio` is configured. No test suite is currently committed, so include tests with behavioral backend changes.

## Commit & Pull Request Guidelines

Git history mixes checkpoint commits and conventional commits; prefer conventional messages such as `fix: correct sign convention` or `feat: add dashboard export`. Keep commits scoped.

Pull requests should include a summary, changed backend/frontend areas, test results, and screenshots for UI changes. Link issues when available and call out dataset, model, or config changes.

## Security & Configuration Tips

Do not commit secrets, credentials, or generated logs. Keep runtime configuration in `configs/config.yaml` or environment variables. Treat model files, Excel datasets, `mlruns/`, `logs/training/`, and `lightning_logs/` as artifacts; update them only when intentional.
