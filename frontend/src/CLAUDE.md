# frontend/src/ — Source Layout

## Directories
- `app/` — Next.js App Router pages (page.js per route)
- `components/` — React components organized by feature folder
- `lib/` — Utility modules (API client, algorithm config, Excel parser)
- `store/` — Zustand state management (index.js)

## State (Zustand)
- `selectedYear` — Forecast year (2021–2100)
- `activeAlgorithm` — Selected algorithm key (default: 'tft')
- `validationData` — Uploaded validation Excel data
- `trainingJobId`, `toastMessage` — Training UI state

## Data Flow
Pages → `lib/api.js` → FastAPI backend (:8000) → Components render with Recharts
