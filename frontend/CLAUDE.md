# frontend/ — Next.js 16 Application

## Stack
Next.js 16 (App Router, Turbopack), React 19, TanStack React Query, Zustand, TailwindCSS 4, Recharts, lucide-react

## Commands
```bash
pnpm install    # Install deps
pnpm dev        # Dev server :3000
pnpm build      # Production build
```

## Pages
- `/` — Dashboard: YearSelector + AlgorithmSelector tabs + 50-reach grid
- `/series/[reach_id]/[bank_side]` — Time series chart + forecast table
- `/evaluate` — All-algorithm metrics comparison + TFT detail panel
- `/anomaly` — PELT changepoint detection table
- `/compare` — Multi-model forecast chart + metrics table
- `/validate` — Upload 2021–2025 XLSX, per-algorithm comparison + best model detection

## Key Patterns
- No API routes — components fetch backend directly via `src/lib/api.js`
- Path alias: `@/*` → `./src/*`
- CSS variables in `globals.css`: `--erosion` (red), `--deposition` (green), `--accent` (cyan)
- Algorithm config centralized in `src/lib/algorithms.js` (11 algorithms with colors)
- Floating InfoButton on every page (bottom-left) for educational content
