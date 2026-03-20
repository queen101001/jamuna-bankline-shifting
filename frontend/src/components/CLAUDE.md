# frontend/src/components/ — React Components

Organized by feature folder. Each folder contains components specific to one page/feature.

## Feature Folders
- `dashboard/` — ReachGrid, ReachCard, ErosionBar, YearSelector
- `series/` — SeriesHeader, HistoryChart (Recharts line chart with quantile bands)
- `evaluate/` — MetricsPanel (NSE/KGE/RMSE/MAE cards + coverage gauge)
- `anomaly/` — ChangepointTable (sortable table with protection badges)
- `compare/` — ComparisonChart (multi-model Recharts), MetricsTable (all-algo comparison)
- `validate/` — ExcelUploader, ValidationResults, BankComparisonChart, ErrorSummary
- `training/` — TrainingDrawer (slide-out panel for background training)
- `nav/` — NavBar (top navigation with page links)
- `ui/` — Shared: AlgorithmSelector, LoadingSpinner, QuantileBadge, InfoButton, InfoModal

## Conventions
- All components use `'use client'` directive
- Styling: inline `style={{}}` with CSS variables (`var(--card)`, `var(--border)`, etc.)
- Charts: Recharts (LineChart, ComposedChart, ResponsiveContainer)
- Icons: lucide-react
