# frontend/src/lib/ — Utility Modules

## Files
- `api.js` — Fetch wrapper functions for all backend endpoints. Base URL from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). Functions: `getHealth`, `postPredict`, `postPredictBaseline`, `getEvaluation`, `getEvaluateCompare`, `getChangepoints`, `getSeriesHistory`, `getPredictionForYear`, `getBaselineYearPrediction`, `postTrain`, `getTrainStatus`, `getTrainLogs`
- `algorithms.js` — Centralized algorithm config: `ALGORITHMS` array (11 entries with key/label/color), `ALGO_COLOR_MAP`
- `parseValidationExcel.js` — Parses uploaded XLSX with 2-row header format. Returns `{years, data}`. Left bank values negated for coordinate convention.
- `pageInfo.js` — Educational content registry for InfoButton modal. Maps pageId → title + sections with formulas and explanations.
