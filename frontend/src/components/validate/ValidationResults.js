'use client';
import { useMemo, useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import { X, Trophy } from 'lucide-react';
import { getPredictionForYear, getBaselineYearPrediction } from '@/lib/api';
import { ALGORITHMS } from '@/lib/algorithms';
import useAppStore from '@/store';
import BankComparisonChart from './BankComparisonChart';
import ErrorSummary from './ErrorSummary';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

const BASELINE_KEYS = ALGORITHMS.filter((a) => a.key !== 'tft').map((a) => a.key);

function computeMetrics(pairs) {
  if (!pairs.length) return null;
  const errors = pairs.map((p) => p.observed - p.predicted);
  const absErrors = errors.map(Math.abs);
  const mae = absErrors.reduce((a, b) => a + b, 0) / absErrors.length;
  const rmse = Math.sqrt(errors.reduce((a, e) => a + e * e, 0) / errors.length);
  const maxError = Math.max(...absErrors);
  return { mae, rmse, maxError, count: pairs.length };
}

export default function ValidationResults() {
  const { validationData, clearValidation } = useAppStore();
  const [selectedAlgo, setSelectedAlgo] = useState('tft');

  const years = validationData?.years ?? [];

  // Fetch TFT predictions for each year
  const tftQueries = useQueries({
    queries: years.map((year) => ({
      queryKey: ['predict-year', year],
      queryFn: () => getPredictionForYear(year),
      staleTime: 60_000,
    })),
  });

  // Fetch baseline predictions for each (baseline × year)
  const baselineQueries = useQueries({
    queries: BASELINE_KEYS.flatMap((algoKey) =>
      years.map((year) => ({
        queryKey: ['predict-baseline-year', year, algoKey],
        queryFn: () => getBaselineYearPrediction(year, algoKey),
        staleTime: 60_000,
      }))
    ),
  });

  const isLoading = tftQueries.some((q) => q.isLoading) || baselineQueries.some((q) => q.isLoading);

  // Build per-algorithm merged data: { [algoKey]: { [year]: [{reach_id, bank_side, observed, predicted, error}] } }
  const perAlgoData = useMemo(() => {
    if (!validationData || isLoading) return {};

    const result = {};

    // Helper: build prediction map from query data
    function buildPredMap(predData) {
      const map = {};
      if (predData?.predictions) {
        for (const p of predData.predictions) {
          map[`${p.reach_id}-${p.bank_side}`] = p.q50;
        }
      }
      return map;
    }

    // Helper: merge observed with predictions
    function mergeYear(year, predMap) {
      const yearObs = validationData.data.filter((d) => d.year === year);
      return yearObs
        .map((obs) => {
          const key = `${obs.reach_id}-${obs.bank_side}`;
          const predicted = predMap[key];
          return {
            reach_id: obs.reach_id,
            bank_side: obs.bank_side,
            observed: obs.observed,
            predicted: predicted ?? null,
            error: predicted != null ? obs.observed - predicted : null,
          };
        })
        .filter((d) => d.predicted != null);
    }

    // TFT
    result.tft = {};
    for (let i = 0; i < years.length; i++) {
      const predMap = buildPredMap(tftQueries[i]?.data);
      result.tft[years[i]] = mergeYear(years[i], predMap);
    }

    // Baselines
    for (let bIdx = 0; bIdx < BASELINE_KEYS.length; bIdx++) {
      const algoKey = BASELINE_KEYS[bIdx];
      result[algoKey] = {};
      for (let yIdx = 0; yIdx < years.length; yIdx++) {
        const qIdx = bIdx * years.length + yIdx;
        const predMap = buildPredMap(baselineQueries[qIdx]?.data);
        result[algoKey][years[yIdx]] = mergeYear(years[yIdx], predMap);
      }
    }

    return result;
  }, [validationData, years, tftQueries, baselineQueries, isLoading]);

  // Compute per-algorithm global metrics
  const perAlgoMetrics = useMemo(() => {
    const metrics = {};
    for (const [algoKey, yearData] of Object.entries(perAlgoData)) {
      const allPairs = Object.values(yearData).flat();
      metrics[algoKey] = computeMetrics(allPairs);
    }
    return metrics;
  }, [perAlgoData]);

  // Determine best algorithm (lowest MAE)
  const bestAlgo = useMemo(() => {
    const entries = Object.entries(perAlgoMetrics).filter(([_, m]) => m != null);
    if (!entries.length) return null;
    entries.sort((a, b) => a[1].mae - b[1].mae);
    const [key, metrics] = entries[0];
    const algo = ALGORITHMS.find((a) => a.key === key);
    return { key, label: algo?.label || key, color: algo?.color || 'var(--accent)', metrics };
  }, [perAlgoMetrics]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center gap-4 py-12">
        <LoadingSpinner label="Fetching predictions from all algorithms..." />
      </div>
    );
  }

  const algoTabs = [
    ...ALGORITHMS,
    { key: 'all', label: 'All Models', color: 'var(--accent)' },
  ];

  const currentMetrics =
    selectedAlgo === 'all'
      ? bestAlgo?.metrics ?? null
      : perAlgoMetrics[selectedAlgo] ?? null;

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
            Comparing {years.length} year{years.length > 1 ? 's' : ''} ({years.join(', ')})
            {' '}&mdash;{' '}
            {validationData.data.length} observed data points
          </p>
        </div>
        <button
          onClick={clearValidation}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{
            background: 'rgba(239,68,68,0.1)',
            color: 'var(--erosion)',
            border: '1px solid rgba(239,68,68,0.25)',
          }}
        >
          <X size={14} />
          Clear
        </button>
      </div>

      {/* Best model badge */}
      {bestAlgo && (
        <div
          className="flex items-center gap-3 rounded-xl border px-5 py-3"
          style={{
            background: 'rgba(34,197,94,0.06)',
            borderColor: 'rgba(34,197,94,0.3)',
          }}
        >
          <Trophy size={20} style={{ color: '#22c55e' }} />
          <div>
            <p className="text-sm font-semibold" style={{ color: '#22c55e' }}>
              Best Model: {bestAlgo.label}
            </p>
            <p className="text-xs" style={{ color: 'var(--text-dim)' }}>
              Lowest MAE: {bestAlgo.metrics.mae.toFixed(2)} m &middot;
              RMSE: {bestAlgo.metrics.rmse.toFixed(2)} m &middot;
              {Object.keys(perAlgoMetrics).length} algorithms compared
            </p>
          </div>
        </div>
      )}

      {/* Algorithm tab bar */}
      <div className="flex flex-wrap gap-1.5">
        {algoTabs.map((a) => {
          const active = selectedAlgo === a.key;
          return (
            <button
              key={a.key}
              onClick={() => setSelectedAlgo(a.key)}
              className="px-3 py-1.5 rounded-full border text-xs font-medium whitespace-nowrap transition-all"
              style={{
                borderColor: active ? a.color : 'var(--border)',
                background: active ? `${a.color}18` : 'transparent',
                color: active ? a.color : 'var(--muted)',
              }}
            >
              {a.label}
              {perAlgoMetrics[a.key] && (
                <span className="ml-1 opacity-70">
                  ({perAlgoMetrics[a.key].mae.toFixed(0)}m)
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Error summary for selected algorithm */}
      <ErrorSummary metrics={currentMetrics} />

      {/* Per-year charts */}
      {years.map((year) => {
        if (selectedAlgo === 'all') {
          // Multi-model mode: observed + all algorithm lines
          const observedData = validationData.data
            .filter((d) => d.year === year)
            .reduce((acc, d) => {
              acc[`${d.reach_id}-${d.bank_side}`] = d;
              return acc;
            }, {});

          const leftObs = validationData.data
            .filter((d) => d.year === year && d.bank_side === 'left')
            .sort((a, b) => a.reach_id - b.reach_id);
          const rightObs = validationData.data
            .filter((d) => d.year === year && d.bank_side === 'right')
            .sort((a, b) => a.reach_id - b.reach_id);

          // Build allModelsData for each bank side
          function buildAllModels(bankSide) {
            return ALGORITHMS.map((algo) => {
              const yearData = perAlgoData[algo.key]?.[year] ?? [];
              return {
                key: algo.key,
                label: algo.label,
                color: algo.color,
                points: yearData
                  .filter((d) => d.bank_side === bankSide)
                  .sort((a, b) => a.reach_id - b.reach_id)
                  .map((d) => ({ reach_id: d.reach_id, value: d.predicted })),
              };
            }).filter((m) => m.points.length > 0);
          }

          return (
            <div key={year} className="flex flex-col gap-4">
              <h2 className="text-xl font-bold font-mono" style={{ color: 'var(--accent)' }}>
                {year}
              </h2>
              <div className="grid gap-4 lg:grid-cols-2">
                {leftObs.length > 0 && (
                  <BankComparisonChart
                    year={year}
                    bankSide="left"
                    data={leftObs.map((d) => ({ reach_id: d.reach_id, bank_side: 'left', observed: d.observed, predicted: null }))}
                    allModelsData={buildAllModels('left')}
                  />
                )}
                {rightObs.length > 0 && (
                  <BankComparisonChart
                    year={year}
                    bankSide="right"
                    data={rightObs.map((d) => ({ reach_id: d.reach_id, bank_side: 'right', observed: d.observed, predicted: null }))}
                    allModelsData={buildAllModels('right')}
                  />
                )}
              </div>
            </div>
          );
        }

        // Single algorithm mode
        const yearData = perAlgoData[selectedAlgo]?.[year] ?? [];
        const leftData = yearData
          .filter((d) => d.bank_side === 'left')
          .sort((a, b) => a.reach_id - b.reach_id);
        const rightData = yearData
          .filter((d) => d.bank_side === 'right')
          .sort((a, b) => a.reach_id - b.reach_id);

        const yearMetrics = computeMetrics(yearData);

        return (
          <div key={year} className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold font-mono" style={{ color: 'var(--accent)' }}>
                {year}
              </h2>
              {yearMetrics && (
                <span className="text-xs px-2 py-0.5 rounded-full" style={{
                  background: 'rgba(6,182,212,0.1)',
                  color: 'var(--accent)',
                }}>
                  MAE: {yearMetrics.mae.toFixed(1)}m
                </span>
              )}
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              {leftData.length > 0 && (
                <BankComparisonChart year={year} bankSide="left" data={leftData} />
              )}
              {rightData.length > 0 && (
                <BankComparisonChart year={year} bankSide="right" data={rightData} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
