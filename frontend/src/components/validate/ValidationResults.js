'use client';
import { useMemo } from 'react';
import { useQueries } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { getPredictionForYear } from '@/lib/api';
import useAppStore from '@/store';
import BankComparisonChart from './BankComparisonChart';
import ErrorSummary from './ErrorSummary';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

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

  const years = validationData?.years ?? [];

  // Fetch predictions for each year in the uploaded data
  const predQueries = useQueries({
    queries: years.map((year) => ({
      queryKey: ['predict-year', year],
      queryFn: () => getPredictionForYear(year),
      staleTime: 60_000,
    })),
  });

  const isLoading = predQueries.some((q) => q.isLoading);
  const hasError = predQueries.some((q) => q.isError);

  // Merge observed + predicted data
  const mergedByYear = useMemo(() => {
    if (!validationData || isLoading) return {};

    const result = {};

    for (let i = 0; i < years.length; i++) {
      const year = years[i];
      const predData = predQueries[i]?.data;
      if (!predData?.predictions) continue;

      // Build lookup: "reachId-bankSide" → q50
      const predMap = {};
      for (const p of predData.predictions) {
        predMap[`${p.reach_id}-${p.bank_side}`] = p.q50;
      }

      // Match observed data for this year
      const yearObs = validationData.data.filter((d) => d.year === year);

      const merged = yearObs
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

      result[year] = merged;
    }

    return result;
  }, [validationData, years, predQueries, isLoading]);

  // Compute global metrics across all years
  const globalMetrics = useMemo(() => {
    const allPairs = Object.values(mergedByYear).flat();
    return computeMetrics(allPairs);
  }, [mergedByYear]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center gap-4 py-12">
        <LoadingSpinner label="Fetching predictions for comparison..." />
      </div>
    );
  }

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

      {hasError && (
        <p className="text-sm" style={{ color: 'var(--erosion)' }}>
          Some predictions could not be fetched. Ensure the backend is running and the model is loaded.
        </p>
      )}

      {/* Global error summary */}
      <ErrorSummary metrics={globalMetrics} />

      {/* Per-year charts */}
      {years.map((year) => {
        const yearData = mergedByYear[year] || [];
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
