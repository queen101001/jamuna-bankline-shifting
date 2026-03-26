'use client';
import { useQuery } from '@tanstack/react-query';
import { getPredictionForYear, getBaselineYearPrediction, getHistoricalYear } from '@/lib/api';
import useAppStore from '@/store';
import ReachCard from './ReachCard';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

export default function ReachGrid() {
  const confirmedYear = useAppStore((s) => s.confirmedYear);
  const activeAlgorithm = useAppStore((s) => s.activeAlgorithm);

  const isHistorical = confirmedYear <= 2020;

  const { data, isLoading, isFetching, isError, error } = useQuery({
    queryKey: ['yearPrediction', confirmedYear, isHistorical ? 'historical' : activeAlgorithm],
    queryFn: () => {
      if (isHistorical) return getHistoricalYear(confirmedYear);
      if (activeAlgorithm === 'tft') return getPredictionForYear(confirmedYear);
      return getBaselineYearPrediction(confirmedYear, activeAlgorithm);
    },
  });

  if (isLoading || isFetching) return <LoadingSpinner label={`Loading predictions for ${confirmedYear}…`} />;

  if (isError) {
    return (
      <div
        className="rounded-xl p-6 text-center border"
        style={{ background: 'var(--card)', borderColor: 'rgba(239,68,68,0.3)' }}
      >
        <p className="font-medium mb-1" style={{ color: '#ef4444' }}>
          Failed to load predictions
        </p>
        <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
          {error?.message || 'API unavailable. Is the backend running?'}
        </p>
      </div>
    );
  }

  if (!data) return null;

  // Build lookup: reach_id → { left, right }
  const byReach = {};
  for (const p of data.predictions) {
    if (!byReach[p.reach_id]) byReach[p.reach_id] = {};
    byReach[p.reach_id][p.bank_side] = p;
  }

  // Anomalous reach IDs from anomaly_info if present (not in year endpoint, so skip)
  const anomalousIds = new Set();

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
          {data.n_points} predictions · {data.n_points / 2} reaches
        </p>
        <p className="text-sm font-mono" style={{ color: 'var(--muted)' }}>
          {data.n_steps === 0
            ? 'Observed data'
            : `${data.n_steps} step${data.n_steps !== 1 ? 's' : ''} from ${data.last_observed_year}`
          }
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
        {Array.from({ length: 50 }, (_, i) => i + 1).map((reachId) => (
          <ReachCard
            key={reachId}
            reach_id={reachId}
            leftForecast={byReach[reachId]?.left}
            rightForecast={byReach[reachId]?.right}
            isAnomalous={anomalousIds.has(reachId)}
          />
        ))}
      </div>
    </div>
  );
}
