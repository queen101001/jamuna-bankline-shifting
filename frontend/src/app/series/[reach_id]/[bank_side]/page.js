'use client';
import { use } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getSeriesHistory, getChangepoints } from '@/lib/api';
import SeriesHeader from '@/components/series/SeriesHeader';
import HistoryChart from '@/components/series/HistoryChart';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import QuantileBadge from '@/components/ui/QuantileBadge';

export default function SeriesPage({ params }) {
  const { reach_id, bank_side } = use(params);
  const reachId = parseInt(reach_id, 10);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['series', reachId, bank_side],
    queryFn: () => getSeriesHistory(reachId, bank_side, true),
  });

  const { data: cpData } = useQuery({
    queryKey: ['changepoints'],
    queryFn: () => getChangepoints(false),
    staleTime: 300_000,
  });

  const changepointYears = cpData?.changepoints
    ?.filter((c) => c.reach_id === reachId && c.bank_side === bank_side)
    ?.map((c) => c.changepoint_year) ?? [];

  if (isLoading) return <LoadingSpinner label="Loading series data…" />;

  if (isError) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <p style={{ color: '#ef4444' }}>{error?.message || 'Failed to load series'}</p>
      </div>
    );
  }

  const forecast = data?.latest_forecast;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <SeriesHeader reachId={reachId} bankSide={bank_side} seriesData={data} />

      <HistoryChart
        observations={data?.observations ?? []}
        forecast={forecast}
        changepointYears={changepointYears}
      />

      {/* Forecast table */}
      {forecast && forecast.length > 0 && (
        <div
          className="mt-6 rounded-xl p-5 border"
          style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
        >
          <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text)' }}>
            5-step forecast
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Step', 'Year', 'q10 (low)', 'q50 (median)', 'q90 (high)'].map((h) => (
                    <th key={h} className="text-left pb-2 pr-4 font-medium" style={{ color: 'var(--text-dim)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {forecast.map((f) => (
                  <tr key={f.step} style={{ borderBottom: '1px solid rgba(51,65,85,0.4)' }}>
                    <td className="py-2 pr-4 font-mono" style={{ color: 'var(--muted)' }}>
                      +{f.step}
                    </td>
                    <td className="py-2 pr-4 font-mono font-medium" style={{ color: 'var(--text)' }}>
                      {f.estimated_year}
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: 'var(--muted)' }}>
                      {f.q10 != null ? `${f.q10.toFixed(1)} m` : '—'}
                    </td>
                    <td className="py-2 pr-4">
                      <QuantileBadge q50={f.q50} q10={f.q10} q90={f.q90} />
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: 'var(--muted)' }}>
                      {f.q90 != null ? `${f.q90.toFixed(1)} m` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Changepoint info */}
      {changepointYears.length > 0 && (
        <div
          className="mt-4 rounded-xl p-4 border flex items-start gap-3"
          style={{ background: 'rgba(251,146,60,0.06)', borderColor: 'rgba(251,146,60,0.3)' }}
        >
          <span style={{ color: '#fb923c', fontSize: 18 }}>⚑</span>
          <div>
            <p className="text-sm font-medium mb-0.5" style={{ color: '#fb923c' }}>
              Structural changepoints detected
            </p>
            <p className="text-xs" style={{ color: 'var(--text-dim)' }}>
              Years: {changepointYears.join(', ')} — variance reduction ≥70% suggests bank protection works.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
