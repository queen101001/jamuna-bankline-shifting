'use client';
import { useQuery } from '@tanstack/react-query';
import { getEvaluateCompare } from '@/lib/api';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

function fmt(v) {
  if (v == null) return '—';
  return v.toFixed(4);
}

export default function MetricsTable() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['evaluateCompare'],
    queryFn: getEvaluateCompare,
    staleTime: 120_000,
  });

  if (isLoading) return <LoadingSpinner label="Loading comparison metrics…" />;
  if (isError) {
    return (
      <p className="text-sm" style={{ color: '#ef4444' }}>
        {error?.message || 'Failed to load comparison'}
      </p>
    );
  }
  if (!data?.models?.length) {
    return <p className="text-sm" style={{ color: 'var(--muted)' }}>No models to compare.</p>;
  }

  const metrics = ['nse', 'rmse', 'mae', 'kge'];
  // Find best value per metric (highest NSE/KGE, lowest RMSE/MAE)
  const best = {};
  for (const m of metrics) {
    const values = data.models.map((mod) => mod[m]).filter((v) => v != null);
    if (!values.length) continue;
    best[m] = (m === 'rmse' || m === 'mae') ? Math.min(...values) : Math.max(...values);
  }

  return (
    <div
      className="rounded-xl p-5 border overflow-x-auto"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text)' }}>
        Model Performance Comparison (Test Split)
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Model', 'NSE', 'RMSE', 'MAE', 'KGE', 'Series'].map((h) => (
              <th
                key={h}
                className="text-left pb-2 pr-4 font-medium"
                style={{ color: 'var(--text-dim)' }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.models.map((mod) => (
            <tr key={mod.model_name} style={{ borderBottom: '1px solid rgba(51,65,85,0.4)' }}>
              <td className="py-2 pr-4 font-medium" style={{ color: 'var(--text)' }}>
                {mod.model_name.toUpperCase()}
              </td>
              {metrics.map((m) => {
                const isBest = mod[m] != null && mod[m] === best[m];
                return (
                  <td
                    key={m}
                    className="py-2 pr-4 font-mono"
                    style={{ color: isBest ? 'var(--accent)' : 'var(--muted)', fontWeight: isBest ? 700 : 400 }}
                  >
                    {fmt(mod[m])}
                  </td>
                );
              })}
              <td className="py-2 pr-4 font-mono" style={{ color: 'var(--muted)' }}>
                {mod.n_series}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
