'use client';

function metricColor(value, thresholds) {
  if (value < thresholds[0]) return 'var(--deposition)';
  if (value < thresholds[1]) return '#eab308';
  return 'var(--erosion)';
}

function MetricCard({ label, value, unit, thresholds }) {
  const color = thresholds ? metricColor(value, thresholds) : 'var(--accent)';
  return (
    <div
      className="rounded-xl border p-4 flex flex-col gap-1"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--muted)' }}>
        {label}
      </p>
      <p className="text-2xl font-bold font-mono" style={{ color }}>
        {typeof value === 'number' ? value.toFixed(2) : '—'}
        {unit && <span className="text-sm font-normal ml-1" style={{ color: 'var(--text-dim)' }}>{unit}</span>}
      </p>
    </div>
  );
}

export default function ErrorSummary({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard label="MAE" value={metrics.mae} unit="m" thresholds={[100, 500]} />
      <MetricCard label="RMSE" value={metrics.rmse} unit="m" thresholds={[100, 500]} />
      <MetricCard label="Max Error" value={metrics.maxError} unit="m" thresholds={[200, 1000]} />
      <MetricCard label="Data Points" value={metrics.count} />
    </div>
  );
}
