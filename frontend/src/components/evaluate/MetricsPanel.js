'use client';

function metricColor(key, value) {
  if (value == null) return 'var(--muted)';
  if (key === 'rmse' || key === 'mae') {
    if (value < 100) return '#22c55e';
    if (value < 500) return '#eab308';
    return '#ef4444';
  }
  // nse, kge: higher is better
  if (value >= 0.8) return '#22c55e';
  if (value >= 0.5) return '#eab308';
  return '#ef4444';
}

function MetricCard({ label, value, unit, description, metricKey }) {
  const color = metricColor(metricKey, value);
  return (
    <div
      className="rounded-xl p-5 border flex flex-col gap-2"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
        {label}
      </p>
      <div className="flex items-baseline gap-1">
        <span className="text-3xl font-mono font-bold" style={{ color }}>
          {value != null ? value.toFixed(3) : '—'}
        </span>
        {unit && (
          <span className="text-sm" style={{ color: 'var(--muted)' }}>
            {unit}
          </span>
        )}
      </div>
      <p className="text-xs" style={{ color: 'var(--text-dim)' }}>
        {description}
      </p>
    </div>
  );
}

function CoverageGauge({ value }) {
  const pct = value != null ? Math.round(value * 100) : null;
  const ideal = 80;
  const diff = pct != null ? Math.abs(pct - ideal) : null;
  const color = diff != null && diff <= 5 ? '#22c55e' : diff != null && diff <= 15 ? '#eab308' : '#ef4444';

  return (
    <div
      className="rounded-xl p-5 border"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color: 'var(--muted)' }}>
        80% Quantile Coverage
      </p>
      <div className="flex items-center gap-4">
        <div className="relative w-16 h-16">
          <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
            <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--border)" strokeWidth="3" />
            <circle
              cx="18" cy="18" r="15.9" fill="none"
              stroke={color} strokeWidth="3"
              strokeDasharray={`${pct != null ? pct : 0} 100`}
              strokeLinecap="round"
            />
          </svg>
          <span
            className="absolute inset-0 flex items-center justify-center text-sm font-mono font-bold"
            style={{ color }}
          >
            {pct != null ? `${pct}%` : '—'}
          </span>
        </div>
        <div>
          <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
            Target: 80%
          </p>
          <p className="text-xs" style={{ color: 'var(--text-dim)' }}>
            Fraction of actuals inside the q10–q90 interval. Ideal ≈ 80%.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function MetricsPanel({ data }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <MetricCard
        label="NSE"
        metricKey="nse"
        value={data?.nse}
        description="Nash-Sutcliffe Efficiency. 1.0 = perfect, <0 = worse than mean."
      />
      <MetricCard
        label="KGE"
        metricKey="kge"
        value={data?.kge}
        description="Kling-Gupta Efficiency. Balances correlation, bias, and variability."
      />
      <MetricCard
        label="RMSE"
        metricKey="rmse"
        value={data?.rmse}
        unit="m"
        description="Root Mean Square Error in metres. Lower is better."
      />
      <MetricCard
        label="MAE"
        metricKey="mae"
        value={data?.mae}
        unit="m"
        description="Mean Absolute Error in metres. Lower is better."
      />
      <CoverageGauge value={data?.quantile_coverage_80} />
      <div
        className="rounded-xl p-5 border flex flex-col justify-center gap-1"
        style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <p className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
          Series evaluated
        </p>
        <p className="text-3xl font-mono font-bold" style={{ color: 'var(--text)' }}>
          {data?.n_series ?? '—'}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-dim)' }}>
          100 total (50 reaches × 2 banks)
        </p>
      </div>
    </div>
  );
}
