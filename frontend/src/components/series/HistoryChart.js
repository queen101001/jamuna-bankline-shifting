'use client';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const LAST_OBS_YEAR = 2020;

function buildChartData(observations, forecast) {
  const data = observations.map((o) => ({
    year: o.year,
    actual: o.bank_distance,
    q50: null,
    q10: null,
    q90: null,
    band: null,
  }));

  if (forecast) {
    for (const f of forecast) {
      data.push({
        year: f.estimated_year,
        actual: null,
        q50: f.q50,
        q10: f.q10,
        q90: f.q90,
        band: f.q10 != null && f.q90 != null ? [f.q10, f.q90] : null,
      });
    }
  }

  return data;
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg px-3 py-2 text-xs border"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="font-mono font-bold mb-1" style={{ color: 'var(--accent)' }}>
        {label}
      </p>
      {payload.map((p) =>
        p.value != null ? (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: {typeof p.value === 'number' ? `${p.value.toFixed(1)} m` : p.value}
          </p>
        ) : null
      )}
    </div>
  );
};

export default function HistoryChart({ observations, forecast, changepointYears = [] }) {
  const data = buildChartData(observations, forecast);

  return (
    <div
      className="rounded-xl p-5 border"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text)' }}>
        Historical observations + forecast
      </h3>
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(51,65,85,0.6)" />
          <XAxis
            dataKey="year"
            tick={{ fill: 'var(--text-dim)', fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: 'var(--border)' }}
          />
          <YAxis
            tick={{ fill: 'var(--text-dim)', fontSize: 11, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={{ stroke: 'var(--border)' }}
            tickFormatter={(v) => `${v.toFixed(0)}`}
            width={72}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: 'var(--text-dim)', paddingTop: 8 }}
          />

          {/* 80% confidence band */}
          <Area
            type="monotone"
            dataKey="q90"
            name="q90 (upper)"
            stroke="none"
            fill="rgba(6,182,212,0.08)"
            legendType="none"
          />
          <Area
            type="monotone"
            dataKey="q10"
            name="q10 (lower)"
            stroke="none"
            fill="var(--bg)"
            legendType="none"
          />

          {/* Historical line */}
          <Line
            type="monotone"
            dataKey="actual"
            name="Observed"
            stroke="#f1f5f9"
            strokeWidth={2}
            dot={{ r: 3, fill: '#f1f5f9', strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls={false}
          />

          {/* Forecast median */}
          <Line
            type="monotone"
            dataKey="q50"
            name="Forecast (q50)"
            stroke="var(--accent)"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={{ r: 3, fill: 'var(--accent)', strokeWidth: 0 }}
            activeDot={{ r: 5 }}
            connectNulls={false}
          />

          {/* Last observed year reference */}
          <ReferenceLine
            x={LAST_OBS_YEAR}
            stroke="rgba(255,255,255,0.2)"
            strokeDasharray="4 2"
            label={{ value: 'Last obs.', fill: 'var(--muted)', fontSize: 10, position: 'top' }}
          />

          {/* Changepoint markers */}
          {changepointYears.map((yr) => (
            <ReferenceLine
              key={yr}
              x={yr}
              stroke="#fb923c"
              strokeDasharray="3 2"
              label={{ value: '⚑', fill: '#fb923c', fontSize: 10, position: 'top' }}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
