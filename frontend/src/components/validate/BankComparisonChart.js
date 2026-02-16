'use client';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div
      className="rounded-lg border px-3 py-2 text-xs shadow-lg"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="font-semibold mb-1" style={{ color: 'var(--text)' }}>
        Reach {label}
      </p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value != null ? p.value.toFixed(2) : '—'} m
        </p>
      ))}
      {payload.length >= 2 && payload[0].value != null && payload[1].value != null && (
        <p className="mt-1 pt-1 border-t" style={{ color: 'var(--muted)', borderColor: 'var(--border)' }}>
          Error: {(payload[0].value - payload[1].value).toFixed(2)} m
        </p>
      )}
    </div>
  );
}

export default function BankComparisonChart({ year, bankSide, data }) {
  const label = bankSide === 'left' ? 'Left Bank' : 'Right Bank';

  return (
    <div
      className="rounded-xl border p-4"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="text-sm font-semibold mb-3" style={{ color: 'var(--text)' }}>
        {label} — {year}
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="reach_id"
            label={{ value: 'Reach Point', position: 'insideBottom', offset: -10, fill: 'var(--muted)', fontSize: 12 }}
            tick={{ fill: 'var(--muted)', fontSize: 11 }}
            stroke="var(--border)"
          />
          <YAxis
            label={{ value: 'Distance (m)', angle: -90, position: 'insideLeft', offset: -5, fill: 'var(--muted)', fontSize: 12 }}
            tick={{ fill: 'var(--muted)', fontSize: 11 }}
            stroke="var(--border)"
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: 'var(--text-dim)' }}
          />
          <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="observed"
            name="Observed"
            stroke="#06b6d4"
            strokeWidth={2}
            dot={{ r: 2, fill: '#06b6d4' }}
            activeDot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            name="Predicted"
            stroke="#94a3b8"
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={{ r: 2, fill: '#94a3b8' }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
