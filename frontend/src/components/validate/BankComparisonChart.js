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
      className="rounded-lg border px-3 py-2 text-xs shadow-lg max-w-xs"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="font-semibold mb-1" style={{ color: 'var(--text)' }}>
        Reach {label}
      </p>
      {payload
        .filter((p) => p.value != null)
        .map((p) => (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: {p.value.toFixed(2)} m
          </p>
        ))}
    </div>
  );
}

/**
 * BankComparisonChart
 *
 * Single-model: data=[{reach_id, observed, predicted}]
 * Multi-model:  data=[{reach_id, observed}] + allModelsData=[{key, label, color, points:[{reach_id, value}]}]
 */
export default function BankComparisonChart({ year, bankSide, data, allModelsData }) {
  const label = bankSide === 'left' ? 'Left Bank' : 'Right Bank';
  const isMultiModel = allModelsData && allModelsData.length > 0;

  let chartData;
  if (isMultiModel) {
    const reachIds = [...new Set(data.map((d) => d.reach_id))].sort((a, b) => a - b);
    chartData = reachIds.map((rid) => {
      const row = { reach_id: rid };
      const obs = data.find((d) => d.reach_id === rid);
      row.observed = obs?.observed ?? null;
      for (const model of allModelsData) {
        const pt = model.points.find((p) => p.reach_id === rid);
        row[model.key] = pt?.value ?? null;
      }
      return row;
    });
  } else {
    chartData = data;
  }

  return (
    <div
      className="rounded-xl border p-4"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="text-sm font-semibold mb-3" style={{ color: 'var(--text)' }}>
        {label} — {year}
      </p>
      <ResponsiveContainer width="100%" height={isMultiModel ? 380 : 320}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
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
          <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-dim)' }} />
          <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="4 4" />

          {/* Observed line */}
          <Line
            type="monotone"
            dataKey="observed"
            name="Observed"
            stroke="#06b6d4"
            strokeWidth={2}
            dot={{ r: 2, fill: '#06b6d4' }}
            activeDot={{ r: 4 }}
          />

          {isMultiModel
            ? allModelsData.map((model) => (
                <Line
                  key={model.key}
                  type="monotone"
                  dataKey={model.key}
                  name={model.label}
                  stroke={model.color}
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={{ r: 1.5, fill: model.color }}
                  activeDot={{ r: 3 }}
                  connectNulls={false}
                />
              ))
            : (
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
            )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
