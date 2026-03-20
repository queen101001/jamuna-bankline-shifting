'use client';
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { getSeriesHistory, postPredictBaseline } from '@/lib/api';
import { ALGORITHMS } from '@/lib/algorithms';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

const LAST_OBS_YEAR = 2020;

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg px-3 py-2 text-xs border max-w-xs"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <p className="font-mono font-bold mb-1" style={{ color: 'var(--accent)' }}>
        {label}
      </p>
      {payload
        .filter((p) => p.value != null)
        .map((p) => (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: {typeof p.value === 'number' ? `${p.value.toFixed(1)} m` : p.value}
          </p>
        ))}
    </div>
  );
};

export default function ComparisonChart({ reachId = 1, bankSide = 'left' }) {
  const [enabled, setEnabled] = useState(
    () => new Set(['tft', 'arima', 'random_forest', 'xgboost'])
  );

  // Fetch historical + TFT forecast
  const { data: seriesData, isLoading: seriesLoading } = useQuery({
    queryKey: ['series', reachId, bankSide],
    queryFn: () => getSeriesHistory(reachId, bankSide, true),
  });

  // Fetch each enabled baseline
  const baselineQueries = ALGORITHMS.filter((a) => a.key !== 'tft' && enabled.has(a.key));
  const baselineResults = useQuery({
    queryKey: ['compareBaselines', reachId, bankSide, [...enabled].sort().join(',')],
    queryFn: async () => {
      const results = {};
      const promises = baselineQueries.map(async (a) => {
        try {
          const res = await postPredictBaseline({
            reach_id: reachId,
            bank_side: bankSide,
            model_name: a.key,
            n_steps: 5,
          });
          results[a.key] = res.forecasts;
        } catch {
          results[a.key] = null;
        }
      });
      await Promise.all(promises);
      return results;
    },
    staleTime: 60_000,
  });

  const chartData = useMemo(() => {
    if (!seriesData) return [];

    const data = (seriesData.observations || []).map((o) => ({
      year: o.year,
      actual: o.bank_distance,
    }));

    // Add TFT forecast
    if (seriesData.latest_forecast) {
      for (const f of seriesData.latest_forecast) {
        const existing = data.find((d) => d.year === f.estimated_year);
        if (existing) {
          existing.tft = f.q50;
        } else {
          data.push({ year: f.estimated_year, actual: null, tft: f.q50 });
        }
      }
    }

    // Add baseline forecasts
    if (baselineResults.data) {
      for (const [algo, forecasts] of Object.entries(baselineResults.data)) {
        if (!forecasts) continue;
        for (const f of forecasts) {
          const existing = data.find((d) => d.year === f.estimated_year);
          if (existing) {
            existing[algo] = f.q50;
          } else {
            const row = { year: f.estimated_year, actual: null };
            row[algo] = f.q50;
            data.push(row);
          }
        }
      }
    }

    data.sort((a, b) => a.year - b.year);
    return data;
  }, [seriesData, baselineResults.data]);

  function toggleAlgo(key) {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  if (seriesLoading) return <LoadingSpinner label="Loading series data…" />;

  return (
    <div
      className="rounded-xl p-5 border"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text)' }}>
        Multi-Model Forecast Comparison — Reach {reachId} ({bankSide} bank)
      </h3>

      {/* Toggle buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        {ALGORITHMS.map((a) => (
          <button
            key={a.key}
            onClick={() => toggleAlgo(a.key)}
            className="text-xs px-2.5 py-1 rounded-full border transition-all"
            style={{
              borderColor: enabled.has(a.key) ? a.color : 'var(--border)',
              background: enabled.has(a.key) ? `${a.color}18` : 'transparent',
              color: enabled.has(a.key) ? a.color : 'var(--muted)',
            }}
          >
            {a.label}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
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

          {/* Historical line */}
          <Line
            type="monotone"
            dataKey="actual"
            name="Observed"
            stroke="#f1f5f9"
            strokeWidth={2}
            dot={{ r: 3, fill: '#f1f5f9', strokeWidth: 0 }}
            connectNulls={false}
          />

          {/* Model forecast lines */}
          {ALGORITHMS.filter((a) => enabled.has(a.key)).map((a) => (
            <Line
              key={a.key}
              type="monotone"
              dataKey={a.key}
              name={a.label}
              stroke={a.color}
              strokeWidth={a.key === 'tft' ? 2.5 : 1.5}
              strokeDasharray={a.key === 'tft' ? undefined : '5 3'}
              dot={{ r: 2.5, fill: a.color, strokeWidth: 0 }}
              connectNulls={false}
            />
          ))}

          <ReferenceLine
            x={LAST_OBS_YEAR}
            stroke="rgba(255,255,255,0.2)"
            strokeDasharray="4 2"
            label={{ value: 'Last obs.', fill: 'var(--muted)', fontSize: 10, position: 'top' }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
