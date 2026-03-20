'use client';
import { useState } from 'react';
import ComparisonChart from '@/components/compare/ComparisonChart';
import MetricsTable from '@/components/compare/MetricsTable';
import InfoButton from '@/components/ui/InfoButton';

export default function ComparePage() {
  const [reachId, setReachId] = useState(1);
  const [bankSide, setBankSide] = useState('left');

  return (
    <div className="min-h-screen px-4 py-8 max-w-[1600px] mx-auto">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text)' }}>
          Model Comparison
        </h1>
        <p className="text-sm max-w-xl mx-auto" style={{ color: 'var(--text-dim)' }}>
          Compare forecast trajectories and performance metrics across all 11 models (TFT + 10 baselines).
        </p>
      </div>

      {/* Reach / Bank selectors */}
      <div className="flex flex-wrap justify-center items-center gap-4 mb-8">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium uppercase tracking-widest" style={{ color: 'var(--muted)' }}>
            Reach
          </label>
          <select
            value={reachId}
            onChange={(e) => setReachId(parseInt(e.target.value, 10))}
            className="rounded-lg border px-3 py-1.5 text-sm bg-transparent outline-none"
            style={{ borderColor: 'var(--border)', color: 'var(--text)', background: 'var(--card)' }}
          >
            {Array.from({ length: 50 }, (_, i) => i + 1).map((id) => (
              <option key={id} value={id} style={{ background: 'var(--bg)', color: 'var(--text)' }}>
                R{String(id).padStart(2, '0')}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-medium uppercase tracking-widest" style={{ color: 'var(--muted)' }}>
            Bank
          </label>
          <select
            value={bankSide}
            onChange={(e) => setBankSide(e.target.value)}
            className="rounded-lg border px-3 py-1.5 text-sm bg-transparent outline-none"
            style={{ borderColor: 'var(--border)', color: 'var(--text)', background: 'var(--card)' }}
          >
            <option value="left" style={{ background: 'var(--bg)', color: 'var(--text)' }}>Left Bank</option>
            <option value="right" style={{ background: 'var(--bg)', color: 'var(--text)' }}>Right Bank</option>
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="mb-8">
        <ComparisonChart reachId={reachId} bankSide={bankSide} />
      </div>

      {/* Metrics table */}
      <MetricsTable />

      <InfoButton pageId="compare" />
    </div>
  );
}
