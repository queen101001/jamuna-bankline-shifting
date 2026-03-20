'use client';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getEvaluation } from '@/lib/api';
import MetricsPanel from '@/components/evaluate/MetricsPanel';
import MetricsTable from '@/components/compare/MetricsTable';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import InfoButton from '@/components/ui/InfoButton';

export default function EvaluatePage() {
  const [split, setSplit] = useState('test');

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['evaluation', split],
    queryFn: () => getEvaluation(split),
  });

  function errorMessage() {
    if (!error) return 'Failed to load metrics';
    if (error.status === 404) return 'Run the evaluation script first: python -m src.training.evaluate --checkpoint <path>';
    return error.message || 'Failed to load metrics';
  }

  function errorTitle() {
    if (!error) return 'Failed to load metrics';
    if (error.status === 404) return 'Metrics not generated yet';
    return 'Failed to load metrics';
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text)' }}>
            Model Performance
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
            Side-by-side comparison of all 11 algorithms. TFT detail metrics shown below for the selected split.
          </p>
        </div>

        {/* Split toggle */}
        <div
          className="flex items-center rounded-lg overflow-hidden border text-sm"
          style={{ borderColor: 'var(--border)' }}
        >
          {['val', 'test'].map((s) => (
            <button
              key={s}
              onClick={() => setSplit(s)}
              className="px-4 py-2 transition-colors capitalize"
              style={{
                background: split === s ? 'rgba(6,182,212,0.12)' : 'var(--card)',
                color: split === s ? 'var(--accent)' : 'var(--text-dim)',
              }}
            >
              {s} split
            </button>
          ))}
        </div>
      </div>

      {/* All-algorithm comparison table */}
      <div className="mb-8">
        <h2 className="text-sm font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--muted)' }}>
          All Algorithms — Test Split
        </h2>
        <MetricsTable />
      </div>

      {/* TFT detail metrics for selected split */}
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--muted)' }}>
          TFT Detail — {split === 'val' ? 'Validation' : 'Test'} Split
        </h2>

        {isLoading && <LoadingSpinner label="Loading evaluation metrics…" />}

        {isError && (
          <div
            className="rounded-xl p-6 border text-center"
            style={{ background: 'var(--card)', borderColor: 'rgba(239,68,68,0.3)' }}
          >
            <p className="font-medium mb-1" style={{ color: '#ef4444' }}>
              {errorTitle()}
            </p>
            <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
              {errorMessage()}
            </p>
          </div>
        )}

        {data && <MetricsPanel data={data} />}
      </div>

      <InfoButton pageId="evaluate" />
    </div>
  );
}
