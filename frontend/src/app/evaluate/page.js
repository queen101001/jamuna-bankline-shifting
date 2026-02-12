'use client';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getEvaluation } from '@/lib/api';
import MetricsPanel from '@/components/evaluate/MetricsPanel';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

export default function EvaluatePage() {
  const [split, setSplit] = useState('test');

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['evaluation', split],
    queryFn: () => getEvaluation(split),
  });

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text)' }}>
            Model Performance
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
            TFT evaluation metrics. Run{' '}
            <code
              className="px-1.5 py-0.5 rounded text-xs"
              style={{ background: 'var(--card)', color: 'var(--accent)' }}
            >
              python -m src.training.evaluate
            </code>{' '}
            to generate results.
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

      {isLoading && <LoadingSpinner label="Loading evaluation metrics…" />}

      {isError && (
        <div
          className="rounded-xl p-6 border text-center"
          style={{ background: 'var(--card)', borderColor: 'rgba(239,68,68,0.3)' }}
        >
          <p className="font-medium mb-1" style={{ color: '#ef4444' }}>
            {error?.status === 404 ? 'Metrics not generated yet' : 'Failed to load metrics'}
          </p>
          <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
            {error?.status === 404
              ? 'Run the evaluation script first: python -m src.training.evaluate --checkpoint <path>'
              : error?.message}
          </p>
        </div>
      )}

      {data && <MetricsPanel data={data} />}
    </div>
  );
}
