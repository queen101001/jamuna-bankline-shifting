'use client';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Shield, Activity } from 'lucide-react';
import { getChangepoints } from '@/lib/api';
import ChangepointTable from '@/components/anomaly/ChangepointTable';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

export default function AnomalyPage() {
  const [protectedOnly, setProtectedOnly] = useState(false);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['changepoints', protectedOnly],
    queryFn: () => getChangepoints(protectedOnly),
  });

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text)' }}>
          Anomaly Detection
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
          PELT changepoint detection across all 100 series. Protection signature = variance reduction ≥ 70%.
        </p>
      </div>

      {/* Stats + filter */}
      <div className="flex items-center justify-between mb-6 gap-4 flex-wrap">
        {data && (
          <div className="flex items-center gap-6">
            <div
              className="flex items-center gap-2 px-4 py-2 rounded-lg border"
              style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
            >
              <Activity size={16} style={{ color: 'var(--accent)' }} />
              <span className="text-sm" style={{ color: 'var(--text-dim)' }}>
                Total changepoints:
              </span>
              <span className="font-mono font-bold" style={{ color: 'var(--text)' }}>
                {data.total_changepoints}
              </span>
            </div>
            <div
              className="flex items-center gap-2 px-4 py-2 rounded-lg border"
              style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
            >
              <Shield size={16} style={{ color: '#22c55e' }} />
              <span className="text-sm" style={{ color: 'var(--text-dim)' }}>
                Protected reaches:
              </span>
              <span className="font-mono font-bold" style={{ color: '#22c55e' }}>
                {data.potentially_protected_reaches}
              </span>
            </div>
          </div>
        )}

        {/* Filter toggle */}
        <div
          className="flex items-center rounded-lg overflow-hidden border text-sm"
          style={{ borderColor: 'var(--border)' }}
        >
          <button
            onClick={() => setProtectedOnly(false)}
            className="px-4 py-2 transition-colors"
            style={{
              background: !protectedOnly ? 'rgba(6,182,212,0.12)' : 'var(--card)',
              color: !protectedOnly ? 'var(--accent)' : 'var(--text-dim)',
            }}
          >
            All
          </button>
          <button
            onClick={() => setProtectedOnly(true)}
            className="px-4 py-2 transition-colors"
            style={{
              background: protectedOnly ? 'rgba(34,197,94,0.12)' : 'var(--card)',
              color: protectedOnly ? '#22c55e' : 'var(--text-dim)',
            }}
          >
            🛡 Protected only
          </button>
        </div>
      </div>

      {isLoading && <LoadingSpinner label="Loading changepoints…" />}

      {isError && (
        <p className="text-center py-12" style={{ color: '#ef4444' }}>
          {error?.message || 'Failed to load changepoints'}
        </p>
      )}

      {data && <ChangepointTable changepoints={data.changepoints} />}
    </div>
  );
}
