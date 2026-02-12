import { ArrowLeft, AlertTriangle, TrendingDown, TrendingUp } from 'lucide-react';
import Link from 'next/link';

function isErosion(bankSide, value) {
  if (bankSide === 'left') return value > 0;
  return value < 0;
}

export default function SeriesHeader({ reachId, bankSide, seriesData }) {
  const lastObs = seriesData?.observations?.at(-1);
  const eroding = lastObs ? isErosion(bankSide, lastObs.bank_distance) : null;

  return (
    <div className="mb-8">
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-sm mb-4 no-underline transition-colors hover:opacity-80"
        style={{ color: 'var(--text-dim)' }}
      >
        <ArrowLeft size={14} />
        Back to Dashboard
      </Link>

      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <span
              className="font-mono font-bold text-2xl"
              style={{ color: 'var(--accent)' }}
            >
              R{String(reachId).padStart(2, '0')}
            </span>
            <span
              className="px-2 py-0.5 rounded text-sm font-medium capitalize"
              style={{ background: 'var(--card)', color: 'var(--text-dim)', border: '1px solid var(--border)' }}
            >
              {bankSide} bank
            </span>
            {seriesData?.latest_forecast && (
              <span
                className="flex items-center gap-1 text-sm"
                style={{ color: eroding ? '#ef4444' : '#22c55e' }}
              >
                {eroding ? <TrendingDown size={15} /> : <TrendingUp size={15} />}
                {eroding ? 'Eroding' : 'Depositing'}
              </span>
            )}
          </div>

          {lastObs && (
            <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
              Last observed: <span className="font-mono" style={{ color: 'var(--text)' }}>{lastObs.year}</span>
              {' · '}
              <span className="font-mono" style={{ color: 'var(--text)' }}>
                {lastObs.bank_distance >= 0 ? '+' : ''}{lastObs.bank_distance.toFixed(1)} m
              </span>
            </p>
          )}
        </div>

        {seriesData?.series_id && (
          <span className="text-xs font-mono" style={{ color: 'var(--muted)' }}>
            {seriesData.series_id}
          </span>
        )}
      </div>
    </div>
  );
}
