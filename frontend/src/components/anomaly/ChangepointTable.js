'use client';
import { useState } from 'react';
import Link from 'next/link';
import { ArrowUpDown } from 'lucide-react';

function SortButton({ field, sortField, sortDir, onSort }) {
  const active = sortField === field;
  return (
    <button
      onClick={() => onSort(field)}
      className="inline-flex items-center gap-1 hover:opacity-80 transition-opacity"
      style={{ color: active ? 'var(--accent)' : 'var(--text-dim)' }}
    >
      {field}
      <ArrowUpDown size={12} />
    </button>
  );
}

export default function ChangepointTable({ changepoints }) {
  const [sortField, setSortField] = useState('variance_reduction');
  const [sortDir, setSortDir] = useState('desc');

  function handleSort(field) {
    if (sortField === field) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortField(field); setSortDir('desc'); }
  }

  const sorted = [...changepoints].sort((a, b) => {
    const av = a[sortField] ?? 0;
    const bv = b[sortField] ?? 0;
    return sortDir === 'asc' ? av - bv : bv - av;
  });

  if (sorted.length === 0) {
    return (
      <div className="text-center py-16" style={{ color: 'var(--text-dim)' }}>
        No changepoints match the current filter.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border" style={{ borderColor: 'var(--border)' }}>
      <table className="w-full text-sm">
        <thead style={{ background: 'var(--surface)' }}>
          <tr>
            {[
              { key: 'series_id', label: 'Series' },
              { key: 'changepoint_year', label: 'Year' },
              { key: 'variance_before', label: 'Var. before' },
              { key: 'variance_after', label: 'Var. after' },
              { key: 'variance_reduction', label: 'Reduction %' },
            ].map(({ key, label }) => (
              <th key={key} className="text-left px-4 py-3 font-medium">
                <SortButton
                  field={key}
                  sortField={sortField}
                  sortDir={sortDir}
                  onSort={handleSort}
                />
              </th>
            ))}
            <th className="text-left px-4 py-3 font-medium" style={{ color: 'var(--text-dim)' }}>
              Status
            </th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((cp, i) => (
            <tr
              key={`${cp.series_id}-${cp.changepoint_year}`}
              style={{
                background: i % 2 === 0 ? 'var(--card)' : 'transparent',
                borderTop: '1px solid rgba(51,65,85,0.4)',
              }}
            >
              <td className="px-4 py-3 font-mono" style={{ color: 'var(--accent)' }}>
                {cp.series_id}
              </td>
              <td className="px-4 py-3 font-mono font-medium" style={{ color: 'var(--text)' }}>
                {cp.changepoint_year}
              </td>
              <td className="px-4 py-3 font-mono text-xs" style={{ color: 'var(--text-dim)' }}>
                {cp.variance_before?.toFixed(0)}
              </td>
              <td className="px-4 py-3 font-mono text-xs" style={{ color: 'var(--text-dim)' }}>
                {cp.variance_after?.toFixed(0)}
              </td>
              <td className="px-4 py-3">
                <span
                  className="font-mono text-xs font-semibold"
                  style={{ color: cp.variance_reduction >= 0.7 ? '#22c55e' : 'var(--text-dim)' }}
                >
                  {((cp.variance_reduction ?? 0) * 100).toFixed(1)}%
                </span>
              </td>
              <td className="px-4 py-3">
                {cp.is_protection_signature ? (
                  <span
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
                    style={{
                      background: 'rgba(34,197,94,0.1)',
                      color: '#22c55e',
                      border: '1px solid rgba(34,197,94,0.25)',
                    }}
                  >
                    🛡 Protected
                  </span>
                ) : (
                  <span className="text-xs" style={{ color: 'var(--muted)' }}>
                    Structural shift
                  </span>
                )}
              </td>
              <td className="px-4 py-3">
                <Link
                  href={`/series/${cp.reach_id}/${cp.bank_side}`}
                  className="text-xs no-underline transition-opacity hover:opacity-80"
                  style={{ color: 'var(--accent)' }}
                >
                  View →
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
