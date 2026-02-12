'use client';

/**
 * Sign convention:
 *   Left bank:  q50 > 0 → Erosion (red),  q50 < 0 → Deposition (green)
 *   Right bank: q50 < 0 → Erosion (red),  q50 > 0 → Deposition (green)
 */
function isErosion(bankSide, q50) {
  if (bankSide === 'left') return q50 > 0;
  return q50 < 0;
}

function erosionColor(magnitude) {
  // Scale from orange (low) to red (high)
  const clamped = Math.min(magnitude / 5000, 1);
  const r = Math.round(239 + (255 - 239) * 0 + 0);
  const g = Math.round(68 * (1 - clamped) + 68);
  const b = 68;
  if (clamped > 0.6) return '#ef4444';
  if (clamped > 0.3) return '#f97316';
  return '#fb923c';
}

export default function ErosionBar({ bankSide, q50, q10, q90, maxValue = 10000 }) {
  const pct = Math.min((Math.abs(q50) / maxValue) * 100, 100);
  const eroding = isErosion(bankSide, q50);
  const color = eroding ? erosionColor(Math.abs(q50)) : 'var(--deposition)';
  const label = eroding ? 'Erosion' : 'Deposition';

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium capitalize" style={{ color: 'var(--text-dim)' }}>
          {bankSide} bank
        </span>
        <span className="text-xs font-mono" style={{ color }}>
          {q50 >= 0 ? '+' : ''}{q50.toFixed(0)} m
        </span>
      </div>
      <div
        className="relative h-2 rounded-full overflow-hidden"
        style={{ background: 'var(--border)' }}
        title={q10 != null ? `q10: ${q10.toFixed(0)} m · q50: ${q50.toFixed(0)} m · q90: ${q90.toFixed(0)} m` : `${q50.toFixed(0)} m`}
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className="flex items-center gap-1 mt-0.5">
        <span
          className="inline-block w-1.5 h-1.5 rounded-full"
          style={{ background: color }}
        />
        <span className="text-xs" style={{ color: 'var(--muted)' }}>
          {label}
        </span>
      </div>
    </div>
  );
}
