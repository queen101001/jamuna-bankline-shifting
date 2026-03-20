'use client';

export default function MetricScale({ label, ranges, unit }) {
  // ranges: [{from, to, color, label}] sorted left to right
  return (
    <div className="my-2">
      <p className="text-[10px] font-medium mb-1.5" style={{ color: 'var(--text)' }}>
        {label} {unit && <span style={{ color: 'var(--muted)' }}>({unit})</span>}
      </p>
      <div className="flex rounded-md overflow-hidden h-5">
        {ranges.map((r, i) => (
          <div
            key={i}
            className="flex items-center justify-center text-[9px] font-medium"
            style={{
              flex: 1,
              background: r.color,
              color: '#fff',
              opacity: 0.85,
            }}
          >
            {r.label}
          </div>
        ))}
      </div>
      <div className="flex justify-between mt-0.5">
        {ranges.map((r, i) => (
          <span key={i} className="text-[9px]" style={{ color: 'var(--muted)', flex: 1 }}>
            {r.from !== undefined ? `${r.from}${unit || ''}` : ''}
            {r.to !== undefined ? ` → ${r.to}${unit || ''}` : '+'}
          </span>
        ))}
      </div>
    </div>
  );
}
