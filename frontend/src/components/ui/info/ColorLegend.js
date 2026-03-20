'use client';

export default function ColorLegend({ items }) {
  return (
    <div className="flex flex-col gap-1.5 my-1">
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="w-3 h-3 rounded-full shrink-0"
            style={{ background: item.color }}
          />
          <span className="text-xs" style={{ color: 'var(--text-dim)' }}>
            {item.label}
          </span>
        </div>
      ))}
    </div>
  );
}
