export default function QuantileBadge({ q10, q50, q90 }) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      {q10 != null && (
        <span style={{ color: 'var(--muted)' }}>
          {q10.toFixed(0)}
        </span>
      )}
      <span className="font-semibold" style={{ color: 'var(--accent)' }}>
        {q50.toFixed(0)}
      </span>
      {q90 != null && (
        <span style={{ color: 'var(--muted)' }}>
          {q90.toFixed(0)}
        </span>
      )}
      <span style={{ color: 'var(--muted)' }}>m</span>
    </div>
  );
}
