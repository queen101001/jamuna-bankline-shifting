export default function LoadingSpinner({ size = 24, label = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16">
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        className="animate-spin"
        style={{ color: 'var(--accent)' }}
      >
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.2" />
        <path
          d="M12 2a10 10 0 0 1 10 10"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
        />
      </svg>
      <p className="text-sm" style={{ color: 'var(--text-dim)' }}>
        {label}
      </p>
    </div>
  );
}
