'use client';

export default function Formula({ children }) {
  return (
    <code
      className="block my-2 px-3 py-2.5 rounded-lg text-xs font-mono whitespace-pre-line"
      style={{ background: 'rgba(6,182,212,0.08)', color: 'var(--accent)' }}
    >
      {children}
    </code>
  );
}
