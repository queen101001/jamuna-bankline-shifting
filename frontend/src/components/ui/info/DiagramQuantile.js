'use client';

export default function DiagramQuantile() {
  return (
    <div className="my-2 flex justify-center">
      <svg viewBox="0 0 340 170" width="340" height="170" className="max-w-full">
        {/* Background grid */}
        <line x1="40" y1="30" x2="240" y2="30" stroke="#334155" strokeWidth="0.5" strokeDasharray="3 3" />
        <line x1="40" y1="80" x2="240" y2="80" stroke="#334155" strokeWidth="0.5" strokeDasharray="3 3" />
        <line x1="40" y1="130" x2="240" y2="130" stroke="#334155" strokeWidth="0.5" strokeDasharray="3 3" />

        {/* Shaded band between q10 and q90 */}
        <rect x="40" y="30" width="200" height="100" rx="4" fill="#06b6d4" opacity="0.1" />

        {/* q90 line */}
        <line x1="40" y1="30" x2="240" y2="30" stroke="#06b6d4" strokeWidth="1" strokeDasharray="4 3" />
        {/* q50 line (bold) */}
        <line x1="40" y1="80" x2="240" y2="80" stroke="#06b6d4" strokeWidth="2.5" />
        {/* q10 line */}
        <line x1="40" y1="130" x2="240" y2="130" stroke="#06b6d4" strokeWidth="1" strokeDasharray="4 3" />

        {/* Actual value dot */}
        <circle cx="150" cy="65" r="5" fill="#f1f5f9" stroke="#06b6d4" strokeWidth="1.5" />

        {/* Labels on right */}
        <text x="250" y="34" fill="#06b6d4" fontSize="10" fontWeight="500">q90 — Upper bound</text>
        <text x="250" y="84" fill="#06b6d4" fontSize="10" fontWeight="700">q50 — Best estimate</text>
        <text x="250" y="134" fill="#06b6d4" fontSize="10" fontWeight="500">q10 — Lower bound</text>

        {/* Band label */}
        <text x="140" y="160" textAnchor="middle" fill="#64748b" fontSize="9">
          80% of actual values should land in the shaded band
        </text>

        {/* Dot label */}
        <line x1="155" y1="63" x2="185" y2="48" stroke="#94a3b8" strokeWidth="0.5" />
        <text x="187" y="48" fill="#f1f5f9" fontSize="8">Actual value</text>

        {/* Band brace on left */}
        <text x="30" y="85" textAnchor="middle" fill="#64748b" fontSize="18">{"}"}</text>
        <text x="15" y="82" textAnchor="middle" fill="#64748b" fontSize="8" transform="rotate(-90,15,82)">80%</text>
      </svg>
    </div>
  );
}
