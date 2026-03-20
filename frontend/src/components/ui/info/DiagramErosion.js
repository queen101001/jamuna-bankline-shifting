'use client';

export default function DiagramErosion() {
  return (
    <div className="my-2 flex justify-center">
      <svg viewBox="0 0 320 200" width="320" height="200" className="max-w-full">
        {/* River channel */}
        <rect x="120" y="30" width="80" height="140" rx="8" fill="#0ea5e9" opacity="0.2" stroke="#0ea5e9" strokeWidth="1.5" />
        <text x="160" y="105" textAnchor="middle" fill="#0ea5e9" fontSize="11" fontWeight="600">
          River
        </text>
        <text x="160" y="118" textAnchor="middle" fill="#0ea5e9" fontSize="9" opacity="0.7">
          Channel
        </text>

        {/* Left Bank */}
        <rect x="10" y="30" width="100" height="140" rx="8" fill="none" stroke="#64748b" strokeWidth="1" strokeDasharray="4 2" />
        <text x="60" y="50" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="600">
          LEFT BANK
        </text>

        {/* Left erosion arrow (pointing right, into river = land loss) */}
        <line x1="85" y1="80" x2="115" y2="80" stroke="#ef4444" strokeWidth="2" markerEnd="url(#arrowRed)" />
        <text x="60" y="78" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="500">
          +value
        </text>
        <text x="60" y="90" textAnchor="middle" fill="#ef4444" fontSize="8">
          = Erosion
        </text>

        {/* Left deposition arrow (pointing left, away from river = land gain) */}
        <line x1="85" y1="130" x2="55" y2="130" stroke="#22c55e" strokeWidth="2" markerEnd="url(#arrowGreen)" />
        <text x="60" y="148" textAnchor="middle" fill="#22c55e" fontSize="9" fontWeight="500">
          −value
        </text>
        <text x="60" y="160" textAnchor="middle" fill="#22c55e" fontSize="8">
          = Deposition
        </text>

        {/* Right Bank */}
        <rect x="210" y="30" width="100" height="140" rx="8" fill="none" stroke="#64748b" strokeWidth="1" strokeDasharray="4 2" />
        <text x="260" y="50" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="600">
          RIGHT BANK
        </text>

        {/* Right erosion arrow (pointing left, into river = land loss) */}
        <line x1="235" y1="80" x2="205" y2="80" stroke="#ef4444" strokeWidth="2" markerEnd="url(#arrowRed)" />
        <text x="260" y="78" textAnchor="middle" fill="#ef4444" fontSize="9" fontWeight="500">
          −value
        </text>
        <text x="260" y="90" textAnchor="middle" fill="#ef4444" fontSize="8">
          = Erosion
        </text>

        {/* Right deposition arrow (pointing right, away from river = land gain) */}
        <line x1="235" y1="130" x2="265" y2="130" stroke="#22c55e" strokeWidth="2" markerEnd="url(#arrowGreen)" />
        <text x="260" y="148" textAnchor="middle" fill="#22c55e" fontSize="9" fontWeight="500">
          +value
        </text>
        <text x="260" y="160" textAnchor="middle" fill="#22c55e" fontSize="8">
          = Deposition
        </text>

        {/* Arrow markers */}
        <defs>
          <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0,0 8,3 0,6" fill="#ef4444" />
          </marker>
          <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0,0 8,3 0,6" fill="#22c55e" />
          </marker>
        </defs>

        {/* Title */}
        <text x="160" y="192" textAnchor="middle" fill="#64748b" fontSize="9">
          Sign convention: opposite for each bank side
        </text>
      </svg>
    </div>
  );
}
