'use client';

export default function DiagramForecastModes() {
  return (
    <div className="my-2 flex justify-center">
      <svg viewBox="0 0 360 140" width="360" height="140" className="max-w-full">
        {/* Direct Forecast Panel */}
        <rect x="5" y="5" width="170" height="125" rx="6" fill="none" stroke="#334155" strokeWidth="1" />
        <text x="90" y="22" textAnchor="middle" fill="#06b6d4" fontSize="10" fontWeight="600">
          Direct Forecast
        </text>
        <text x="90" y="34" textAnchor="middle" fill="#64748b" fontSize="8">
          (2021–2025)
        </text>

        {/* Historical block */}
        <rect x="15" y="50" width="70" height="30" rx="4" fill="rgba(241,245,249,0.1)" stroke="#64748b" strokeWidth="0.5" />
        <text x="50" y="69" textAnchor="middle" fill="#94a3b8" fontSize="8">1991–2020</text>

        {/* Arrow */}
        <line x1="90" y1="65" x2="110" y2="65" stroke="#06b6d4" strokeWidth="1.5" markerEnd="url(#arrowCyan)" />

        {/* Forecast block — narrow band */}
        <rect x="115" y="55" width="50" height="20" rx="4" fill="rgba(6,182,212,0.15)" stroke="#06b6d4" strokeWidth="1" />
        <text x="140" y="69" textAnchor="middle" fill="#06b6d4" fontSize="8">2021–25</text>

        <text x="90" y="105" textAnchor="middle" fill="#64748b" fontSize="8">
          Single pass — narrow uncertainty
        </text>

        {/* Rolling Forecast Panel */}
        <rect x="185" y="5" width="170" height="125" rx="6" fill="none" stroke="#334155" strokeWidth="1" />
        <text x="270" y="22" textAnchor="middle" fill="#eab308" fontSize="10" fontWeight="600">
          Rolling Forecast
        </text>
        <text x="270" y="34" textAnchor="middle" fill="#64748b" fontSize="8">
          (2026–2100)
        </text>

        {/* Cascading blocks with widening bands */}
        <rect x="195" y="48" width="30" height="14" rx="3" fill="rgba(6,182,212,0.2)" stroke="#06b6d4" strokeWidth="0.5" />
        <text x="210" y="58" textAnchor="middle" fill="#06b6d4" fontSize="6">21–25</text>

        <line x1="228" y1="55" x2="238" y2="55" stroke="#eab308" strokeWidth="1" markerEnd="url(#arrowYellow)" />

        <rect x="241" y="45" width="30" height="20" rx="3" fill="rgba(234,179,8,0.15)" stroke="#eab308" strokeWidth="0.5" />
        <text x="256" y="58" textAnchor="middle" fill="#eab308" fontSize="6">26–30</text>

        <line x1="274" y1="55" x2="284" y2="55" stroke="#eab308" strokeWidth="1" markerEnd="url(#arrowYellow)" />

        <rect x="287" y="42" width="30" height="26" rx="3" fill="rgba(234,179,8,0.25)" stroke="#eab308" strokeWidth="0.5" />
        <text x="302" y="58" textAnchor="middle" fill="#eab308" fontSize="6">31–35</text>

        <line x1="320" y1="55" x2="330" y2="55" stroke="#eab308" strokeWidth="1" markerEnd="url(#arrowYellow)" />
        <text x="345" y="58" fill="#eab308" fontSize="10">...</text>

        {/* Widening band indicator */}
        <line x1="210" y1="80" x2="302" y2="80" stroke="#64748b" strokeWidth="0.5" />
        <line x1="210" y1="78" x2="210" y2="82" stroke="#64748b" strokeWidth="0.5" />
        <line x1="302" y1="78" x2="302" y2="82" stroke="#64748b" strokeWidth="0.5" />
        <text x="256" y="92" textAnchor="middle" fill="#64748b" fontSize="7">
          Bands widen each iteration
        </text>

        <text x="270" y="118" textAnchor="middle" fill="#64748b" fontSize="8">
          Each window feeds the next —
        </text>
        <text x="270" y="128" textAnchor="middle" fill="#eab308" fontSize="8">
          uncertainty compounds
        </text>

        {/* Arrow markers */}
        <defs>
          <marker id="arrowCyan" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0,0 6,2 0,4" fill="#06b6d4" />
          </marker>
          <marker id="arrowYellow" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0,0 6,2 0,4" fill="#eab308" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}
