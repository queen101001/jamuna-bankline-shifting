'use client';

export default function DiagramChartReading() {
  return (
    <div className="my-2 flex justify-center">
      <svg viewBox="0 0 380 220" width="380" height="220" className="max-w-full">
        {/* Axes */}
        <line x1="40" y1="170" x2="340" y2="170" stroke="#64748b" strokeWidth="1" />
        <line x1="40" y1="20" x2="40" y2="170" stroke="#64748b" strokeWidth="1" />
        <text x="190" y="195" textAnchor="middle" fill="#64748b" fontSize="9">Year</text>
        <text x="15" y="95" textAnchor="middle" fill="#64748b" fontSize="9" transform="rotate(-90,15,95)">Distance (m)</text>

        {/* X-axis labels */}
        <text x="60" y="183" fill="#64748b" fontSize="8">2000</text>
        <text x="130" y="183" fill="#64748b" fontSize="8">2010</text>
        <text x="200" y="183" fill="#64748b" fontSize="8">2020</text>
        <text x="270" y="183" fill="#64748b" fontSize="8">2025</text>

        {/* Vertical reference line at 2020 */}
        <line x1="200" y1="25" x2="200" y2="170" stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="4 2" />

        {/* Historical data points (solid dots with line) */}
        <polyline
          points="60,120 80,115 100,125 120,110 140,105 160,115 180,100 200,95"
          fill="none" stroke="#f1f5f9" strokeWidth="1.5"
        />
        {[
          [60, 120], [80, 115], [100, 125], [120, 110], [140, 105], [160, 115], [180, 100], [200, 95],
        ].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="3" fill="#f1f5f9" />
        ))}

        {/* Forecast uncertainty band */}
        <polygon
          points="200,85 230,75 260,68 290,60 320,50 320,130 290,115 260,105 230,98 200,95"
          fill="#06b6d4" opacity="0.12"
        />

        {/* Forecast line (dashed) */}
        <polyline
          points="200,95 230,88 260,85 290,82 320,78"
          fill="none" stroke="#06b6d4" strokeWidth="2" strokeDasharray="6 3"
        />
        {[
          [230, 88], [260, 85], [290, 82], [320, 78],
        ].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="2.5" fill="#06b6d4" />
        ))}

        {/* Changepoint marker */}
        <polygon points="140,97 145,90 150,97 145,104" fill="#fb923c" />

        {/* Annotation lines + labels */}
        {/* Historical label */}
        <line x1="80" y1="115" x2="80" y2="40" stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="2 2" />
        <rect x="55" y="28" width="80" height="14" rx="3" fill="rgba(241,245,249,0.1)" />
        <text x="95" y="38" textAnchor="middle" fill="#f1f5f9" fontSize="8" fontWeight="500">
          Historical (solid)
        </text>

        {/* Forecast label */}
        <line x1="290" y1="82" x2="290" y2="40" stroke="#06b6d4" strokeWidth="0.5" strokeDasharray="2 2" />
        <rect x="257" y="28" width="80" height="14" rx="3" fill="rgba(6,182,212,0.1)" />
        <text x="297" y="38" textAnchor="middle" fill="#06b6d4" fontSize="8" fontWeight="500">
          Forecast (dashed)
        </text>

        {/* Band label */}
        <line x1="320" y1="90" x2="345" y2="75" stroke="#64748b" strokeWidth="0.5" />
        <text x="335" y="70" fill="#64748b" fontSize="7">Uncertainty</text>
        <text x="335" y="78" fill="#64748b" fontSize="7">band</text>

        {/* Changepoint label */}
        <line x1="145" y1="104" x2="145" y2="145" stroke="#fb923c" strokeWidth="0.5" strokeDasharray="2 2" />
        <rect x="110" y="145" width="70" height="14" rx="3" fill="rgba(251,146,60,0.1)" />
        <text x="145" y="155" textAnchor="middle" fill="#fb923c" fontSize="8" fontWeight="500">
          Changepoint
        </text>

        {/* Reference line label */}
        <text x="200" y="215" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="8">
          Last observation (2020)
        </text>
      </svg>
    </div>
  );
}
