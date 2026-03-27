const TYPES = {
  historical: { bg: 'rgba(56,189,248,0.12)', color: '#38bdf8', border: 'rgba(56,189,248,0.25)', label: 'Historical data' },
  direct:     { bg: 'rgba(34,197,94,0.12)',   color: '#22c55e', border: 'rgba(34,197,94,0.25)',   label: 'Direct forecast' },
  rolling:    { bg: 'rgba(251,146,60,0.12)',  color: '#fb923c', border: 'rgba(251,146,60,0.25)',  label: 'Rolling forecast' },
};

export default function ForecastTypeTag({ forecastType }) {
  const t = TYPES[forecastType] || TYPES.rolling;
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={{
        background: t.bg,
        color: t.color,
        border: `1px solid ${t.border}`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: t.color }}
      />
      {t.label}
    </span>
  );
}
