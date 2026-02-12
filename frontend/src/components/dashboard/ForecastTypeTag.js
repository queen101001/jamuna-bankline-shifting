export default function ForecastTypeTag({ forecastType }) {
  const direct = forecastType === 'direct';
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={{
        background: direct ? 'rgba(34,197,94,0.12)' : 'rgba(251,146,60,0.12)',
        color: direct ? '#22c55e' : '#fb923c',
        border: `1px solid ${direct ? 'rgba(34,197,94,0.25)' : 'rgba(251,146,60,0.25)'}`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: direct ? '#22c55e' : '#fb923c' }}
      />
      {direct ? 'Direct forecast' : 'Rolling forecast'}
    </span>
  );
}
