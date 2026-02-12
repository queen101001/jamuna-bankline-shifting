export default function StatusDot({ ok }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full"
      style={{ background: ok ? '#22c55e' : '#f97316' }}
    />
  );
}
