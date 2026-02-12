'use client';
import { ChevronUp, ChevronDown } from 'lucide-react';
import useAppStore from '@/store';
import ForecastTypeTag from './ForecastTypeTag';

const MIN_YEAR = 2021;
const MAX_YEAR = 2099;
const DIRECT_MAX = 2025;

export default function YearSelector() {
  const { selectedYear, setSelectedYear } = useAppStore();

  function clamp(y) {
    return Math.max(MIN_YEAR, Math.min(MAX_YEAR, y));
  }

  function handleInput(e) {
    const val = parseInt(e.target.value, 10);
    if (!isNaN(val)) setSelectedYear(clamp(val));
  }

  function handleKey(e) {
    if (e.key === 'ArrowUp') setSelectedYear(clamp(selectedYear + 1));
    if (e.key === 'ArrowDown') setSelectedYear(clamp(selectedYear - 1));
  }

  const forecastType = selectedYear <= DIRECT_MAX ? 'direct' : 'rolling';

  return (
    <div className="flex flex-col items-center gap-4">
      <div>
        <p
          className="text-xs font-medium uppercase tracking-widest text-center mb-2"
          style={{ color: 'var(--muted)' }}
        >
          Forecast Year
        </p>
        <div
          className="flex items-center gap-3 px-5 py-3 rounded-2xl border"
          style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
        >
          <button
            onClick={() => setSelectedYear(clamp(selectedYear - 1))}
            className="p-1 rounded-md transition-colors hover:bg-white/10"
            style={{ color: 'var(--text-dim)' }}
            aria-label="Decrease year"
          >
            <ChevronDown size={20} />
          </button>

          <input
            type="number"
            value={selectedYear}
            onChange={handleInput}
            onKeyDown={handleKey}
            min={MIN_YEAR}
            max={MAX_YEAR}
            className="w-24 text-center text-4xl font-mono font-bold bg-transparent border-none outline-none"
            style={{ color: 'var(--accent)', caretColor: 'var(--accent)' }}
          />

          <button
            onClick={() => setSelectedYear(clamp(selectedYear + 1))}
            className="p-1 rounded-md transition-colors hover:bg-white/10"
            style={{ color: 'var(--text-dim)' }}
            aria-label="Increase year"
          >
            <ChevronUp size={20} />
          </button>
        </div>
      </div>

      <ForecastTypeTag forecastType={forecastType} />

      {forecastType === 'rolling' && (
        <p className="text-xs text-center max-w-sm" style={{ color: 'var(--muted)' }}>
          Predictions beyond 2025 use iterative rolling forecasts. Uncertainty intervals widen with distance from 2020.
        </p>
      )}
    </div>
  );
}
