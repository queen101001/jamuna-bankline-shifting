'use client';
import { useRouter } from 'next/navigation';
import ErosionBar from '@/components/ui/ErosionBar';
import useAppStore from '@/store';

export default function ReachCard({ reach_id, leftForecast, rightForecast, isAnomalous }) {
  const router = useRouter();
  const { setSelectedReach, selectedReach } = useAppStore();
  const isSelected = selectedReach === reach_id;

  function handleClick(bankSide) {
    setSelectedReach(reach_id);
    router.push(`/series/${reach_id}/${bankSide}`);
  }

  return (
    <div
      className="rounded-xl p-4 border transition-all cursor-pointer group"
      style={{
        background: isSelected ? 'rgba(6,182,212,0.06)' : 'var(--card)',
        borderColor: isSelected ? 'var(--accent)' : isAnomalous ? 'rgba(251,146,60,0.4)' : 'var(--border)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-mono font-bold px-2 py-0.5 rounded"
            style={{ background: 'rgba(6,182,212,0.1)', color: 'var(--accent)' }}
          >
            R{String(reach_id).padStart(2, '0')}
          </span>
          {isAnomalous && (
            <span
              className="text-xs px-1.5 py-0.5 rounded"
              style={{ background: 'rgba(251,146,60,0.1)', color: '#fb923c' }}
            >
              ⚠ protected
            </span>
          )}
        </div>
      </div>

      {/* Banks */}
      <div className="flex flex-col gap-3">
        {leftForecast && (
          <div
            onClick={() => handleClick('left')}
            className="hover:opacity-80 transition-opacity"
          >
            <ErosionBar
              bankSide="left"
              q50={leftForecast.q50}
              q10={leftForecast.q10}
              q90={leftForecast.q90}
            />
          </div>
        )}
        {rightForecast && (
          <div
            onClick={() => handleClick('right')}
            className="hover:opacity-80 transition-opacity"
          >
            <ErosionBar
              bankSide="right"
              q50={rightForecast.q50}
              q10={rightForecast.q10}
              q90={rightForecast.q90}
            />
          </div>
        )}
      </div>
    </div>
  );
}
