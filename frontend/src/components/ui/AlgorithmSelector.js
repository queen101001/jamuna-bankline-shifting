'use client';
import useAppStore from '@/store';
import { ALGORITHMS } from '@/lib/algorithms';

export default function AlgorithmSelector() {
  const { activeAlgorithm, setActiveAlgorithm } = useAppStore();

  return (
    <div className="flex flex-col items-center">
      <p
        className="text-xs font-medium uppercase tracking-widest text-center mb-2"
        style={{ color: 'var(--muted)' }}
      >
        Algorithm
      </p>
      <div className="flex flex-wrap justify-center gap-1.5">
        {ALGORITHMS.map((a) => {
          const active = activeAlgorithm === a.key;
          return (
            <button
              key={a.key}
              onClick={() => setActiveAlgorithm(a.key)}
              className="px-3 py-1.5 rounded-full border text-xs font-medium whitespace-nowrap transition-all"
              style={{
                borderColor: active ? a.color : 'var(--border)',
                background: active ? `${a.color}18` : 'transparent',
                color: active ? a.color : 'var(--muted)',
              }}
            >
              {a.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
