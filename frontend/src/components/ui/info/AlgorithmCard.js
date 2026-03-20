'use client';
import { useState } from 'react';
import { ChevronDown, ChevronUp, CheckCircle, XCircle, Lightbulb } from 'lucide-react';

export default function AlgorithmCard({
  key: algoKey,
  name,
  color,
  analogy,
  howItWorks,
  strengths,
  weaknesses,
  trustWhen,
  defaultExpanded = false,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div
      className="rounded-lg border overflow-hidden"
      style={{ borderColor: 'var(--border)', borderTop: `3px solid ${color}` }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 text-left"
        style={{ background: 'rgba(15,23,42,0.4)' }}
      >
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
          <span className="text-xs font-semibold" style={{ color: 'var(--text)' }}>
            {name}
          </span>
        </div>
        {expanded ? (
          <ChevronUp size={14} style={{ color: 'var(--muted)' }} />
        ) : (
          <ChevronDown size={14} style={{ color: 'var(--muted)' }} />
        )}
      </button>

      {/* Collapsed: show analogy */}
      {!expanded && (
        <div className="px-3 pb-2">
          <p className="text-[11px] italic" style={{ color: 'var(--muted)' }}>
            &ldquo;{analogy}&rdquo;
          </p>
        </div>
      )}

      {/* Expanded: full details */}
      {expanded && (
        <div className="px-3 pb-3 flex flex-col gap-2.5">
          <div>
            <p className="text-[11px] italic mb-1.5" style={{ color: 'var(--muted)' }}>
              &ldquo;{analogy}&rdquo;
            </p>
            <p className="text-xs" style={{ color: 'var(--text-dim)' }}>
              {howItWorks}
            </p>
          </div>

          <div className="grid grid-cols-3 gap-2">
            {/* Strengths */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <CheckCircle size={10} style={{ color: '#22c55e' }} />
                <span className="text-[10px] font-semibold" style={{ color: '#22c55e' }}>
                  Strengths
                </span>
              </div>
              {strengths.map((s, i) => (
                <p key={i} className="text-[10px] leading-relaxed" style={{ color: 'var(--text-dim)' }}>
                  &bull; {s}
                </p>
              ))}
            </div>

            {/* Weaknesses */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <XCircle size={10} style={{ color: '#ef4444' }} />
                <span className="text-[10px] font-semibold" style={{ color: '#ef4444' }}>
                  Weaknesses
                </span>
              </div>
              {weaknesses.map((w, i) => (
                <p key={i} className="text-[10px] leading-relaxed" style={{ color: 'var(--text-dim)' }}>
                  &bull; {w}
                </p>
              ))}
            </div>

            {/* Trust when */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <Lightbulb size={10} style={{ color: '#eab308' }} />
                <span className="text-[10px] font-semibold" style={{ color: '#eab308' }}>
                  Trust when
                </span>
              </div>
              {trustWhen.map((t, i) => (
                <p key={i} className="text-[10px] leading-relaxed" style={{ color: 'var(--text-dim)' }}>
                  &bull; {t}
                </p>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
