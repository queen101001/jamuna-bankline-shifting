'use client';
import { Info, AlertTriangle, CheckCircle, TrendingDown, TrendingUp } from 'lucide-react';

const VARIANTS = {
  info: { border: 'var(--accent)', bg: 'rgba(6,182,212,0.06)', Icon: Info },
  warning: { border: '#eab308', bg: 'rgba(234,179,8,0.06)', Icon: AlertTriangle },
  success: { border: '#22c55e', bg: 'rgba(34,197,94,0.06)', Icon: CheckCircle },
  erosion: { border: 'var(--erosion)', bg: 'rgba(239,68,68,0.06)', Icon: TrendingDown },
  deposition: { border: 'var(--deposition)', bg: 'rgba(34,197,94,0.06)', Icon: TrendingUp },
};

export default function InfoTip({ variant = 'info', title, children }) {
  const v = VARIANTS[variant] || VARIANTS.info;
  const { Icon } = v;

  return (
    <div
      className="rounded-lg px-3 py-2.5 my-1"
      style={{
        background: v.bg,
        borderLeft: `3px solid ${v.border}`,
      }}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <Icon size={13} style={{ color: v.border }} />
        <span className="text-xs font-semibold" style={{ color: v.border }}>
          {title}
        </span>
      </div>
      <div className="text-xs leading-relaxed" style={{ color: 'var(--text-dim)' }}>
        {children}
      </div>
    </div>
  );
}
