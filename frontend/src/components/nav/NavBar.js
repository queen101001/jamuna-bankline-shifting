'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { Waves, Activity, AlertTriangle, BarChart2 } from 'lucide-react';
import { getHealth } from '@/lib/api';
import StatusDot from '@/components/ui/StatusDot';

const links = [
  { href: '/', label: 'Dashboard', icon: Waves },
  { href: '/anomaly', label: 'Anomalies', icon: AlertTriangle },
  { href: '/evaluate', label: 'Metrics', icon: BarChart2 },
];

export default function NavBar() {
  const pathname = usePathname();
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 h-16 flex items-center justify-between px-6 border-b"
      style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
    >
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2 no-underline">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'var(--accent)' }}
        >
          <Waves size={18} color="#020817" strokeWidth={2.5} />
        </div>
        <div>
          <p className="text-sm font-bold leading-tight" style={{ color: 'var(--text)' }}>
            Jamuna
          </p>
          <p className="text-xs leading-tight" style={{ color: 'var(--text-dim)' }}>
            Bankline Prediction
          </p>
        </div>
      </Link>

      {/* Nav links */}
      <nav className="flex items-center gap-1">
        {links.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors no-underline"
              style={{
                background: active ? 'rgba(6,182,212,0.12)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--text-dim)',
              }}
            >
              <Icon size={15} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* API status */}
      <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-dim)' }}>
        <StatusDot ok={health?.model_loaded === true} />
        <span>{health?.model_loaded ? 'API ready' : health ? 'Model not loaded' : 'Connecting…'}</span>
      </div>
    </header>
  );
}
