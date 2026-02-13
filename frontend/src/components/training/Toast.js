'use client';
import { useEffect, useRef, useState } from 'react';
import { CheckCircle, X } from 'lucide-react';
import useAppStore from '@/store';

/**
 * Global toast notification.
 * Shown when training phase transitions to "ready".
 * Controlled via Zustand: toastMessage + setToastMessage.
 */
export default function Toast() {
  const { toastMessage, setToastMessage } = useAppStore();
  const [visible, setVisible] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (toastMessage) {
      setVisible(true);
      clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        setVisible(false);
        setTimeout(() => setToastMessage(null), 300); // clear after fade
      }, 5000);
    }
    return () => clearTimeout(timerRef.current);
  }, [toastMessage, setToastMessage]);

  if (!toastMessage) return null;

  return (
    <div
      className="fixed bottom-6 right-6 z-[60] flex items-start gap-3 px-4 py-3 rounded-xl shadow-2xl max-w-sm transition-all duration-300"
      style={{
        background: 'var(--surface)',
        border: '1px solid rgba(34,197,94,0.35)',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(8px)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(34,197,94,0.1)',
      }}
    >
      <CheckCircle size={18} className="shrink-0 mt-0.5" style={{ color: '#22c55e' }} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
          {toastMessage.title}
        </p>
        {toastMessage.body && (
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-dim)' }}>
            {toastMessage.body}
          </p>
        )}
      </div>
      <button
        onClick={() => { setVisible(false); setToastMessage(null); }}
        className="p-0.5 rounded transition-colors shrink-0"
        style={{ color: 'var(--muted)' }}
      >
        <X size={14} />
      </button>
    </div>
  );
}
