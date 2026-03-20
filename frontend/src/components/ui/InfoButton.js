'use client';
import { useState } from 'react';
import { Info } from 'lucide-react';
import InfoModal from './info/InfoModal';
import PAGE_INFO from '@/lib/pageInfo';

export default function InfoButton({ pageId }) {
  const [open, setOpen] = useState(false);
  const info = PAGE_INFO[pageId];

  if (!info) return null;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 left-6 z-40 w-12 h-12 rounded-full border flex items-center justify-center shadow-lg transition-all hover:scale-110"
        style={{
          background: 'var(--card)',
          borderColor: 'var(--accent)',
          color: 'var(--accent)',
          boxShadow: '0 0 20px rgba(6,182,212,0.15)',
        }}
        title="Page information"
      >
        <Info size={20} />
      </button>

      <InfoModal
        open={open}
        onClose={() => setOpen(false)}
        title={info.title}
        subtitle={info.subtitle}
        sections={info.sections}
      />
    </>
  );
}
