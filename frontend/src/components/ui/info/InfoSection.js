'use client';

export default function InfoSection({ id, heading, icon: Icon, children }) {
  return (
    <div id={id} className="scroll-mt-4">
      <div className="flex items-center gap-2 mb-2">
        {Icon && <Icon size={16} style={{ color: 'var(--accent)' }} />}
        <h3 className="text-sm font-semibold" style={{ color: 'var(--accent)' }}>
          {heading}
        </h3>
      </div>
      <div className="text-xs leading-relaxed flex flex-col gap-2" style={{ color: 'var(--text-dim)' }}>
        {children}
      </div>
    </div>
  );
}
