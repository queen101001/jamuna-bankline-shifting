'use client';
import { useEffect, useRef, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import * as LucideIcons from 'lucide-react';
import InfoSection from './InfoSection';
import InfoTip from './InfoTip';
import Formula from './Formula';
import ColorLegend from './ColorLegend';

// Lazy diagram imports — mapped by name
import DiagramErosion from './DiagramErosion';
import DiagramQuantile from './DiagramQuantile';
import DiagramForecastModes from './DiagramForecastModes';
import DiagramExcelFormat from './DiagramExcelFormat';
import DiagramChartReading from './DiagramChartReading';
import MetricScale from './MetricScale';
import AlgorithmCard from './AlgorithmCard';

const DIAGRAM_MAP = {
  erosion: DiagramErosion,
  quantile: DiagramQuantile,
  forecastModes: DiagramForecastModes,
  excelFormat: DiagramExcelFormat,
  chartReading: DiagramChartReading,
};

function BlockRenderer({ block }) {
  switch (block.type) {
    case 'text':
      return <p className="whitespace-pre-line">{block.content}</p>;
    case 'heading':
      return (
        <p className="text-xs font-semibold mt-1" style={{ color: 'var(--text)' }}>
          {block.content}
        </p>
      );
    case 'tip':
      return (
        <InfoTip variant={block.variant} title={block.title}>
          {block.content}
        </InfoTip>
      );
    case 'formula':
      return <Formula>{block.content}</Formula>;
    case 'interpretation':
      return (
        <p className="whitespace-pre-line" style={{ color: 'var(--muted)' }}>
          {block.content}
        </p>
      );
    case 'diagram': {
      const Diagram = DIAGRAM_MAP[block.component];
      return Diagram ? <Diagram /> : null;
    }
    case 'metric-scale':
      return <MetricScale {...block.config} />;
    case 'color-legend':
      return <ColorLegend items={block.items} />;
    case 'algorithms':
      return (
        <div className="flex flex-col gap-2 mt-1">
          {block.algorithms.map((algo) => (
            <AlgorithmCard key={algo.key} {...algo} defaultExpanded={block.expanded} />
          ))}
        </div>
      );
    default:
      return null;
  }
}

function getIcon(iconName) {
  if (!iconName) return null;
  return LucideIcons[iconName] || null;
}

export default function InfoModal({ open, onClose, title, subtitle, sections }) {
  const overlayRef = useRef(null);
  const contentRef = useRef(null);
  const [activeSection, setActiveSection] = useState(sections?.[0]?.id || '');

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handler);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  // Scroll-spy via IntersectionObserver
  useEffect(() => {
    if (!open || !contentRef.current || !sections?.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        }
      },
      {
        root: contentRef.current,
        rootMargin: '-10% 0px -70% 0px',
        threshold: 0.1,
      }
    );

    const els = sections.map((s) => document.getElementById(s.id)).filter(Boolean);
    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [open, sections]);

  const scrollTo = useCallback((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setActiveSection(id);
    }
  }, []);

  if (!open || !sections) return null;

  return createPortal(
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(6px)' }}
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div
        className="relative w-full max-w-4xl max-h-[85vh] rounded-2xl border overflow-hidden flex flex-col"
        style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      >
        {/* Header */}
        <div className="shrink-0 px-6 pt-5 pb-3 border-b" style={{ borderColor: 'var(--border)' }}>
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1 rounded-lg transition-colors z-10"
            style={{ color: 'var(--muted)' }}
            onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text)')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--muted)')}
          >
            <X size={18} />
          </button>
          <h2 className="text-lg font-bold pr-8" style={{ color: 'var(--text)' }}>
            {title}
          </h2>
          {subtitle && (
            <p className="text-xs mt-0.5" style={{ color: 'var(--muted)' }}>
              {subtitle}
            </p>
          )}
        </div>

        {/* Mobile TOC — horizontal pills */}
        <div
          className="md:hidden shrink-0 flex gap-1.5 px-4 py-2 overflow-x-auto border-b"
          style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
        >
          {sections.map((s) => (
            <button
              key={s.id}
              onClick={() => scrollTo(s.id)}
              className="px-2.5 py-1 rounded-full text-[10px] font-medium whitespace-nowrap transition-all shrink-0"
              style={{
                background: activeSection === s.id ? 'rgba(6,182,212,0.12)' : 'transparent',
                color: activeSection === s.id ? 'var(--accent)' : 'var(--muted)',
                border: `1px solid ${activeSection === s.id ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              {s.heading}
            </button>
          ))}
        </div>

        {/* Body: sidebar + content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Desktop sidebar TOC */}
          <nav
            className="hidden md:flex flex-col shrink-0 w-52 overflow-y-auto py-3 px-2 border-r"
            style={{ borderColor: 'var(--border)', background: 'rgba(15,23,42,0.5)' }}
          >
            {sections.map((s) => {
              const Icon = getIcon(s.icon);
              const active = activeSection === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => scrollTo(s.id)}
                  className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-[11px] transition-all w-full"
                  style={{
                    background: active ? 'rgba(6,182,212,0.1)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--muted)',
                  }}
                >
                  {Icon && <Icon size={12} />}
                  <span className="truncate">{s.heading}</span>
                </button>
              );
            })}
          </nav>

          {/* Content area */}
          <div ref={contentRef} className="flex-1 overflow-y-auto px-6 py-4">
            <div className="flex flex-col gap-6">
              {sections.map((s) => {
                const Icon = getIcon(s.icon);
                return (
                  <InfoSection key={s.id} id={s.id} heading={s.heading} icon={Icon}>
                    {s.blocks.map((block, i) => (
                      <BlockRenderer key={i} block={block} />
                    ))}
                  </InfoSection>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
