'use client';
import { useState } from 'react';
import { Download } from 'lucide-react';
import { ALGORITHMS } from '@/lib/algorithms';
import { exportPredictions } from '@/lib/api';

export default function DownloadPanel() {
  const [startYear, setStartYear] = useState(2021);
  const [endYear, setEndYear] = useState(2025);
  const [algorithm, setAlgorithm] = useState('tft');
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState(null);

  async function handleDownload() {
    if (startYear > endYear) {
      setError('Start year must be \u2264 end year');
      return;
    }
    setError(null);
    setDownloading(true);
    try {
      const blob = await exportPredictions(startYear, endYear, algorithm);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `jamuna_predictions_${algorithm}_${startYear}-${endYear}.xlsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message || 'Download failed');
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="flex flex-col items-center">
      <p
        className="text-xs font-medium uppercase tracking-widest text-center mb-2"
        style={{ color: 'var(--muted)' }}
      >
        Export Predictions
      </p>
      <div
        className="flex flex-wrap items-end justify-center gap-3 px-5 py-3 rounded-2xl border"
        style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <label className="flex flex-col gap-1">
          <span className="text-xs" style={{ color: 'var(--text-dim)' }}>From</span>
          <input
            type="number"
            min={2021}
            max={2100}
            value={startYear}
            onChange={(e) => setStartYear(Math.max(2021, Math.min(2100, +e.target.value)))}
            className="w-20 text-center text-sm font-mono bg-transparent border rounded px-2 py-1.5 outline-none"
            style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs" style={{ color: 'var(--text-dim)' }}>To</span>
          <input
            type="number"
            min={2021}
            max={2100}
            value={endYear}
            onChange={(e) => setEndYear(Math.max(2021, Math.min(2100, +e.target.value)))}
            className="w-20 text-center text-sm font-mono bg-transparent border rounded px-2 py-1.5 outline-none"
            style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs" style={{ color: 'var(--text-dim)' }}>Algorithm</span>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="text-sm border rounded px-2 py-1.5 outline-none"
            style={{ borderColor: 'var(--border)', color: 'var(--text)', background: 'var(--card)' }}
          >
            {ALGORITHMS.map((a) => (
              <option key={a.key} value={a.key} style={{ background: 'var(--card)', color: 'var(--text)' }}>{a.label}</option>
            ))}
          </select>
        </label>
        <button
          onClick={handleDownload}
          disabled={downloading}
          className="flex items-center gap-2 px-4 py-1.5 rounded-xl border text-sm font-medium transition-all"
          style={{
            background: downloading ? 'var(--border)' : 'var(--accent)',
            borderColor: downloading ? 'var(--border)' : 'var(--accent)',
            color: downloading ? 'var(--muted)' : '#0f172a',
            opacity: downloading ? 0.7 : 1,
          }}
        >
          <Download size={16} />
          {downloading ? 'Downloading...' : 'Download .xlsx'}
        </button>
      </div>
      {error && (
        <p className="text-xs mt-2" style={{ color: 'var(--erosion)' }}>
          {error}
        </p>
      )}
    </div>
  );
}
