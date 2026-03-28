'use client';
import { useCallback, useRef, useState } from 'react';
import { Upload, FileSpreadsheet, ChevronDown, ChevronUp } from 'lucide-react';
import parseValidationExcel from '@/lib/parseValidationExcel';
import useAppStore from '@/store';

export default function ExcelUploader() {
  const { setValidationData } = useAppStore();
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const [parsing, setParsing] = useState(false);
  const [showGuide, setShowGuide] = useState(false);
  const inputRef = useRef(null);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      if (!file.name.endsWith('.xlsx')) {
        setError('Only .xlsx files are supported');
        return;
      }

      setError(null);
      setParsing(true);

      try {
        const buffer = await file.arrayBuffer();
        const result = parseValidationExcel(buffer);

        if (result.data.length === 0) {
          setError('No valid data found in the file');
          setParsing(false);
          return;
        }

        setValidationData(result);
      } catch (e) {
        setError(e.message || 'Failed to parse Excel file');
      } finally {
        setParsing(false);
      }
    },
    [setValidationData],
  );

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    handleFile(file);
  }

  function onDragOver(e) {
    e.preventDefault();
    setDragOver(true);
  }

  function onDragLeave() {
    setDragOver(false);
  }

  function onChange(e) {
    handleFile(e.target.files?.[0]);
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current?.click()}
        className="w-full max-w-lg rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all"
        style={{
          borderColor: dragOver ? 'var(--accent)' : 'var(--border)',
          background: dragOver ? 'rgba(6,182,212,0.06)' : 'var(--card)',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".xlsx"
          onChange={onChange}
          className="hidden"
        />

        {parsing ? (
          <div className="flex flex-col items-center gap-3">
            <FileSpreadsheet size={40} className="animate-pulse" style={{ color: 'var(--accent)' }} />
            <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
              Parsing Excel file...
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <Upload size={40} style={{ color: dragOver ? 'var(--accent)' : 'var(--muted)' }} />
            <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
              Drop your validation .xlsx file here
            </p>
            <p className="text-xs" style={{ color: 'var(--muted)' }}>
              or click to browse. Supports multi-header format with variable years.
            </p>
          </div>
        )}
      </div>

      {error && (
        <p className="text-sm font-medium" style={{ color: 'var(--erosion)' }}>
          {error}
        </p>
      )}

      {/* Formatting guide */}
      <div className="w-full max-w-lg">
        <button
          onClick={() => setShowGuide((v) => !v)}
          className="flex items-center gap-1.5 text-xs font-medium transition-colors"
          style={{ color: 'var(--text-dim)' }}
        >
          {showGuide ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          Required Excel format
        </button>
        {showGuide && (
          <div
            className="mt-2 rounded-xl border p-4 text-xs space-y-3"
            style={{ background: 'var(--card)', borderColor: 'var(--border)', color: 'var(--text-dim)' }}
          >
            <p style={{ color: 'var(--text)' }}>
              Your .xlsx file must use a <strong>2-row hierarchical header</strong> matching the training data format:
            </p>
            <div className="overflow-x-auto">
              <table className="font-mono text-xs border-collapse" style={{ borderColor: 'var(--border)' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th className="px-2 py-1 text-left" style={{ color: 'var(--muted)' }}>Row</th>
                    <th className="px-2 py-1 text-left">Col A</th>
                    <th className="px-2 py-1 text-left">Col B</th>
                    <th className="px-2 py-1 text-left">Col C</th>
                    <th className="px-2 py-1 text-left">Col D</th>
                    <th className="px-2 py-1 text-left">Col E</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="px-2 py-1" style={{ color: 'var(--muted)' }}>1</td>
                    <td className="px-2 py-1" style={{ color: 'var(--accent)' }}>Reaches</td>
                    <td className="px-2 py-1" style={{ color: 'var(--accent)' }}>Distance(2021)</td>
                    <td className="px-2 py-1" style={{ color: 'var(--accent)' }}></td>
                    <td className="px-2 py-1" style={{ color: 'var(--accent)' }}>Distance(2022)</td>
                    <td className="px-2 py-1" style={{ color: 'var(--accent)' }}>...</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="px-2 py-1" style={{ color: 'var(--muted)' }}>2</td>
                    <td className="px-2 py-1"></td>
                    <td className="px-2 py-1">Right Bank (m)</td>
                    <td className="px-2 py-1">Left Bank (m)</td>
                    <td className="px-2 py-1">Right Bank (m)</td>
                    <td className="px-2 py-1">...</td>
                  </tr>
                  <tr>
                    <td className="px-2 py-1" style={{ color: 'var(--muted)' }}>3+</td>
                    <td className="px-2 py-1">1</td>
                    <td className="px-2 py-1">1234.5</td>
                    <td className="px-2 py-1">5678.9</td>
                    <td className="px-2 py-1">1230.2</td>
                    <td className="px-2 py-1">...</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <ul className="space-y-1 pl-4 list-disc" style={{ color: 'var(--text-dim)' }}>
              <li><strong>Row 1</strong>: "Reaches" in col A, then "Distance(YYYY)" for each year (2021-2025)</li>
              <li><strong>Row 2</strong>: Empty col A, then alternating "Right Bank (m)" / "Left Bank (m)" sub-headers</li>
              <li><strong>Row 3+</strong>: Reach number (1-50) in col A, followed by bank distance values in meters</li>
              <li>Sign convention: Negative = erosion, Positive = deposition (both banks)</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
