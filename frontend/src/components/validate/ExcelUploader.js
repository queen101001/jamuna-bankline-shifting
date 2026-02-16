'use client';
import { useCallback, useRef, useState } from 'react';
import { Upload, FileSpreadsheet } from 'lucide-react';
import parseValidationExcel from '@/lib/parseValidationExcel';
import useAppStore from '@/store';

export default function ExcelUploader() {
  const { setValidationData } = useAppStore();
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const [parsing, setParsing] = useState(false);
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
    </div>
  );
}
