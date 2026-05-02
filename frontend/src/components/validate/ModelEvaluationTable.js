'use client';
import { useMemo, useState } from 'react';
import { useQueries } from '@tanstack/react-query';
import { BarChart3, Download } from 'lucide-react';
import { getBaselineYearPrediction, getPredictionForYear } from '@/lib/api';
import { ALGORITHMS } from '@/lib/algorithms';
import {
  buildEvaluationRows,
  buildPredictionMap,
  EXPECTED_BANK_POINT_COUNT,
} from '@/lib/validationMetrics';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

function formatMetric(value, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function downloadEvaluationTableImage({ rows, algorithmLabel, years }) {
  const width = 2400;
  const margin = 96;
  const titleHeight = 150;
  const rowHeight = 78;
  const tableTop = margin + titleHeight;
  const columns = [
    { label: 'Year', width: 300 },
    { label: 'Section', width: 650 },
    { label: 'RMSE', width: 400 },
    { label: 'MAE', width: 400 },
    { label: 'R²', width: 350 },
  ];
  const tableWidth = columns.reduce((sum, column) => sum + column.width, 0);
  const height = tableTop + rowHeight * (rows.length + 1) + margin;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');

  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);

  context.fillStyle = '#111827';
  context.font = '700 52px Arial, Helvetica, sans-serif';
  context.textBaseline = 'top';
  context.fillText('Model Evaluation', margin, margin);

  context.fillStyle = '#4b5563';
  context.font = '400 30px Arial, Helvetica, sans-serif';
  context.fillText(`Algorithm: ${algorithmLabel} | Years: ${years.join(', ')}`, margin, margin + 70);

  let x = margin;
  context.font = '700 30px Arial, Helvetica, sans-serif';
  context.fillStyle = '#f3f4f6';
  context.fillRect(margin, tableTop, tableWidth, rowHeight);
  context.strokeStyle = '#d1d5db';
  context.lineWidth = 2;

  for (const column of columns) {
    context.strokeRect(x, tableTop, column.width, rowHeight);
    context.fillStyle = '#111827';
    context.fillText(column.label, x + 24, tableTop + 22);
    x += column.width;
  }

  context.font = '400 30px Arial, Helvetica, sans-serif';
  rows.forEach((row, rowIndex) => {
    const y = tableTop + rowHeight * (rowIndex + 1);
    const incomplete = !row.metrics.complete;
    const values = [
      row.year,
      incomplete ? `${row.section} (${row.pairCount}/${row.expectedCount})` : row.section,
      formatMetric(row.metrics.rmse),
      formatMetric(row.metrics.mae),
      formatMetric(row.metrics.r2, 4),
    ];

    context.fillStyle = rowIndex % 2 === 0 ? '#ffffff' : '#fafafa';
    context.fillRect(margin, y, tableWidth, rowHeight);

    x = margin;
    values.forEach((value, columnIndex) => {
      const column = columns[columnIndex];
      context.strokeStyle = '#d1d5db';
      context.strokeRect(x, y, column.width, rowHeight);
      context.fillStyle = incomplete ? '#6b7280' : '#111827';
      context.font =
        columnIndex >= 2
          ? '400 30px "Courier New", monospace'
          : '400 30px Arial, Helvetica, sans-serif';
      context.fillText(String(value), x + 24, y + 22, column.width - 48);
      x += column.width;
    });
  });

  const pngBlob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png', 1));
  if (!pngBlob) {
    throw new Error('Could not create table image');
  }

  const safeAlgorithm = algorithmLabel.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  downloadBlob(pngBlob, `model-evaluation-${safeAlgorithm}-${years[0]}-${years.at(-1)}.png`);
}

function MetricCell({ value, digits = 2, muted = false }) {
  return (
    <td
      className="px-4 py-3 font-mono text-sm"
      style={{ color: muted ? 'var(--muted)' : 'var(--text)' }}
    >
      {formatMetric(value, digits)}
    </td>
  );
}

export default function ModelEvaluationTable({ validationData }) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('tft');
  const [isDownloading, setIsDownloading] = useState(false);

  const selectedAlgorithmMeta =
    ALGORITHMS.find((algorithm) => algorithm.key === selectedAlgorithm) ?? ALGORITHMS[0];

  const years = validationData?.years ?? [];
  const observedData = validationData?.data ?? [];

  const predictionQueries = useQueries({
    queries: years.map((year) => ({
      queryKey: ['model-evaluation-year', selectedAlgorithm, year],
      queryFn: () =>
        selectedAlgorithm === 'tft'
          ? getPredictionForYear(year)
          : getBaselineYearPrediction(year, selectedAlgorithm),
      staleTime: 60_000,
    })),
  });

  const isLoading = predictionQueries.some((query) => query.isLoading);
  const failedQuery = predictionQueries.find((query) => query.isError);

  const predictionMapsByYear = useMemo(() => {
    return Object.fromEntries(
      years.map((year, index) => [year, buildPredictionMap(predictionQueries[index]?.data)]),
    );
  }, [years, predictionQueries]);

  const rows = useMemo(
    () =>
      buildEvaluationRows({
        years,
        observedData,
        predictionMapsByYear,
      }),
    [years, observedData, predictionMapsByYear],
  );

  const completeRows = rows.filter((row) => row.metrics.complete).length;

  async function handleDownloadTable() {
    setIsDownloading(true);
    try {
      await downloadEvaluationTableImage({
        rows,
        algorithmLabel: selectedAlgorithmMeta.label,
        years,
      });
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <div
        className="rounded-xl border p-4"
        style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div className="flex items-start gap-3">
            <BarChart3 size={22} style={{ color: selectedAlgorithmMeta.color }} />
            <div>
              <h2 className="text-lg font-semibold" style={{ color: 'var(--text)' }}>
                Model Evaluation
              </h2>
              <p className="text-sm mt-1" style={{ color: 'var(--text-dim)' }}>
                Comparing {years.length} uploaded year{years.length === 1 ? '' : 's'} with{' '}
                {EXPECTED_BANK_POINT_COUNT} points per bank section.
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <label className="flex flex-col gap-2 text-xs font-medium" style={{ color: 'var(--muted)' }}>
              Algorithm
              <select
                value={selectedAlgorithm}
                onChange={(event) => setSelectedAlgorithm(event.target.value)}
                className="min-w-56 rounded-lg border px-3 py-2 text-sm outline-none transition-colors"
                style={{
                  background: 'var(--surface)',
                  borderColor: 'var(--border)',
                  color: 'var(--text)',
                }}
              >
                {ALGORITHMS.map((algorithm) => (
                  <option key={algorithm.key} value={algorithm.key}>
                    {algorithm.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              onClick={handleDownloadTable}
              disabled={isLoading || Boolean(failedQuery) || rows.length === 0 || isDownloading}
              className="flex h-10 items-center justify-center gap-2 rounded-lg border px-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                background: 'rgba(6,182,212,0.12)',
                borderColor: 'rgba(6,182,212,0.3)',
                color: 'var(--accent)',
              }}
            >
              <Download size={16} />
              {isDownloading ? 'Preparing...' : 'Download Table'}
            </button>
          </div>
        </div>
      </div>

      {isLoading && (
        <div className="flex flex-col items-center gap-4 py-12">
          <LoadingSpinner label={`Fetching ${selectedAlgorithmMeta.label} predictions...`} />
        </div>
      )}

      {!isLoading && failedQuery && (
        <div
          className="rounded-xl border px-4 py-3 text-sm"
          style={{
            background: 'rgba(239,68,68,0.08)',
            borderColor: 'rgba(239,68,68,0.3)',
            color: 'var(--erosion)',
          }}
        >
          {failedQuery.error?.message || 'Could not load predictions for the selected algorithm.'}
        </div>
      )}

      {!isLoading && !failedQuery && (
        <div
          className="overflow-hidden rounded-xl border"
          style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead style={{ background: 'rgba(15,23,42,0.72)' }}>
                <tr>
                  {['Year', 'Section', 'RMSE', 'MAE', 'R²'].map((heading) => (
                    <th
                      key={heading}
                      className="border-b px-4 py-3 text-xs font-semibold uppercase tracking-wide"
                      style={{ borderColor: 'var(--border)', color: 'var(--text-dim)' }}
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {years.map((year) => {
                  const yearRows = rows.filter((row) => row.year === year);

                  return yearRows.map((row, index) => {
                    const incomplete = !row.metrics.complete;

                    return (
                      <tr
                        key={`${row.year}-${row.bankSide}`}
                        style={{
                          borderBottom: '1px solid var(--border)',
                          background: incomplete ? 'rgba(234,179,8,0.04)' : 'transparent',
                        }}
                      >
                        {index === 0 && (
                          <td
                            rowSpan={yearRows.length}
                            className="px-4 py-3 align-middle text-sm font-semibold"
                            style={{ color: 'var(--text)' }}
                          >
                            {year}
                          </td>
                        )}
                        <td className="px-4 py-3 text-sm" style={{ color: 'var(--text)' }}>
                          <div className="flex flex-col gap-1">
                            <span>{row.section}</span>
                            {incomplete && (
                              <span className="text-xs" style={{ color: '#eab308' }}>
                                Incomplete: {row.pairCount}/{row.expectedCount} matched points
                              </span>
                            )}
                          </div>
                        </td>
                        <MetricCell value={row.metrics.rmse} muted={incomplete} />
                        <MetricCell value={row.metrics.mae} muted={incomplete} />
                        <MetricCell value={row.metrics.r2} digits={4} muted={incomplete} />
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>

          <div
            className="border-t px-4 py-3 text-xs"
            style={{ borderColor: 'var(--border)', color: 'var(--text-dim)' }}
          >
            {completeRows}/{rows.length} rows complete for {selectedAlgorithmMeta.label}.
          </div>
        </div>
      )}
    </div>
  );
}
