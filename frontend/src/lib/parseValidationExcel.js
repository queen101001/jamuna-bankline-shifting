import * as XLSX from 'xlsx';

const YEAR_RE = /Distance\((\d{4})\)/i;
const LEFT_BANK_RE = /Left Bank/i;
const RIGHT_BANK_RE = /Right Bank/i;

/**
 * Parse an uploaded validation Excel file with multi-header format.
 *
 * Expected format (2-row header):
 *   Row 1: Reaches | Distance(2021) | (merged) | Distance(2022) | ...
 *   Row 2:         | Right Bank (m) | Left Bank (m) | Right Bank (m) | ...
 *   Row 3+: 1      | 2361.95        | 1712.42       | ...
 *
 * Left Bank values are multiplied by -1 to match the system's corrected
 * coordinate space.
 *
 * @param {ArrayBuffer} buffer - File contents as ArrayBuffer
 * @returns {{ years: number[], data: Array<{reach_id: number, bank_side: string, year: number, observed: number}> }}
 */
export default function parseValidationExcel(buffer) {
  const wb = XLSX.read(buffer, { type: 'array' });

  // Prefer "Validation" sheet, fall back to "Data"
  const sheetName = wb.SheetNames.includes('Validation') && hasData(wb, 'Validation')
    ? 'Validation'
    : wb.SheetNames.includes('Data')
    ? 'Data'
    : wb.SheetNames[0];

  const ws = wb.Sheets[sheetName];
  const rows = XLSX.utils.sheet_to_json(ws, { header: 1, defval: null });

  if (rows.length < 3) {
    throw new Error('Excel file must have at least 3 rows (2 header rows + data)');
  }

  const headerRow1 = rows[0]; // Year labels
  const headerRow2 = rows[1]; // Bank side labels

  // Build column mapping: colIndex → { year, bankSide }
  const colMap = [];
  let reachesCol = -1;
  let lastYear = null;

  for (let c = 0; c < headerRow1.length; c++) {
    const top = headerRow1[c] != null ? String(headerRow1[c]).trim() : '';
    const bottom = headerRow2[c] != null ? String(headerRow2[c]).trim() : '';

    if (/reaches/i.test(top)) {
      reachesCol = c;
      continue;
    }

    // Check if this column has a year label (or inherits from previous via merge)
    const yearMatch = YEAR_RE.exec(top);
    if (yearMatch) {
      lastYear = parseInt(yearMatch[1], 10);
    }

    if (lastYear && RIGHT_BANK_RE.test(bottom)) {
      colMap.push({ col: c, year: lastYear, bankSide: 'right' });
    } else if (lastYear && LEFT_BANK_RE.test(bottom)) {
      colMap.push({ col: c, year: lastYear, bankSide: 'left' });
    }
  }

  if (reachesCol === -1) {
    // Try to find reaches column by looking for numeric sequential values in first col
    reachesCol = 0;
  }

  if (colMap.length === 0) {
    throw new Error('No valid Distance columns found. Expected format: Distance(YEAR) / Right Bank (m) | Left Bank (m)');
  }

  const years = [...new Set(colMap.map((c) => c.year))].sort((a, b) => a - b);
  const data = [];

  // Parse data rows (starting from row index 2)
  for (let r = 2; r < rows.length; r++) {
    const row = rows[r];
    const reachId = row[reachesCol];
    if (reachId == null || isNaN(Number(reachId))) continue;
    const reach = Number(reachId);

    for (const { col, year, bankSide } of colMap) {
      const raw = row[col];
      if (raw == null || raw === '' || isNaN(Number(raw))) continue;

      let value = Number(raw);

      // Apply Left Bank * (-1) normalization
      if (bankSide === 'left') {
        value = value * -1;
      }

      data.push({
        reach_id: reach,
        bank_side: bankSide,
        year,
        observed: value,
      });
    }
  }

  return { years, data };
}

function hasData(wb, sheetName) {
  const ws = wb.Sheets[sheetName];
  if (!ws) return false;
  const rows = XLSX.utils.sheet_to_json(ws, { header: 1, defval: null });
  // Check if there's actual data beyond headers
  return rows.length > 2 && rows.some((r, i) => i >= 2 && r.some((c) => c != null));
}
