'use client';

export default function DiagramExcelFormat() {
  const cellStyle = {
    padding: '4px 8px',
    fontSize: '10px',
    border: '1px solid var(--border)',
    color: 'var(--text-dim)',
  };
  const headerStyle = {
    ...cellStyle,
    color: 'var(--accent)',
    fontWeight: 600,
    background: 'rgba(6,182,212,0.08)',
  };
  const subHeaderStyle = {
    ...cellStyle,
    color: 'var(--muted)',
    fontWeight: 500,
    background: 'rgba(6,182,212,0.04)',
  };

  return (
    <div className="my-2 overflow-x-auto">
      <table
        className="rounded-lg overflow-hidden text-xs"
        style={{ borderCollapse: 'collapse', background: 'var(--surface)' }}
      >
        <thead>
          {/* Row 1: Year headers */}
          <tr>
            <th style={headerStyle}>Reaches</th>
            <th colSpan={2} style={headerStyle}>Distance(2021)</th>
            <th colSpan={2} style={headerStyle}>Distance(2022)</th>
          </tr>
          {/* Row 2: Bank side headers */}
          <tr>
            <th style={subHeaderStyle}></th>
            <th style={subHeaderStyle}>Right Bank (m)</th>
            <th style={subHeaderStyle}>Left Bank (m)</th>
            <th style={subHeaderStyle}>Right Bank (m)</th>
            <th style={subHeaderStyle}>Left Bank (m)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ ...cellStyle, fontWeight: 600, color: 'var(--text)' }}>1</td>
            <td style={cellStyle}>2361.95</td>
            <td style={cellStyle}>1712.42</td>
            <td style={cellStyle}>2400.10</td>
            <td style={cellStyle}>1698.33</td>
          </tr>
          <tr>
            <td style={{ ...cellStyle, fontWeight: 600, color: 'var(--text)' }}>2</td>
            <td style={cellStyle}>1890.50</td>
            <td style={cellStyle}>2105.67</td>
            <td style={cellStyle}>1875.22</td>
            <td style={cellStyle}>2120.44</td>
          </tr>
          <tr>
            <td style={{ ...cellStyle, fontWeight: 600, color: 'var(--text)' }}>3</td>
            <td style={cellStyle}>...</td>
            <td style={cellStyle}>...</td>
            <td style={cellStyle}>...</td>
            <td style={cellStyle}>...</td>
          </tr>
        </tbody>
      </table>
      <p className="text-[9px] mt-1" style={{ color: 'var(--muted)' }}>
        Row 1 = year labels (merged across bank columns) &middot; Row 2 = bank side labels &middot; Row 3+ = data
      </p>
    </div>
  );
}
