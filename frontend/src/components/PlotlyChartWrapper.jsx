import React from 'react';
import Plot from 'react-plotly.js';

export default function PlotlyChartWrapper({ chartJson, defaultTitle = 'Chart' }) {
  if (!chartJson) {
    return (
      <div style={{ height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        No chart data available
      </div>
    );
  }

  let plotObj = null;
  try {
    plotObj = typeof chartJson === 'string' ? JSON.parse(chartJson) : chartJson;
  } catch (err) {
    console.error('Failed to parse Plotly JSON:', err);
    return (
      <div style={{ height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-refute)' }}>
        Invalid Chart Format
      </div>
    );
  }

  const data = plotObj.data || [];
  const layout = {
    autosize: true,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#94a3b8', family: 'Inter, sans-serif' },
    margin: { l: 50, r: 30, t: 40, b: 40 },
    ...plotObj.layout,
  };

  return (
    <div style={{ width: '100%', height: '380px' }}>
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
