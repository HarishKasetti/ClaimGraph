import React, { useState } from 'react';
import PlotlyChartWrapper from './PlotlyChartWrapper';
import { Box, BarChart3, Grid3X3 } from 'lucide-react';

export default function ChartsContainer({ scatter3dJson, confidenceChartJson, contradictionMapJson }) {
  const [activeTab, setActiveTab] = useState('all');

  return (
    <div style={{ marginBottom: '3rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.2rem' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Interactive Visual Analytics & Spatial Graphing
        </h3>

        {/* View mode toggle */}
        <div style={{ display: 'flex', gap: '0.4rem', background: 'rgba(15, 23, 42, 0.6)', padding: '0.3rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <button
            onClick={() => setActiveTab('all')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              background: activeTab === 'all' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'all' ? '#fff' : 'var(--text-secondary)',
            }}
          >
            3-Panel View
          </button>
          <button
            onClick={() => setActiveTab('scatter')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              background: activeTab === 'scatter' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'scatter' ? '#fff' : 'var(--text-secondary)',
            }}
          >
            3D Scatter
          </button>
          <button
            onClick={() => setActiveTab('confidence')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              background: activeTab === 'confidence' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'confidence' ? '#fff' : 'var(--text-secondary)',
            }}
          >
            Confidence
          </button>
          <button
            onClick={() => setActiveTab('contradiction')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              background: activeTab === 'contradiction' ? 'var(--accent-indigo)' : 'transparent',
              color: activeTab === 'contradiction' ? '#fff' : 'var(--text-secondary)',
            }}
          >
            Contradictions
          </button>
        </div>
      </div>

      {/* Grid Layout for 3 panels side by side */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: activeTab === 'all' ? 'repeat(auto-fit, minmax(380px, 1fr))' : '1fr',
          gap: '1.5rem',
        }}
      >
        {/* Panel 1: 3D Scatter Plot */}
        {(activeTab === 'all' || activeTab === 'scatter') && (
          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem' }}>
              <Box size={18} color="var(--accent-cyan)" />
              <h4 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                3D UMAP Vector Embedding Projection
              </h4>
            </div>
            <PlotlyChartWrapper chartJson={scatter3dJson} defaultTitle="3D Stance Scatter Plot" />
          </div>
        )}

        {/* Panel 2: Calibrated Confidence Bar Chart */}
        {(activeTab === 'all' || activeTab === 'confidence') && (
          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem' }}>
              <BarChart3 size={18} color="var(--accent-indigo)" />
              <h4 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Calibrated Stance Margin & Confidence
              </h4>
            </div>
            <PlotlyChartWrapper chartJson={confidenceChartJson} defaultTitle="Confidence Bar Chart" />
          </div>
        )}

        {/* Panel 3: Pairwise Paper Contradiction Heatmap */}
        {(activeTab === 'all' || activeTab === 'contradiction') && (
          <div className="glass-card" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem' }}>
              <Grid3X3 size={18} color="var(--accent-purple)" />
              <h4 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Literature Contradiction Heatmap (Phase 6)
              </h4>
            </div>
            <PlotlyChartWrapper chartJson={contradictionMapJson} defaultTitle="Contradiction Map" />
          </div>
        )}
      </div>
    </div>
  );
}
