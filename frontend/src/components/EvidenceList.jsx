import React from 'react';
import { Quote, FileText, Target } from 'lucide-react';

export default function EvidenceList({ evidence = [] }) {
  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.2rem' }}>
        <Quote size={20} color="var(--accent-cyan)" />
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Top Grounding Evidence Sentences ({evidence.length})
        </h3>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
        {evidence.map((item, idx) => {
          const simPct = Math.round((item.cosine_sim || 0) * 100);
          return (
            <div
              key={idx}
              style={{
                background: 'rgba(15, 23, 42, 0.7)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 'var(--radius-sm)',
                padding: '1.1rem 1.3rem',
                borderLeft: '4px solid var(--accent-cyan)',
              }}
            >
              <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)', fontStyle: 'italic', marginBottom: '0.6rem', lineHeight: 1.5 }}>
                "{item.text}"
              </p>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <FileText size={14} color="var(--text-secondary)" />
                  <span>Paper ID: <strong style={{ color: 'var(--text-secondary)' }}>{item.paper_id}</strong></span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', background: 'rgba(6, 182, 212, 0.12)', padding: '0.2rem 0.6rem', borderRadius: '12px', border: '1px solid rgba(6, 182, 212, 0.2)' }}>
                  <Target size={12} color="var(--accent-cyan)" />
                  <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{simPct}% Similarity</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
