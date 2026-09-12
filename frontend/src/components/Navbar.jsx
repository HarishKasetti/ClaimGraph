import React from 'react';
import { Network, Activity, RotateCcw } from 'lucide-react';

export default function Navbar({ activeTopic, onReset }) {
  return (
    <header className="glass-card" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0, marginBottom: '2rem' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1rem 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', cursor: 'pointer' }} onClick={onReset}>
          <div style={{ background: 'linear-gradient(135deg, #6366f1, #06b6d4)', padding: '0.5rem', borderRadius: '10px', display: 'flex' }}>
            <Network size={24} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.4rem', fontWeight: 800, background: 'linear-gradient(135deg, #f8fafc, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              ClaimGraph
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
              Autonomous Scientific Claim Verification Engine
            </p>
          </div>
        </div>

        {activeTopic && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(15, 23, 42, 0.6)', padding: '0.4rem 1rem', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.08)' }}>
              <Activity size={16} color="var(--accent-cyan)" />
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Topic:</span>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>{activeTopic.topic}</span>
            </div>

            <button
              onClick={onReset}
              className="btn-primary"
              style={{ padding: '0.4rem 0.9rem', fontSize: '0.85rem', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', boxShadow: 'none' }}
            >
              <RotateCcw size={14} /> New Topic
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
