import React from 'react';
import { Sparkles, MessageSquareQuote } from 'lucide-react';

export default function ReformulationBadge({ reformulatedClaim }) {
  if (!reformulatedClaim) return null;

  return (
    <div
      className="glass-card"
      style={{
        padding: '1rem 1.4rem',
        marginBottom: '1.5rem',
        background: 'rgba(99, 102, 241, 0.1)',
        border: '1px solid rgba(99, 102, 241, 0.3)',
        borderRadius: 'var(--radius-sm)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.8rem',
      }}
    >
      <div style={{ background: 'rgba(99, 102, 241, 0.2)', padding: '0.4rem', borderRadius: '8px', display: 'flex' }}>
        <MessageSquareQuote size={18} color="var(--accent-cyan)" />
      </div>
      <div>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Question Reformulated into Declarative Claim
        </span>
        <p style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--text-primary)', margin: 0 }}>
          Your question was interpreted as: <strong style={{ color: '#ffffff' }}>"{reformulatedClaim}"</strong>
        </p>
      </div>
    </div>
  );
}
