import React from 'react';
import { CheckCircle2, XCircle, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function VerdictBadge({ verdict, confidence }) {
  const normVerdict = (verdict || 'insufficient_evidence').toLowerCase();
  const confidencePct = Math.round((confidence || 0) * 100);

  let badgeConfig = {
    title: 'INSUFFICIENT EVIDENCE',
    color: 'var(--color-neutral)',
    bg: 'var(--color-neutral-bg)',
    border: 'rgba(245, 158, 11, 0.4)',
    icon: <AlertTriangle size={24} color="var(--color-neutral)" />,
  };

  if (normVerdict === 'support' || normVerdict === 'supported') {
    badgeConfig = {
      title: 'CLAIM SUPPORTED',
      color: 'var(--color-support)',
      bg: 'var(--color-support-bg)',
      border: 'rgba(16, 185, 129, 0.4)',
      icon: <CheckCircle2 size={24} color="var(--color-support)" />,
    };
  } else if (normVerdict === 'refute' || normVerdict === 'refuted') {
    badgeConfig = {
      title: 'CLAIM REFUTED',
      color: 'var(--color-refute)',
      bg: 'var(--color-refute-bg)',
      border: 'rgba(239, 68, 68, 0.4)',
      icon: <XCircle size={24} color="var(--color-refute)" />,
    };
  }

  return (
    <div className="glass-card" style={{ padding: '1.8rem 2rem', marginBottom: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: `1px solid ${badgeConfig.border}`, background: badgeConfig.bg }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem' }}>
        <div style={{ padding: '0.8rem', borderRadius: '50%', background: 'rgba(0,0,0,0.3)', display: 'flex' }}>
          {badgeConfig.icon}
        </div>
        <div>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.2rem' }}>
            Consensus Verdict
          </div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: badgeConfig.color, margin: 0 }}>
            {badgeConfig.title}
          </h2>
        </div>
      </div>

      {/* Confidence Percentage Badge */}
      <div style={{ textAlign: 'right', display: 'flex', alignItems: 'center', gap: '1.2rem' }}>
        <div>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.2rem' }}>
            Calibrated Confidence
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-heading)' }}>
            {confidencePct}%
          </div>
        </div>

        {/* Circular gauge indicator */}
        <div style={{ position: 'relative', width: '56px', height: '56px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="56" height="56" viewBox="0 0 36 36">
            <path
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke="rgba(255, 255, 255, 0.1)"
              strokeWidth="3.5"
            />
            <path
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke={badgeConfig.color}
              strokeWidth="3.5"
              strokeDasharray={`${confidencePct}, 100`}
              strokeLinecap="round"
            />
          </svg>
          <ShieldCheck size={18} color="var(--text-secondary)" style={{ position: 'absolute' }} />
        </div>
      </div>
    </div>
  );
}
