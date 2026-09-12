import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function LowSampleBanner({ show }) {
  if (!show) return null;

  return (
    <div
      style={{
        background: 'rgba(245, 158, 11, 0.15)',
        border: '1px solid rgba(245, 158, 11, 0.4)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.9rem 1.2rem',
        marginBottom: '1.5rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.8rem',
      }}
    >
      <AlertTriangle size={20} color="var(--color-neutral)" />
      <div>
        <h5 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--color-neutral)', margin: 0 }}>
          Low Sample Size Warning
        </h5>
        <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
          Fewer than 3 parsed paper vector chunks were available for distance calibration. Confidence score degraded accordingly.
        </p>
      </div>
    </div>
  );
}
