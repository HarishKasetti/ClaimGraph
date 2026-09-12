import React from 'react';
import { Send, Loader2, HelpCircle } from 'lucide-react';

export default function ClaimInput({ claimText, setClaimText, onVerify, isVerifying }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (claimText.trim()) {
      onVerify(claimText.trim());
    }
  };

  return (
    <div className="glass-card" style={{ padding: '1.8rem', marginBottom: '2rem', border: '1px solid rgba(99, 102, 241, 0.3)', boxShadow: '0 8px 32px rgba(99, 102, 241, 0.12)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.8rem' }}>
        <HelpCircle size={20} color="var(--accent-cyan)" />
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Scientific Claim Verification Prompt
        </h3>
      </div>

      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1.2rem' }}>
        Formulate a natural language question or hypothesis to verify against the ingested papers (e.g. <em>"Does intermittent fasting improve insulin sensitivity?"</em>).
      </p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.8rem' }}>
        <input
          type="text"
          className="input-field"
          placeholder="Enter claim or question to verify..."
          value={claimText}
          onChange={(e) => setClaimText(e.target.value)}
          disabled={isVerifying}
          style={{ fontSize: '1rem', padding: '0.9rem 1.2rem', background: 'rgba(15, 23, 42, 0.9)' }}
        />
        <button
          type="submit"
          className="btn-primary"
          disabled={isVerifying || !claimText.trim()}
          style={{ padding: '0 1.8rem', whiteSpace: 'nowrap' }}
        >
          {isVerifying ? (
            <>
              <Loader2 size={18} className="animate-spin" /> Verifying...
            </>
          ) : (
            <>
              <Send size={18} /> Verify Claim
            </>
          )}
        </button>
      </form>
    </div>
  );
}
