import React from 'react';
import { Lightbulb, ArrowUpRight } from 'lucide-react';

export default function ClaimSuggestions({ suggestions = [], onSelectSuggestion }) {
  if (!suggestions || suggestions.length === 0) return null;

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
        <Lightbulb size={16} color="var(--color-neutral)" />
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Suggested Claims from Literature
        </span>
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
        {suggestions.map((suggestion, idx) => (
          <button
            key={idx}
            onClick={() => onSelectSuggestion(suggestion)}
            className="glass-card"
            style={{
              padding: '0.6rem 1rem',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.85rem',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              background: 'rgba(30, 41, 59, 0.5)',
              textAlign: 'left',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-indigo)';
              e.currentTarget.style.background = 'rgba(99, 102, 241, 0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
              e.currentTarget.style.background = 'rgba(30, 41, 59, 0.5)';
            }}
          >
            <span>{suggestion}</span>
            <ArrowUpRight size={14} color="var(--accent-cyan)" />
          </button>
        ))}
      </div>
    </div>
  );
}
