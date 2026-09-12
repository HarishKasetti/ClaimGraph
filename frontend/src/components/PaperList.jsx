import React from 'react';
import { BookOpen, ExternalLink, FileCheck, Layers } from 'lucide-react';

export default function PaperList({ papers = [] }) {
  if (!papers || papers.length === 0) return null;

  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Layers size={18} color="var(--accent-indigo)" />
          <h4 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Topic Literature Corpus ({papers.length} Ingested Papers)
          </h4>
        </div>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Secondary Evidence Context
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
        {papers.map((paper, idx) => {
          const title = paper.title || paper.paper_title || `Paper #${idx + 1}`;
          const venue = paper.venue || paper.source || 'Journal Article';
          const citations = paper.citation_count ?? 0;
          const doi = paper.doi;
          const pdfUrl = paper.pdf_url;
          const paperId = paper.paper_id || paper._id || `paper-${idx}`;

          return (
            <div key={paperId} className="glass-card" style={{ padding: '1.2rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.5rem', marginBottom: '0.6rem' }}>
                  <h5 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.4, flex: 1 }}>
                    {title}
                  </h5>
                  {pdfUrl && (
                    <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', border: '1px solid rgba(6, 182, 212, 0.3)', fontSize: '0.7rem' }}>
                      <FileCheck size={12} /> PDF Parsed
                    </span>
                  )}
                </div>

                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.8rem' }}>
                  {venue} • <span style={{ color: 'var(--text-muted)' }}>{citations} citations</span>
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '0.6rem', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                  ID: {paperId.substring(0, 10)}...
                </span>

                {doi ? (
                  <a
                    href={`https://doi.org/${doi}`}
                    target="_blank"
                    rel="noreferrer"
                    style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.2rem' }}
                  >
                    DOI Link <ExternalLink size={12} />
                  </a>
                ) : (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Pubmed</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
