import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Navbar from './components/Navbar';
import TopicEntry from './components/TopicEntry';
import ClaimSuggestions from './components/ClaimSuggestions';
import ClaimInput from './components/ClaimInput';
import PaperList from './components/PaperList';
import VerdictBadge from './components/VerdictBadge';
import ReformulationBadge from './components/ReformulationBadge';
import EvidenceList from './components/EvidenceList';
import LowSampleBanner from './components/LowSampleBanner';
import ChartsContainer from './components/ChartsContainer';

const API_BASE = '/api/v1';

export default function App() {
  const [activeTopic, setActiveTopic] = useState(null); // { topic, topic_id }
  const [isSubmittingTopic, setIsSubmittingTopic] = useState(false);
  const [claimText, setClaimText] = useState('');
  const [verificationResult, setVerificationResult] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyError, setVerifyError] = useState(null);

  // 1. Poll topic status every 3s while status is pending or ingesting
  const { data: statusData } = useQuery({
    queryKey: ['topicStatus', activeTopic?.topic_id],
    queryFn: async () => {
      if (!activeTopic?.topic_id) return null;
      const res = await fetch(`${API_BASE}/topics/${activeTopic.topic_id}/status`);
      if (!res.ok) throw new Error('Failed to fetch topic status');
      return res.json();
    },
    enabled: !!activeTopic?.topic_id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'ready' || status === 'failed' ? false : 3000;
    },
  });

  // 2. Fetch claim suggestions once topic status is 'ready'
  const isReady = statusData?.status === 'ready';

  const { data: suggestionsData } = useQuery({
    queryKey: ['topicSuggestions', activeTopic?.topic_id],
    queryFn: async () => {
      if (!activeTopic?.topic_id || !isReady) return null;
      const res = await fetch(`${API_BASE}/topics/${activeTopic.topic_id}/suggestions`);
      if (!res.ok) return { suggestions: [] };
      return res.json();
    },
    enabled: !!activeTopic?.topic_id && isReady,
  });

  // Handle submit topic
  const handleSubmitTopic = async (topicText) => {
    setIsSubmittingTopic(true);
    setVerificationResult(null);
    setVerifyError(null);
    try {
      const res = await fetch(`${API_BASE}/topics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topicText }),
      });
      if (!res.ok) throw new Error('Failed to create topic');
      const data = await res.json();
      setActiveTopic({ topic: topicText, topic_id: data.topic_id });
    } catch (err) {
      console.error(err);
      alert('Error initiating topic ingestion. Make sure the backend is running at http://localhost:8000');
    } finally {
      setIsSubmittingTopic(false);
    }
  };

  // Handle verify claim
  const handleVerifyClaim = async (claimToVerify) => {
    if (!activeTopic?.topic_id) return;
    setIsVerifying(true);
    setVerifyError(null);
    try {
      const res = await fetch(`${API_BASE}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic_id: activeTopic.topic_id,
          claim: claimToVerify,
        }),
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Verification request failed');
      }
      const data = await res.json();
      setVerificationResult(data);
    } catch (err) {
      console.error('Verify error:', err);
      setVerifyError(err.message);
    } finally {
      setIsVerifying(false);
    }
  };

  // Reset screen
  const handleReset = () => {
    setActiveTopic(null);
    setClaimText('');
    setVerificationResult(null);
    setVerifyError(null);
  };

  return (
    <div style={{ minHeight: '100vh', paddingBottom: '4rem' }}>
      <Navbar activeTopic={activeTopic} onReset={handleReset} />

      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 1.5rem' }}>
        {/* Screen 1: Topic Entry & Ingestion Progress */}
        {!activeTopic || !isReady ? (
          <TopicEntry
            onSubmitTopic={handleSubmitTopic}
            statusData={statusData}
            isSubmitting={isSubmittingTopic}
          />
        ) : (
          <div>
            {/* Screen 2: Ready Topic Workspace */}

            {/* Suggestions Strip at top */}
            <ClaimSuggestions
              suggestions={suggestionsData?.suggestions}
              onSelectSuggestion={(sug) => {
                setClaimText(sug);
                handleVerifyClaim(sug);
              }}
            />

            {/* Prominent Claim Input Box (above papers) */}
            <ClaimInput
              claimText={claimText}
              setClaimText={setClaimText}
              onVerify={handleVerifyClaim}
              isVerifying={isVerifying}
            />

            {/* Error banner if verify failed */}
            {verifyError && (
              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: 'var(--radius-sm)', padding: '1rem', marginBottom: '1.5rem', color: 'var(--color-refute)' }}>
                <strong>Verification Error:</strong> {verifyError}
              </div>
            )}

            {/* Screen 3: Verification Verdict & Visual Analytics */}
            {verificationResult && (
              <div style={{ marginBottom: '3rem' }}>
                <VerdictBadge
                  verdict={verificationResult.verdict}
                  confidence={verificationResult.confidence}
                />

                <ReformulationBadge
                  reformulatedClaim={
                    verificationResult.reformulated_claim !== claimText
                      ? verificationResult.reformulated_claim
                      : null
                  }
                />

                <LowSampleBanner show={verificationResult.low_sample_warning} />

                <EvidenceList evidence={verificationResult.evidence} />

                <ChartsContainer
                  scatter3dJson={verificationResult.scatter_3d}
                  confidenceChartJson={verificationResult.confidence_chart}
                  contradictionMapJson={verificationResult.contradiction_map}
                />
              </div>
            )}

            {/* Screen 2 Secondary Context: 10 Paper Cards below */}
            <PaperList papers={statusData?.papers || []} />
          </div>
        )}
      </main>
    </div>
  );
}
