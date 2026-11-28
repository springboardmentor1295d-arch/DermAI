import React from 'react';
import { Info } from 'lucide-react';

export const DisclaimerBanner: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  if (compact) {
    return (
      <div
        style={{
          background: '#f8fafc',
          border: '1px solid #cbd5e1',
          borderLeft: '3px solid var(--primary)',
          borderRadius: 'var(--radius-sm)',
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
        }}
      >
        <Info size={16} style={{ color: 'var(--primary)', flexShrink: 0 }} aria-hidden="true" />
        <span>
          <b style={{ color: 'var(--text-main)' }}>Clinical Advisory: </b>
          This output is an AI-assisted dermatoscopic decision-support classification, not a standalone medical diagnosis. Please consult a qualified dermatologist for clinical evaluation.
        </span>
      </div>
    );
  }

  return (
    <div
      style={{
        background: '#f8fafc',
        border: '1px solid #cbd5e1',
        borderLeft: '4px solid var(--primary)',
        borderRadius: 'var(--radius-md)',
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '14px',
        margin: '16px 0 8px 0',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      <Info size={20} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: '2px' }} aria-hidden="true" />
      <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: '1.55' }}>
        <b style={{ color: 'var(--text-main)', display: 'block', fontSize: '0.88rem', marginBottom: '4px' }}>
          Clinical Decision-Support Notice
        </b>
        This result is an AI-generated skin lesion classification intended strictly for informational and clinical decision support. It does not constitute a medical diagnosis or determine disease severity, staging, or clinical urgency. Consult a qualified dermatologist for formal evaluation.
        <div style={{ marginTop: '5px', fontSize: '0.78rem', color: 'var(--text-dim)' }}>
          Confidence values reflect temperature-calibrated class probabilities across the 6 PAD-UFES-20 diagnostic categories rather than clinical certainty.
        </div>
      </div>
    </div>
  );
};
