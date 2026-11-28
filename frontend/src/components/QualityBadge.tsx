import React from 'react';
import { QualityCheck, QualityTier } from '../types';
import { CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

export const QualityBadge: React.FC<{ quality?: QualityCheck }> = ({ quality }) => {
  if (!quality) return null;

  const getTierConfig = (tier: QualityTier) => {
    switch (tier) {
      case 'good':
        return {
          label: 'Good Quality',
          badgeClass: 'badge-success',
          icon: <CheckCircle2 size={14} aria-hidden="true" />,
          desc: 'Image satisfies resolution, sharpness, and illumination criteria.'
        };
      case 'acceptable':
        return {
          label: 'Acceptable Quality',
          badgeClass: 'badge-warning',
          icon: <AlertTriangle size={14} aria-hidden="true" />,
          desc: 'Minor quality flags detected; suitable for preliminary analysis with caution.'
        };
      case 'poor':
      default:
        return {
          label: 'Poor Quality (Advisory)',
          badgeClass: 'badge-danger',
          icon: <XCircle size={14} aria-hidden="true" />,
          desc: 'Image quality is low. Please capture a clear, well-lit image with the affected skin area visible.'
        };
    }
  };

  const conf = getTierConfig(quality.quality_tier);

  return (
    <div className="glass-panel" style={{ padding: '16px', borderRadius: 'var(--radius-md)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px', gap: '8px' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-main)' }}>Image Quality Check</span>
        <span className={`badge ${conf.badgeClass}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
          {conf.icon}
          <span style={{ fontWeight: 600 }}>{conf.label}</span>
        </span>
      </div>

      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
        {conf.desc}
      </p>

      {/* Metrics Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '8px',
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border-subtle)',
        padding: '10px 8px',
        borderRadius: 'var(--radius-sm)',
        fontSize: '0.78rem',
        textAlign: 'center'
      }}>
        <div>
          <span style={{ color: 'var(--text-dim)', display: 'block', fontSize: '0.72rem' }}>Sharpness</span>
          <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>{quality.blur_score}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-dim)', display: 'block', fontSize: '0.72rem' }}>Luminance</span>
          <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>{quality.brightness_mean}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-dim)', display: 'block', fontSize: '0.72rem' }}>Contrast</span>
          <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>{quality.contrast_std}</span>
        </div>
      </div>

      {quality.advisory_flags && quality.advisory_flags.length > 0 && (
        <div style={{ marginTop: '10px', fontSize: '0.76rem', color: 'var(--danger)', background: 'var(--danger-bg)', padding: '6px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--danger-border)' }}>
          <b>Advisory Flags: </b>
          <span>{quality.advisory_flags.join(', ')}</span>
        </div>
      )}
    </div>
  );
};
