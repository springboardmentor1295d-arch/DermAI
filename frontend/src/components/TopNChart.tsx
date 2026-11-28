import React from 'react';
import { TopNItem } from '../types';
import { BarChart3, Sliders, CheckCircle2 } from 'lucide-react';

export const TopNChart: React.FC<{ items: TopNItem[] }> = ({ items }) => {
  const getBarColor = (code: string) => {
    switch (code) {
      case 'MEL':
      case 'BCC':
      case 'SCC':
        return '#dc2626'; // Clinical red for malignant/carcinoma
      case 'ACK':
        return '#d97706'; // Clinical amber for pre-malignant
      case 'NEV':
      case 'SEK':
      default:
        return '#0d9488'; // Clinical teal for benign
    }
  };

  const hasCalibration = items.some((item) => item.calibrated_score !== undefined && item.calibrated_score !== null);
  const primaryItem = items.length > 0 ? items[0] : null;
  const supportingItems = items.length > 1 ? items.slice(1) : [];

  return (
    <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '24px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BarChart3 size={20} color="var(--primary)" aria-hidden="true" />
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-main)', margin: 0 }}>
            Model Predictions & Confidence Distribution
          </h3>
        </div>

        {hasCalibration && (
          <span
            className="badge badge-success"
            style={{ fontSize: '0.74rem', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
          >
            <Sliders size={12} aria-hidden="true" />
            <span style={{ fontWeight: 600 }}>Calibrated (Temperature Scaling)</span>
          </span>
        )}
      </div>

      {/* Primary Predicted Condition Card (Prominent, No #1 badge) */}
      {primaryItem && (() => {
        const primaryRaw = primaryItem.raw_score !== undefined ? primaryItem.raw_score : primaryItem.probability;
        const primaryCal = primaryItem.calibrated_score !== undefined ? primaryItem.calibrated_score : primaryRaw;
        const primaryRawPct = (primaryRaw * 100).toFixed(1);
        const primaryCalPct = (primaryCal * 100).toFixed(1);

        return (
          <div style={{
            background: 'var(--primary-subtle)',
            border: '1px solid rgba(13, 148, 136, 0.3)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
            marginBottom: '20px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={16} color="var(--primary)" />
                <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--primary-dark)' }}>
                  Primary Predicted Condition
                </span>
              </div>
              <span className="badge badge-info" style={{ fontSize: '0.75rem' }}>
                Top Ranked Prediction
              </span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
              <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-main)' }}>
                {primaryItem.class_code} – {primaryItem.class_name}
              </div>
              <div style={{ display: 'flex', gap: '16px', fontSize: '0.86rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>
                  Raw Model Score: <b style={{ color: 'var(--text-main)' }}>{primaryRawPct}%</b>
                </span>
                {hasCalibration && (
                  <span style={{ color: 'var(--primary-dark)', fontWeight: 700 }}>
                    Calibrated Confidence: {primaryCalPct}%
                  </span>
                )}
              </div>
            </div>

            {/* Primary Confidence Bar */}
            <div style={{
              height: '10px',
              background: '#e2e8f0',
              borderRadius: '5px',
              overflow: 'hidden',
              position: 'relative'
            }}>
              <div style={{
                height: '100%',
                width: `${Math.min(100, Math.max(0, parseFloat(primaryCalPct)))}%`,
                background: getBarColor(primaryItem.class_code),
                borderRadius: '5px',
                transition: 'width 0.6s ease'
              }} />
            </div>
          </div>
        );
      })()}

      {/* Supporting Predictions List (No #2, #3 badges) */}
      {supportingItems.length > 0 && (
        <div>
          <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Other Model Predictions
          </h4>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {supportingItems.map((item) => {
              const rawScore = item.raw_score !== undefined ? item.raw_score : item.probability;
              const calScore = item.calibrated_score !== undefined ? item.calibrated_score : rawScore;
              const rawPct = (rawScore * 100).toFixed(1);
              const calPct = (calScore * 100).toFixed(1);

              return (
                <div key={item.class_code} style={{
                  padding: '12px 14px',
                  background: '#f8fafc',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: '0.88rem', marginBottom: '6px', flexWrap: 'wrap', gap: '6px' }}>
                    <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>
                      <span style={{ color: 'var(--primary-dark)', fontWeight: 700, marginRight: '8px' }}>
                        {item.class_code}
                      </span>
                      {item.class_name}
                    </span>

                    <div style={{ display: 'flex', gap: '14px', fontSize: '0.82rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>
                        Raw Score: <b style={{ color: 'var(--text-main)' }}>{rawPct}%</b>
                      </span>
                      {hasCalibration && (
                        <span style={{ color: 'var(--primary-dark)', fontWeight: 600 }}>
                          Calibrated Confidence: {calPct}%
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar Container */}
                  <div style={{
                    height: '6px',
                    background: '#e2e8f0',
                    borderRadius: '3px',
                    overflow: 'hidden',
                    position: 'relative'
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.min(100, Math.max(0, parseFloat(calPct)))}%`,
                      background: getBarColor(item.class_code),
                      borderRadius: '3px',
                      transition: 'width 0.6s ease'
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Model & Calibration Attribution Footer */}
      <div style={{
        marginTop: '18px',
        paddingTop: '12px',
        borderTop: '1px solid var(--border-subtle)',
        fontSize: '0.78rem',
        color: 'var(--text-dim)',
        display: 'flex',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '8px'
      }}>
        <span>Model Architecture: <b>EfficientNet-B0</b></span>
        <span>Calibration: <b>Temperature Scaling (Validation Split Fitted)</b></span>
      </div>
    </div>
  );
};
