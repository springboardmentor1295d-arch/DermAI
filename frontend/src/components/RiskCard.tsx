import React from 'react';
import { RiskTier } from '../types';
import { ShieldAlert, AlertCircle, CheckCircle, HelpCircle, Info } from 'lucide-react';

export const RiskCard: React.FC<{
  riskTier: RiskTier;
  rationale?: string;
  predictedClass: string;
  confidenceStr: string;
}> = ({ riskTier, rationale, predictedClass, confidenceStr }) => {
  const getRiskConfig = () => {
    switch (riskTier) {
      case 'high':
        return {
          title: 'High-Priority Diagnostic Category (Malignant Lesion Screen)',
          badgeClass: 'badge-danger',
          border: '1px solid #fca5a5',
          bg: '#fef2f2',
          textColor: '#991b1b',
          icon: <ShieldAlert size={22} color="#b91c1c" aria-hidden="true" />,
          action: 'Predicted category associated with malignant pathology (e.g. Melanoma or Carcinoma). In-person clinical biopsy/dermoscopy evaluation recommended.'
        };
      case 'moderate':
        return {
          title: 'Moderate-Priority Screening Category (Pre-malignant / Keratotic)',
          badgeClass: 'badge-warning',
          border: '1px solid #fde68a',
          bg: '#fffbeb',
          textColor: '#92400e',
          icon: <AlertCircle size={22} color="#b45309" aria-hidden="true" />,
          action: 'Predicted category includes dysplastic or actinic features. Dermatologist evaluation recommended.'
        };
      case 'requires_review':
        return {
          title: 'Ambiguous / Low-Confidence Classification',
          badgeClass: 'badge-warning',
          border: '1px solid #fde68a',
          bg: '#fffbeb',
          textColor: '#92400e',
          icon: <HelpCircle size={22} color="#b45309" aria-hidden="true" />,
          action: 'Model confidence is distributed across multiple classes or image quality is borderline. Expert manual review is required.'
        };
      case 'low':
      default:
        return {
          title: 'Low-Priority Screening Category (Benign Lesion Features)',
          badgeClass: 'badge-success',
          border: '1px solid #bbf7d0',
          bg: '#f0fdf4',
          textColor: '#166534',
          icon: <CheckCircle size={22} color="#15803d" aria-hidden="true" />,
          action: 'Morphological features consistent with benign skin lesion (Nevus or Seborrheic Keratosis). Monitor for changes.'
        };
    }
  };

  const conf = getRiskConfig();

  return (
    <div className="glass-panel" style={{
      padding: '20px',
      borderRadius: 'var(--radius-lg)',
      background: conf.bg,
      border: conf.border,
      marginBottom: '20px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {conf.icon}
          <div>
            <h3 style={{ fontSize: '1.05rem', color: conf.textColor, margin: 0, fontWeight: 700 }}>{conf.title}</h3>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Screening categorization based on predicted {predictedClass} ({confidenceStr} model score)
            </span>
          </div>
        </div>
        <span className={`badge ${conf.badgeClass}`} style={{ fontSize: '0.8rem', padding: '4px 10px' }}>
          {riskTier.toUpperCase().replace('_', ' ')}
        </span>
      </div>

      <div style={{
        background: '#ffffff',
        border: '1px solid var(--border-subtle)',
        padding: '12px 14px',
        borderRadius: 'var(--radius-md)',
        fontSize: '0.85rem',
        color: 'var(--text-main)',
        marginBottom: '10px',
        lineHeight: '1.5'
      }}>
        <b style={{ color: conf.textColor }}>Clinical Context: </b>
        <span>{rationale || conf.action}</span>
      </div>

      <div style={{ fontSize: '0.76rem', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <Info size={14} aria-hidden="true" />
        <span>
          Note: This flag categorizes disease pathology; it does not measure disease severity, tumor depth (Breslow), or clinical stage.
        </span>
      </div>
    </div>
  );
};
