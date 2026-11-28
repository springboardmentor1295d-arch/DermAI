import React, { useState } from 'react';
import { Case } from '../types';
import { GitCompare, Calendar, ArrowRight } from 'lucide-react';
import { formatDateTimeIST, formatDateIST } from '../utils/dateUtils';

interface ProgressionComparatorProps {
  cases: Case[];
}

export const ProgressionComparator: React.FC<ProgressionComparatorProps> = ({ cases }) => {
  const completedCases = cases.filter(c => c.images.length > 0 && c.images[0].prediction);
  
  const [case1Id, setCase1Id] = useState<string>(completedCases[0]?.id || '');
  const [case2Id, setCase2Id] = useState<string>(completedCases[1]?.id || completedCases[0]?.id || '');

  if (completedCases.length < 2) {
    return (
      <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <GitCompare size={28} style={{ margin: '0 auto 8px auto', display: 'block', color: 'var(--text-dim)' }} />
        <h4 style={{ color: 'var(--text-main)', marginBottom: '6px' }}>Longitudinal Progression Comparator</h4>
        <p style={{ fontSize: '0.82rem' }}>
          Requires at least two analyzed cases to perform longitudinal side-by-side progression analysis.
        </p>
      </div>
    );
  }

  const case1 = completedCases.find(c => c.id === case1Id);
  const case2 = completedCases.find(c => c.id === case2Id);

  return (
    <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <GitCompare size={22} color="var(--primary)" />
        <div>
          <h3 style={{ margin: 0, color: 'var(--text-main)' }}>Longitudinal Lesion Progression Comparison</h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Compare lesion changes, risk level shifts, and attention heatmaps across separate consultation dates.
          </span>
        </div>
      </div>

      {/* Selectors */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '16px', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <label className="form-label">Baseline Analysis (Session A)</label>
          <select 
            value={case1Id} 
            onChange={(e) => setCase1Id(e.target.value)}
            className="form-select"
          >
            {completedCases.map(c => (
              <option key={c.id} value={c.id}>
                {c.patient_ref || c.id.slice(0, 8)} — {formatDateIST(c.images[0]?.prediction?.created_at || c.created_at)} ({c.images[0]?.prediction?.predicted_class})
              </option>
            ))}
          </select>
        </div>

        <div style={{ paddingTop: '22px', color: 'var(--text-dim)' }}>
          <ArrowRight size={20} />
        </div>

        <div>
          <label className="form-label">Follow-Up Analysis (Session B)</label>
          <select 
            value={case2Id} 
            onChange={(e) => setCase2Id(e.target.value)}
            className="form-select"
          >
            {completedCases.map(c => (
              <option key={c.id} value={c.id}>
                {c.patient_ref || c.id.slice(0, 8)} — {formatDateIST(c.images[0]?.prediction?.created_at || c.created_at)} ({c.images[0]?.prediction?.predicted_class})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Comparison Grid */}
      {case1 && case2 && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '18px' }}>
            {/* Case 1 Card */}
            <div style={{ background: '#f8fafc', padding: '18px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>Baseline: {case1.patient_ref || case1.id.slice(0, 8)}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <Calendar size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  {formatDateTimeIST(case1.created_at)}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: case1.images[0]?.prediction?.gradcam_url ? '1fr 1fr' : '1fr', gap: '10px', marginBottom: '14px' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginBottom: '4px' }}>Original Image</div>
                  <img 
                    src={case1.images[0]?.image_url} 
                    alt="Baseline Original" 
                    style={{ width: '100%', height: '180px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }} 
                  />
                </div>
                {case1.images[0]?.prediction?.gradcam_url && (
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginBottom: '4px' }}>Grad-CAM Overlay</div>
                    <img 
                      src={case1.images[0]?.prediction?.gradcam_url} 
                      alt="Baseline Grad-CAM" 
                      style={{ width: '100%', height: '180px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }} 
                    />
                  </div>
                )}
              </div>
              <div style={{ fontSize: '0.85rem', lineHeight: '1.6' }}>
                <div><b>Diagnosis:</b> <span style={{ color: 'var(--primary-dark)', fontWeight: 700 }}>{case1.images[0]?.prediction?.predicted_class} ({case1.images[0]?.prediction?.class_name})</span></div>
                <div><b>Raw Score:</b> {case1.images[0]?.prediction?.raw_model_score !== undefined ? `${(case1.images[0]?.prediction.raw_model_score * 100).toFixed(1)}%` : case1.images[0]?.prediction?.confidence_percentage_str}</div>
                <div><b>Calibrated Confidence:</b> <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{case1.images[0]?.prediction?.calibrated_confidence !== undefined ? `${(case1.images[0]?.prediction.calibrated_confidence * 100).toFixed(1)}%` : case1.images[0]?.prediction?.confidence_percentage_str}</span></div>
                <div><b>Risk Priority:</b> <span className={`badge ${case1.images[0]?.prediction?.risk_tier === 'high' ? 'badge-danger' : case1.images[0]?.prediction?.risk_tier === 'moderate' ? 'badge-warning' : 'badge-success'}`}>{case1.images[0]?.prediction?.risk_tier.toUpperCase()}</span></div>
                <div><b>Patient Age:</b> {case1.age !== null && case1.age !== undefined ? `${case1.age} years` : 'Not provided'}</div>
                <div><b>Anatomical Site:</b> {case1.anatomical_site || 'Not specified'}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>Model: {case1.images[0]?.prediction?.model_version || 'EfficientNet-B0'}</div>
              </div>
            </div>

            {/* Case 2 Card */}
            <div style={{ background: '#f8fafc', padding: '18px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>Follow-Up: {case2.patient_ref || case2.id.slice(0, 8)}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <Calendar size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  {formatDateTimeIST(case2.created_at)}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: case2.images[0]?.prediction?.gradcam_url ? '1fr 1fr' : '1fr', gap: '10px', marginBottom: '14px' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginBottom: '4px' }}>Original Image</div>
                  <img 
                    src={case2.images[0]?.image_url} 
                    alt="Follow-Up Original" 
                    style={{ width: '100%', height: '180px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }} 
                  />
                </div>
                {case2.images[0]?.prediction?.gradcam_url && (
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginBottom: '4px' }}>Grad-CAM Overlay</div>
                    <img 
                      src={case2.images[0]?.prediction?.gradcam_url} 
                      alt="Follow-Up Grad-CAM" 
                      style={{ width: '100%', height: '180px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }} 
                    />
                  </div>
                )}
              </div>
              <div style={{ fontSize: '0.85rem', lineHeight: '1.6' }}>
                <div><b>Diagnosis:</b> <span style={{ color: 'var(--primary-dark)', fontWeight: 700 }}>{case2.images[0]?.prediction?.predicted_class} ({case2.images[0]?.prediction?.class_name})</span></div>
                <div><b>Raw Score:</b> {case2.images[0]?.prediction?.raw_model_score !== undefined ? `${(case2.images[0]?.prediction.raw_model_score * 100).toFixed(1)}%` : case2.images[0]?.prediction?.confidence_percentage_str}</div>
                <div><b>Calibrated Confidence:</b> <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{case2.images[0]?.prediction?.calibrated_confidence !== undefined ? `${(case2.images[0]?.prediction.calibrated_confidence * 100).toFixed(1)}%` : case2.images[0]?.prediction?.confidence_percentage_str}</span></div>
                <div><b>Risk Priority:</b> <span className={`badge ${case2.images[0]?.prediction?.risk_tier === 'high' ? 'badge-danger' : case2.images[0]?.prediction?.risk_tier === 'moderate' ? 'badge-warning' : 'badge-success'}`}>{case2.images[0]?.prediction?.risk_tier.toUpperCase()}</span></div>
                <div><b>Patient Age:</b> {case2.age !== null && case2.age !== undefined ? `${case2.age} years` : 'Not provided'}</div>
                <div><b>Anatomical Site:</b> {case2.anatomical_site || 'Not specified'}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>Model: {case2.images[0]?.prediction?.model_version || 'EfficientNet-B0'}</div>
              </div>
            </div>
          </div>

          {/* Clinical Non-Progression Limitation Notice */}
          <div style={{ background: 'var(--warning-bg)', border: '1px solid var(--warning-border)', borderRadius: 'var(--radius-md)', padding: '12px 16px', fontSize: '0.82rem', color: 'var(--warning)', lineHeight: '1.5' }}>
            <strong>Comparative Analysis Limitation:</strong> Differences in model confidence scores across visits represent independent statistical evaluations, not validated clinical lesion progression or deterioration.
          </div>
        </>
      )}
    </div>
  );
};
