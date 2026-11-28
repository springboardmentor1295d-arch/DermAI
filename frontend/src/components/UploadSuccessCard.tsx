import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Case } from '../types';
import { formatDateTimeIST } from '../utils/dateUtils';
import { 
  CheckCircle2, ArrowRight, PlusCircle, History, Copy, Check, 
  MapPin, User, Calendar, ShieldCheck 
} from 'lucide-react';

export interface UploadSuccessCardProps {
  caseData: Case;
  onReset: () => void;
}

export const UploadSuccessCard: React.FC<UploadSuccessCardProps> = ({ caseData, onReset }) => {
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);

  const handleCopyId = () => {
    navigator.clipboard.writeText(caseData.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const primaryImage = caseData.images && caseData.images.length > 0 ? caseData.images[0] : null;

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return 'N/A';
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  };

  return (
    <div
      id="upload-success-panel"
      className="glass-panel"
      style={{
        padding: '36px',
        borderRadius: 'var(--radius-xl)',
        border: '1px solid var(--success-border)',
        background: '#ffffff',
        boxShadow: 'var(--shadow-md)',
      }}
    >
      {/* Success Header */}
      <div style={{ textAlign: 'center', marginBottom: '28px' }}>
        <div
          style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            background: 'var(--success-bg)',
            border: '2px solid var(--success)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 14px auto',
            color: 'var(--success)',
          }}
        >
          <CheckCircle2 size={32} />
        </div>
        <span className="badge badge-success" style={{ marginBottom: '8px', fontSize: '0.8rem', padding: '4px 12px' }}>
          Upload Succeeded (201 Created)
        </span>
        <h2 style={{ fontSize: '1.6rem', color: 'var(--text-main)', marginTop: '6px', marginBottom: '8px', fontWeight: 800 }}>
          Lesion Image Successfully Uploaded
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '560px', margin: '0 auto' }}>
          The dermatological photograph was validated, persisted to secure storage, and registered to a newly created clinical case record.
        </p>
      </div>

      {/* Case Details Card */}
      <div
        style={{
          background: '#f8fafc',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg)',
          padding: '24px',
          marginBottom: '24px',
        }}
      >
        {/* Case ID Row with Copy */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '12px',
            paddingBottom: '16px',
            borderBottom: '1px solid var(--border-subtle)',
            marginBottom: '18px',
          }}
        >
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Case Identifier (UUID)
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px' }}>
              <code
                id="created-case-id"
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.92rem',
                  color: 'var(--primary-dark)',
                  background: 'var(--primary-subtle)',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  fontWeight: 600,
                  border: '1px solid var(--border-subtle)'
                }}
              >
                {caseData.id}
              </code>
              <button
                type="button"
                id="copy-case-id-btn"
                onClick={handleCopyId}
                className="btn btn-secondary btn-sm"
                title="Copy Case ID to clipboard"
                style={{ padding: '4px 8px', fontSize: '0.75rem' }}
              >
                {copied ? <Check size={14} color="var(--success)" /> : <Copy size={14} />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Pipeline Status
            </span>
            <div style={{ marginTop: '4px', display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
              <span className="badge badge-warning">
                Status: {caseData.status.toUpperCase()}
              </span>
              <span className="badge badge-success">
                Storage: UPLOADED
              </span>
            </div>
          </div>
        </div>

        {/* Metadata Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'block' }}>Patient Reference</span>
            <span id="created-patient-ref" style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: 600 }}>
              {caseData.patient_ref || 'None'}
            </span>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'block' }}>Anatomical Site</span>
            <span style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <MapPin size={14} color="var(--primary)" />
              {caseData.anatomical_site || 'Not Specified'}
            </span>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'block' }}>Patient Age & Gender</span>
            <span style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <User size={14} color="var(--primary)" />
              Age: {caseData.age !== null && caseData.age !== undefined ? `${caseData.age} years` : 'Not provided'} | {caseData.gender || 'Not provided'}
            </span>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'block' }}>Registered Timestamp (IST)</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Calendar size={14} />
              {formatDateTimeIST(caseData.created_at)}
            </span>
          </div>
        </div>

        {/* Uploaded Lesion Image Details */}
        {primaryImage && (
          <div
            style={{
              marginTop: '20px',
              paddingTop: '16px',
              borderTop: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              gap: '16px',
            }}
          >
            <div
              style={{
                width: '60px',
                height: '60px',
                borderRadius: '8px',
                overflow: 'hidden',
                background: '#ffffff',
                border: '1px solid var(--border-subtle)',
                flexShrink: 0,
              }}
            >
              <img
                src={primaryImage.image_url}
                alt="Uploaded Lesion Thumbnail"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)', display: 'block' }}>
                {primaryImage.original_filename}
              </span>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', gap: '12px', marginTop: '3px' }}>
                <span>Size: <b>{formatFileSize(primaryImage.file_size_bytes)}</b></span>
                <span>MIME: <b>{primaryImage.mime_type}</b></span>
                {primaryImage.width && primaryImage.height && (
                  <span>Resolution: <b>{primaryImage.width} × {primaryImage.height} px</b></span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Clinical Notes if provided */}
        {caseData.clinical_notes && (
          <div style={{ marginTop: '16px', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'block', marginBottom: '4px' }}>
              Clinical Consultation Notes:
            </span>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-main)', margin: 0, fontStyle: 'italic' }}>
              "{caseData.clinical_notes}"
            </p>
          </div>
        )}
      </div>

      {/* Contract Verification Notice */}
      <div
        style={{
          background: 'var(--primary-subtle)',
          border: '1px solid rgba(13, 148, 136, 0.3)',
          borderRadius: 'var(--radius-md)',
          padding: '14px 18px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '12px',
        }}
      >
        <ShieldCheck size={20} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
          <b style={{ color: 'var(--text-main)' }}>Decoupled ML Architecture Notice:</b> In strict accordance with the backend API contract (`POST /api/v1/cases/upload`), the image upload transaction is complete and atomic. Machine learning inference and Grad-CAM explainability are decoupled and have <i>not</i> been run automatically. You can initiate analysis below at your discretion.
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            type="button"
            id="upload-another-btn"
            className="btn btn-secondary"
            onClick={onReset}
          >
            <PlusCircle size={16} /> Upload Another Case
          </button>
          <button
            type="button"
            id="view-history-btn"
            className="btn btn-secondary"
            onClick={() => navigate('/history')}
          >
            <History size={16} /> View Case History
          </button>
        </div>

        <button
          type="button"
          id="proceed-to-analysis-btn"
          className="btn btn-primary btn-lg"
          onClick={() => navigate(`/cases/${caseData.id}`)}
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          Proceed to ML Analysis & Explainability <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
};
