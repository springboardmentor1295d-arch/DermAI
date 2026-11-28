import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../services/api';
import { Case } from '../types';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { RiskCard } from '../components/RiskCard';
import { QualityBadge } from '../components/QualityBadge';
import { ImageComparisonView } from '../components/ImageComparisonView';
import { TopNChart } from '../components/TopNChart';
import { formatDateTimeIST } from '../utils/dateUtils';
import { useNavigate } from 'react-router-dom';
import {
  FileDown, ArrowLeft, Calendar, MapPin, User, Save,
  CheckCircle2, Sparkles, AlertCircle, Trash2, AlertTriangle
} from 'lucide-react';

export const AnalysisResultPage: React.FC = () => {
  const { caseId } = useParams<{ caseId: string }>();
  const navigate = useNavigate();
  const [caseData, setCaseData] = useState<Case | null>(null);
  const [loading, setLoading] = useState(true);
  const [notes, setNotes] = useState('');
  const [savingNotes, setSavingNotes] = useState(false);
  const [notesSaved, setNotesSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [downloadingCsv, setDownloadingCsv] = useState(false);
  const [downloadingJson, setDownloadingJson] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletingCase, setDeletingCase] = useState(false);

  useEffect(() => {
    if (caseId) {
      api.getCase(caseId)
        .then((c) => {
          setCaseData(c);
          setNotes(c.clinical_notes || '');
        })
        .catch((err) => setError(err.message || 'Failed to load case'))
        .finally(() => setLoading(false));
    }
  }, [caseId]);

  const handleRunInference = async () => {
    if (!caseId) return;
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const updatedCase = await api.analyzeCaseImage(caseId);
      setCaseData(updatedCase);
    } catch (err: any) {
      setAnalyzeError(err.message || 'Inference failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSaveNotes = async () => {
    if (!caseId) return;
    setSavingNotes(true);
    try {
      const updated = await api.updateCaseNotes(caseId, notes);
      setCaseData(updated);
      setNotesSaved(true);
      setTimeout(() => setNotesSaved(false), 2500);
    } catch {
      alert('Failed to save notes');
    } finally {
      setSavingNotes(false);
    }
  };

  const handleDeleteCase = async () => {
    if (!caseData) return;
    setDeletingCase(true);
    try {
      await api.deleteCase(caseData.id);
      navigate('/history');
    } catch (err: any) {
      setDownloadError(err.message || 'Failed to delete analysis record');
      setShowDeleteModal(false);
      setDeletingCase(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!caseId) return;
    setDownloadingPdf(true);
    setDownloadError(null);
    try {
      await api.downloadCasePdf(caseId, `DermAI_Report_${caseData?.patient_ref || caseId.slice(0, 8)}.pdf`);
    } catch (err: any) {
      setDownloadError(err.message || 'Failed to download PDF report');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleDownloadCsv = async () => {
    if (!caseId) return;
    setDownloadingCsv(true);
    setDownloadError(null);
    try {
      await api.downloadCaseCsv(caseId, `DermAI_Analysis_${caseData?.patient_ref || caseId.slice(0, 8)}.csv`);
    } catch (err: any) {
      setDownloadError(err.message || 'Failed to export CSV report');
    } finally {
      setDownloadingCsv(false);
    }
  };

  const handleDownloadJson = async () => {
    if (!caseId) return;
    setDownloadingJson(true);
    setDownloadError(null);
    try {
      await api.downloadCaseJson(caseId, `DermAI_Analysis_${caseData?.patient_ref || caseId.slice(0, 8)}.json`);
    } catch (err: any) {
      setDownloadError(err.message || 'Failed to download JSON report');
    } finally {
      setDownloadingJson(false);
    }
  };

  if (loading) {
    return (
      <div className="container" style={{ padding: '80px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <Sparkles size={32} color="var(--primary)" style={{ animation: 'spin 2s linear infinite', margin: '0 auto 12px auto' }} />
        <p>Loading dermatological analysis report...</p>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="container" style={{ padding: '60px 16px', textAlign: 'center' }}>
        <AlertCircle size={40} color="var(--danger)" style={{ margin: '0 auto 12px auto' }} />
        <h3 style={{ color: 'var(--text-main)', marginBottom: '8px' }}>Unable to Load Analysis</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '20px' }}>{error || 'Case not found'}</p>
        <Link to="/dashboard" className="btn btn-secondary">
          <ArrowLeft size={16} /> Return to Dashboard
        </Link>
      </div>
    );
  }

  const primaryImage = caseData.images[0];
  const prediction = primaryImage?.prediction;
  const quality = primaryImage?.quality_check;

  return (
    <div className="container" style={{ maxWidth: '1040px', padding: '36px 16px' }}>

      {/* Top Navigation & Actions */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <Link to="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.85rem' }}>
          <ArrowLeft size={16} /> Back to Dashboard
        </Link>

        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button onClick={handleDownloadCsv} className="btn btn-secondary btn-sm" disabled={downloadingCsv}>
            <FileDown size={15} /> {downloadingCsv ? 'Exporting CSV...' : 'Export CSV'}
          </button>
          <button onClick={handleDownloadJson} className="btn btn-secondary btn-sm" disabled={downloadingJson}>
            <FileDown size={15} /> {downloadingJson ? 'Exporting JSON...' : 'Export JSON'}
          </button>
          <button onClick={handleDownloadPdf} className="btn btn-primary btn-sm" disabled={downloadingPdf}>
            <FileDown size={15} /> {downloadingPdf ? 'Generating PDF...' : 'Download PDF Report'}
          </button>
          <button
            type="button"
            onClick={() => setShowDeleteModal(true)}
            className="btn btn-sm"
            style={{ background: 'var(--danger-bg)', color: 'var(--danger)', border: '1px solid var(--danger-border)' }}
            title="Permanently Delete Case"
          >
            <Trash2 size={15} /> Delete Case
          </button>
        </div>
      </div>

      {downloadError && (
        <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-md)', padding: '10px 14px', marginBottom: '16px', color: 'var(--danger)', fontSize: '0.85rem' }}>
          {downloadError}
        </div>
      )}

      {/* Case Header Card */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '14px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.8rem', margin: 0, color: 'var(--text-main)' }}>
                Case {caseData.patient_ref || caseData.id.slice(0, 8)}
              </h1>
              <span className={`badge ${caseData.status === 'completed' ? 'badge-success' : 'badge-warning'}`}>
                {caseData.status.toUpperCase()}
              </span>
            </div>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', display: 'block', marginTop: '4px' }}>
              Case Identifier: {caseData.id}
            </span>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Analysis Timestamp (IST):</span>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 600 }}>
              {formatDateTimeIST(prediction?.created_at || caseData.created_at)}
            </div>
          </div>
        </div>

        {/* Metadata pills */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', fontSize: '0.85rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <MapPin size={15} color="var(--primary)" />
            <span>Site: <b>{caseData.anatomical_site || 'Not specified'}</b></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <User size={15} color="var(--primary)" />
            <span>Patient Age: <b>{caseData.age !== null && caseData.age !== undefined ? `${caseData.age} years` : 'Not provided'}</b> | Gender: <b>{caseData.gender || 'Not provided'}</b></span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={15} color="var(--accent)" />
            <span>Model: <b>{prediction?.model_version || 'Active Production Model'}</b></span>
          </div>
        </div>
      </div>

      {prediction && <DisclaimerBanner />}

      {/* Pending Inference Section if prediction is null */}
      {!prediction && primaryImage && (
        <div
          id="pending-analysis-card"
          className="glass-panel"
          style={{
            padding: '32px',
            borderRadius: 'var(--radius-lg)',
            marginBottom: '24px',
            textAlign: 'center',
            border: '1px solid rgba(13, 148, 136, 0.4)',
            background: 'var(--primary-subtle)',
          }}
        >
          <Sparkles size={36} color="var(--primary)" style={{ margin: '0 auto 12px auto' }} />
          <h3 style={{ fontSize: '1.4rem', color: 'var(--text-main)', marginBottom: '8px' }}>
            Lesion Image Uploaded — Ready for Deep Learning Analysis
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: '600px', margin: '0 auto 20px auto' }}>
            This case was created via the upload contract. Deep learning inference (active production classifier), image quality assessment, and Grad-CAM explainability can now be executed on demand.
          </p>

          {analyzing ? (
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '10px', color: 'var(--primary-dark)' }}>
              <Sparkles size={18} style={{ animation: 'spin 1.5s linear infinite' }} />
              <span>Executing PyTorch inference & Grad-CAM pipeline...</span>
            </div>
          ) : (
            <button
              id="run-inference-btn"
              type="button"
              className="btn btn-primary btn-lg"
              onClick={handleRunInference}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
            >
              <Sparkles size={18} /> Execute Deep Learning Analysis & Grad-CAM
            </button>
          )}

          {analyzeError && (
            <div style={{ color: 'var(--danger)', fontSize: '0.85rem', marginTop: '12px' }}>
              {analyzeError}
            </div>
          )}
        </div>
      )}

      {/* Clinical Risk Card */}
      {prediction && (
        <RiskCard
          riskTier={prediction.risk_tier}
          rationale={prediction.risk_rationale}
          predictedClass={prediction.predicted_class}
          confidenceStr={prediction.confidence_percentage_str}
        />
      )}

      {/* Multi-Panel Image Comparison View */}
      {primaryImage && (
        <ImageComparisonView
          originalUrl={primaryImage.image_url}
          preprocessedUrl={primaryImage.preprocessed_image_url || prediction?.preprocessed_image_url}
          gradcamUrl={prediction?.gradcam_url}
          gradcamHeatmapUrl={prediction?.gradcam_heatmap_url}
          roiMaskUrl={prediction?.roi_mask_url}
          predictedClass={prediction?.predicted_class || 'Lesion'}
        />
      )}

      {/* Two Column Grid: Top-N Chart & Image Quality */}
      <div style={{ display: 'grid', gridTemplateColumns: prediction ? 'minmax(0, 1.4fr) minmax(0, 1fr)' : '1fr', gap: '20px', marginBottom: '24px' }}>
        {prediction?.top_n_probabilities && (
          <TopNChart items={prediction.top_n_probabilities} />
        )}

        <div>
          {quality && <QualityBadge quality={quality} />}

          {/* Clinical Findings Summary */}
          {prediction && (
            <div className="glass-panel" style={{ padding: '20px', borderRadius: 'var(--radius-lg)', marginTop: '16px' }}>
              <h4 style={{ color: 'var(--text-main)', fontSize: '0.95rem', marginBottom: '10px', fontWeight: 700 }}>
                Diagnostic Reference & Calibration Details
              </h4>
              <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                <div>
                  <b>Primary Condition: </b>
                  <span style={{ color: 'var(--primary-dark)', fontWeight: 700 }}>
                    {prediction.predicted_class} ({prediction.class_name})
                  </span>
                </div>
                <div style={{ marginTop: '4px' }}>
                  <b>Raw Softmax Score: </b>
                  <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>
                    {prediction.raw_model_score !== undefined ? `${(prediction.raw_model_score * 100).toFixed(1)}%` : prediction.confidence_percentage_str}
                  </span>
                </div>
                {prediction.calibrated_confidence !== undefined && (
                  <div style={{ marginTop: '4px' }}>
                    <b>Calibrated Confidence: </b>
                    <span style={{ color: 'var(--primary)', fontWeight: 700 }}>
                      {(prediction.calibrated_confidence * 100).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginLeft: '6px' }}>
                      (Temperature Scaled on Validation Set)
                    </span>
                  </div>
                )}
                {((prediction.calibrated_confidence ?? prediction.confidence_score) < 0.60) && (
                  <div style={{ marginTop: '8px', padding: '8px 10px', background: 'var(--warning-bg)', border: '1px solid var(--warning-border)', borderRadius: 'var(--radius-sm)', color: 'var(--warning)', fontSize: '0.78rem' }}>
                    <b>Low Confidence / Uncertainty Notice:</b> The model is uncertain across multiple conditions. Please inspect the Top-3 differential predictions and consult a dermatologist.
                  </div>
                )}
                <div style={{ marginTop: '8px', fontSize: '0.78rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-subtle)', paddingTop: '8px' }}>
                  Evaluated with EfficientNet-B0 against the six PAD-UFES-20 ground-truth classes: BCC, ACK, NEV, SEK, SCC, MEL.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Clinical Consultation Notes Section */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)', margin: 0 }}>
            Clinical Consultation & Specialist Notes
          </h3>
          {notesSaved && (
            <span className="badge badge-success" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <CheckCircle2 size={12} /> Notes Saved
            </span>
          )}
        </div>
        <textarea
          className="form-textarea"
          rows={3}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Record clinical observations, dermoscopic findings, biopsy recommendations, or follow-up timelines..."
        />
        <div style={{ textAlign: 'right', marginTop: '10px' }}>
          <button
            onClick={handleSaveNotes}
            className="btn btn-secondary btn-sm"
            disabled={savingNotes}
          >
            <Save size={14} /> {savingNotes ? 'Saving...' : 'Save Consultation Notes'}
          </button>
        </div>
      </div>

      {/* Permanent Deletion Confirmation Modal */}
      {showDeleteModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(15, 23, 42, 0.65)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '16px'
        }}>
          <div className="glass-panel" style={{
            maxWidth: '480px',
            width: '100%',
            padding: '24px',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--danger-border)',
            background: '#ffffff',
            boxShadow: 'var(--shadow-lg)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--danger)', marginBottom: '16px' }}>
              <AlertTriangle size={28} />
              <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-main)', fontWeight: 700 }}>Confirm Permanent Deletion</h3>
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: '1.5', marginBottom: '16px' }}>
              Are you sure you want to permanently delete case <strong style={{ color: 'var(--text-main)' }}>{caseData.patient_ref || caseData.id.slice(0, 8)}</strong>?
            </p>

            <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-sm)', padding: '12px', marginBottom: '20px', fontSize: '0.82rem', color: 'var(--danger)' }}>
              <strong>Permanent Action Warning:</strong> This operation deletes the case record, all associated image uploads, preprocessed files, and Grad-CAM artifacts permanently from the database and storage. It cannot be recovered.
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setShowDeleteModal(false)}
                disabled={deletingCase}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-sm btn-danger"
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                onClick={handleDeleteCase}
                disabled={deletingCase}
              >
                <Trash2 size={14} /> {deletingCase ? 'Deleting...' : 'Confirm Permanent Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
