import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Case } from '../types';
import { ImageUploadDropzone } from '../components/ImageUploadDropzone';
import { UploadSuccessCard } from '../components/UploadSuccessCard';
import { CameraCaptureModal } from '../components/CameraCaptureModal';
import { 
  Upload, Sparkles, ShieldAlert, AlertCircle, ArrowRight, RefreshCw 
} from 'lucide-react';

export const NewCasePage: React.FC = () => {
  // Case / Patient Metadata Form State
  const [patientRef, setPatientRef] = useState('');
  const [age, setAge] = useState<number | ''>('');
  const [gender, setGender] = useState('FEMALE');
  const [anatomicalSite, setAnatomicalSite] = useState('ARM');
  const [clinicalNotes, setClinicalNotes] = useState('');

  // Image Selection & Preview State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [cameraModalOpen, setCameraModalOpen] = useState(false);

  // Validation & Error States
  const [validationError, setValidationError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  // Submission & Pipeline States
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressText, setUploadProgressText] = useState('Uploading image to secure storage...');
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [createdCase, setCreatedCase] = useState<Case | null>(null);

  // Resource Cleanup: Revoke object URL on unmount or when URL changes
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = (file: File) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const newUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(newUrl);
    setValidationError(null);
    setApiError(null);
  };

  const handleFileRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationError(null);
  };

  const handleResetForm = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationError(null);
    setApiError(null);
    setUploadSuccess(false);
    setCreatedCase(null);
    setIsUploading(false);
    setPatientRef('');
    setAge('');
    setClinicalNotes('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile) {
      setValidationError('Please select or capture a skin lesion image before uploading.');
      return;
    }

    if (isUploading) return;

    setIsUploading(true);
    setApiError(null);
    setValidationError(null);
    setUploadProgressText('Verifying image and uploading via POST /api/v1/cases/upload...');

    try {
      const result = await api.uploadCaseImage(selectedFile, {
        patient_ref: patientRef.trim() || undefined,
        age: age !== '' ? Number(age) : undefined,
        gender: gender || undefined,
        anatomical_site: anatomicalSite,
        clinical_notes: clinicalNotes.trim() || undefined,
      });

      setCreatedCase(result);
      setUploadSuccess(true);
      setIsUploading(false);

    } catch (err: any) {
      setIsUploading(false);
      const rawMessage = err?.message || 'Upload failed.';
      if (rawMessage.toLowerCase().includes('failed to fetch') || rawMessage.toLowerCase().includes('network')) {
        setApiError(
          'Network connection error: Unable to reach the DermAI backend server. Please verify the API server is online and retry.'
        );
      } else {
        setApiError(rawMessage);
      }
    }
  };

  return (
    <div className="container" style={{ maxWidth: '860px', padding: '40px 16px' }}>
      
      {/* Page Title & Breadcrumb */}
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '6px', color: 'var(--text-main)' }}>Clinical Lesion Upload</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Acquire or select a dermatological photograph for quality verification, database registration, and future deep learning screening.
        </p>
      </div>

      {/* Distinct Success State */}
      {uploadSuccess && createdCase ? (
        <UploadSuccessCard caseData={createdCase} onReset={handleResetForm} />
      ) : (
        /* Upload Form */
        <form
          id="case-upload-form"
          onSubmit={handleSubmit}
          className="glass-panel"
          style={{ padding: '36px', borderRadius: 'var(--radius-xl)' }}
        >
          {/* Top API Error Banner */}
          {apiError && (
            <div
              id="upload-api-error"
              style={{
                background: 'var(--danger-bg)',
                border: '1px solid var(--danger-border)',
                borderRadius: 'var(--radius-md)',
                padding: '16px',
                color: 'var(--danger)',
                fontSize: '0.88rem',
                marginBottom: '24px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
              }}
            >
              <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div style={{ flex: 1 }}>
                <b style={{ display: 'block', marginBottom: '3px' }}>Backend Upload Request Failed:</b>
                {apiError}
              </div>
            </div>
          )}

          {/* Section 1: Image Acquisition & Preview Component */}
          <ImageUploadDropzone
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onFileRemove={handleFileRemove}
            onCameraClick={() => setCameraModalOpen(true)}
            disabled={isUploading}
            error={validationError}
            onErrorClear={() => setValidationError(null)}
            onValidationError={(err) => setValidationError(err)}
          />

          {/* Section 2: Patient & Consultation Information */}
          <div style={{ marginBottom: '32px' }}>
            <label className="form-label" style={{ fontSize: '1rem', color: 'var(--text-main)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Patient & Consultation Metadata</span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 400 }}>(De-identified)</span>
            </label>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
              <div className="form-group">
                <label htmlFor="patient-ref-input" className="form-label">Patient / Case Reference</label>
                <input
                  id="patient-ref-input"
                  type="text"
                  className="form-input"
                  value={patientRef}
                  onChange={(e) => setPatientRef(e.target.value)}
                  placeholder="e.g. PT-4981"
                  disabled={isUploading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="anatomical-site-select" className="form-label">Anatomical Body Location</label>
                <select
                  id="anatomical-site-select"
                  className="form-select"
                  value={anatomicalSite}
                  onChange={(e) => setAnatomicalSite(e.target.value)}
                  disabled={isUploading}
                >
                  <option value="ARM">Arm</option>
                  <option value="FOREARM">Forearm</option>
                  <option value="HAND">Hand</option>
                  <option value="CHEST">Chest</option>
                  <option value="BACK">Back</option>
                  <option value="FACE">Face</option>
                  <option value="NECK">Neck</option>
                  <option value="LEG">Leg</option>
                  <option value="FOOT">Foot</option>
                  <option value="ABDOMEN">Abdomen</option>
                  <option value="OTHER">Other / Scalp</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
              <div className="form-group">
                <label htmlFor="patient-age-input" className="form-label">Patient Age (Years, optional)</label>
                <input
                  id="patient-age-input"
                  type="number"
                  min="0"
                  max="120"
                  className="form-input"
                  value={age}
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val === '') {
                      setAge('');
                    } else {
                      const num = Math.max(0, Math.min(120, Number(val)));
                      setAge(isNaN(num) ? '' : num);
                    }
                  }}
                  placeholder="Enter age (e.g. 45)"
                  disabled={isUploading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="patient-gender-select" className="form-label">Gender (optional)</label>
                <select
                  id="patient-gender-select"
                  className="form-select"
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  disabled={isUploading}
                >
                  <option value="FEMALE">Female</option>
                  <option value="MALE">Male</option>
                  <option value="OTHER">Other / Prefer not to say</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="clinical-notes-textarea" className="form-label">Clinical Consultation & Observation Notes</label>
              <textarea
                id="clinical-notes-textarea"
                className="form-textarea"
                rows={3}
                value={clinicalNotes}
                onChange={(e) => setClinicalNotes(e.target.value)}
                placeholder="e.g. Asymmetrical pigmented lesion noted on right forearm. Patient reports slight enlargement over the past 3 months."
                disabled={isUploading}
              />
            </div>
          </div>

          {/* Submission Bar / Distinct Loading State */}
          <div>
            {isUploading ? (
              <div
                id="upload-loading-indicator"
                style={{
                  background: 'var(--primary-subtle)',
                  border: '1px solid var(--primary)',
                  borderRadius: 'var(--radius-md)',
                  padding: '20px',
                  textAlign: 'center',
                  color: 'var(--text-main)',
                }}
              >
                <RefreshCw
                  size={26}
                  color="var(--primary)"
                  style={{ animation: 'spin 1.5s linear infinite', margin: '0 auto 10px auto', display: 'block' }}
                />
                <b style={{ display: 'block', fontSize: '1rem', marginBottom: '4px' }}>
                  Uploading & Registering Case
                </b>
                <span style={{ fontSize: '0.84rem', color: 'var(--primary-dark)' }}>
                  {uploadProgressText}
                </span>
              </div>
            ) : (
              <button
                type="submit"
                id="submit-case-upload-btn"
                className="btn btn-primary btn-lg"
                disabled={!selectedFile || isUploading}
                style={{
                  width: '100%',
                  padding: '16px',
                  opacity: !selectedFile ? 0.6 : 1,
                  cursor: !selectedFile ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                  fontSize: '1.05rem',
                }}
              >
                <Upload size={20} />
                Upload Lesion Photograph
              </button>
            )}
          </div>
        </form>
      )}

      {/* Camera Capture Modal */}
      <CameraCaptureModal
        isOpen={cameraModalOpen}
        onClose={() => setCameraModalOpen(false)}
        onCapture={(file) => handleFileSelect(file)}
      />

    </div>
  );
};
