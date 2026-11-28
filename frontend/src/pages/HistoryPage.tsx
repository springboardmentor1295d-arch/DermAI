import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { Case } from '../types';
import { ProgressionComparator } from '../components/ProgressionComparator';
import { formatDateTimeIST, getISTExportDateString } from '../utils/dateUtils';
import { 
  History, Search, Filter, Calendar, MapPin, Eye, ArrowRight,
  Trash2, FileDown, AlertTriangle, CheckCircle2, X
} from 'lucide-react';

export const HistoryPage: React.FC = () => {
  const [cases, setCases] = useState<Case[]>([]);
  const [filteredCases, setFilteredCases] = useState<Case[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRisk, setSelectedRisk] = useState('all');
  const [selectedClass, setSelectedClass] = useState('all');
  const [loading, setLoading] = useState(true);
  const [exportingCsv, setExportingCsv] = useState(false);
  const [deleteModalCase, setDeleteModalCase] = useState<Case | null>(null);
  const [deletingCase, setDeletingCase] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchCases = () => {
    setLoading(true);
    api.getCases()
      .then((data) => {
        setCases(data);
        setFilteredCases(data);
      })
      .catch((err: any) => setErrorMessage(err.message || 'Failed to fetch cases'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchCases();
  }, []);

  useEffect(() => {
    let result = cases;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(c => 
        (c.patient_ref && c.patient_ref.toLowerCase().includes(q)) ||
        (c.anatomical_site && c.anatomical_site.toLowerCase().includes(q)) ||
        (c.id.toLowerCase().includes(q))
      );
    }
    if (selectedRisk !== 'all') {
      result = result.filter(c => c.images[0]?.prediction?.risk_tier === selectedRisk);
    }
    if (selectedClass !== 'all') {
      result = result.filter(c => c.images[0]?.prediction?.predicted_class === selectedClass);
    }
    setFilteredCases(result);
  }, [searchQuery, selectedRisk, selectedClass, cases]);

  const handleExportAllCsv = async () => {
    setExportingCsv(true);
    setErrorMessage(null);
    try {
      await api.downloadAllHistoryCsv(`DermAI_History_Export_${getISTExportDateString()}.csv`);
      setSuccessMessage('History exported as CSV successfully.');
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to export history CSV');
    } finally {
      setExportingCsv(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteModalCase) return;
    setDeletingCase(true);
    setErrorMessage(null);
    try {
      await api.deleteCase(deleteModalCase.id);
      const deletedRef = deleteModalCase.patient_ref || deleteModalCase.id.slice(0, 8);
      setCases(prev => prev.filter(c => c.id !== deleteModalCase.id));
      setFilteredCases(prev => prev.filter(c => c.id !== deleteModalCase.id));
      setSuccessMessage(`Case ${deletedRef} permanently deleted from database and filesystem.`);
      setTimeout(() => setSuccessMessage(null), 5000);
      setDeleteModalCase(null);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to delete case record');
    } finally {
      setDeletingCase(false);
    }
  };

  const getRiskBadge = (risk?: string) => {
    switch (risk) {
      case 'high': return <span className="badge badge-danger">High Risk</span>;
      case 'moderate': return <span className="badge badge-warning">Moderate</span>;
      case 'requires_review': return <span className="badge badge-warning">Review Flag</span>;
      case 'low': return <span className="badge badge-success">Low Risk</span>;
      default: return <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>Pending</span>;
    }
  };

  return (
    <div className="container" style={{ padding: '36px 16px' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', marginBottom: '6px', color: 'var(--text-main)' }}>Case History & Follow-Up Tracking</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Browse past dermatological screening cases, track longitudinal lesion changes, and compare follow-up consultations.
          </p>
        </div>

        <div>
          <button 
            onClick={handleExportAllCsv} 
            className="btn btn-secondary" 
            disabled={exportingCsv || cases.length === 0}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <FileDown size={16} /> {exportingCsv ? 'Exporting CSV...' : 'Export All History as CSV'}
          </button>
        </div>
      </div>

      {/* Alert Banners */}
      {successMessage && (
        <div style={{ background: 'var(--success-bg)', border: '1px solid var(--success-border)', borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '20px', color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <CheckCircle2 size={18} />
          <span>{successMessage}</span>
        </div>
      )}

      {errorMessage && (
        <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '20px', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertTriangle size={18} />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Embedded Progression Comparator */}
      <div style={{ marginBottom: '32px' }}>
        <ProgressionComparator cases={cases} />
      </div>

      {/* Search & Filter Bar */}
      <div className="glass-panel" style={{ padding: '16px 20px', borderRadius: 'var(--radius-lg)', marginBottom: '20px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(200px, 1.5fr) 1fr 1fr', gap: '14px', alignItems: 'center' }}>
          
          <div style={{ position: 'relative' }}>
            <Search size={16} color="var(--text-dim)" style={{ position: 'absolute', left: '12px', top: '13px' }} />
            <input
              type="text"
              placeholder="Search by patient ref, site, ID..."
              className="form-input"
              style={{ paddingLeft: '36px' }}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div>
            <select
              className="form-select"
              value={selectedRisk}
              onChange={(e) => setSelectedRisk(e.target.value)}
            >
              <option value="all">All Risk Levels</option>
              <option value="high">High Risk</option>
              <option value="moderate">Moderate Risk</option>
              <option value="requires_review">Requires Review</option>
              <option value="low">Low Risk</option>
            </select>
          </div>

          <div>
            <select
              className="form-select"
              value={selectedClass}
              onChange={(e) => setSelectedClass(e.target.value)}
            >
              <option value="all">All Diagnostic Classes</option>
              <option value="BCC">BCC (Basal Cell Carcinoma)</option>
              <option value="ACK">ACK (Actinic Keratosis)</option>
              <option value="NEV">NEV (Melanocytic Nevus)</option>
              <option value="SEK">SEK (Seborrheic Keratosis)</option>
              <option value="SCC">SCC (Squamous Cell Carcinoma)</option>
              <option value="MEL">MEL (Melanoma)</option>
            </select>
          </div>

        </div>
      </div>

      {/* Cases List Table */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', margin: 0, fontWeight: 700 }}>
            Analysis Records ({filteredCases.length} of {cases.length})
          </h3>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading case records...
          </div>
        ) : filteredCases.length === 0 ? (
          <div style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No cases match the selected search or filter criteria.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left', background: '#f8fafc' }}>
                  <th style={{ padding: '10px 14px' }}>Thumbnail</th>
                  <th style={{ padding: '10px 14px' }}>Patient Ref</th>
                  <th style={{ padding: '10px 14px' }}>Patient Age</th>
                  <th style={{ padding: '10px 14px' }}>Anatomical Site</th>
                  <th style={{ padding: '10px 14px' }}>Primary Diagnosis</th>
                  <th style={{ padding: '10px 14px' }}>Confidence</th>
                  <th style={{ padding: '10px 14px' }}>Risk Priority</th>
                  <th style={{ padding: '10px 14px' }}>Date (IST)</th>
                  <th style={{ padding: '10px 14px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredCases.map((c) => {
                  const img = c.images[0];
                  const pred = img?.prediction;
                  return (
                    <tr key={c.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '10px 14px' }}>
                        {img ? (
                          <img
                            src={img.image_url}
                            alt="Lesion"
                            style={{ width: '44px', height: '44px', objectFit: 'cover', borderRadius: '6px', background: '#f1f5f9', border: '1px solid var(--border-subtle)' }}
                          />
                        ) : (
                          <div style={{ width: '44px', height: '44px', background: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-subtle)' }} />
                        )}
                      </td>
                      <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-main)' }}>
                        {c.patient_ref || c.id.slice(0, 8)}
                      </td>
                      <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>
                        {c.age !== null && c.age !== undefined ? `${c.age} yrs` : 'Not provided'}
                      </td>
                      <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>
                        {c.anatomical_site || 'Unspecified'}
                      </td>
                      <td style={{ padding: '10px 14px' }}>
                        {pred ? (
                          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>
                            {pred.predicted_class}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)' }}>—</span>
                        )}
                      </td>
                      <td style={{ padding: '10px 14px', color: 'var(--primary-dark)', fontWeight: 600 }}>
                        {pred?.confidence_percentage_str || '—'}
                      </td>
                      <td style={{ padding: '10px 14px' }}>
                        {getRiskBadge(pred?.risk_tier)}
                      </td>
                      <td style={{ padding: '10px 14px', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                        {formatDateTimeIST(pred?.created_at || c.created_at)}
                      </td>
                      <td style={{ padding: '10px 14px', textAlign: 'right' }}>
                        <div style={{ display: 'inline-flex', gap: '8px', alignItems: 'center' }}>
                          <Link to={`/cases/${c.id}`} className="btn btn-secondary btn-sm" style={{ padding: '4px 10px' }} title="View Analysis">
                            <Eye size={14} /> View
                          </Link>
                          <button
                            type="button"
                            onClick={() => setDeleteModalCase(c)}
                            className="btn btn-sm"
                            style={{ padding: '4px 8px', background: 'var(--danger-bg)', color: 'var(--danger)', border: '1px solid var(--danger-border)' }}
                            title="Permanently Delete Case"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Permanent Deletion Confirmation Modal */}
      {deleteModalCase && (
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
              Are you sure you want to permanently delete case <strong style={{ color: 'var(--text-main)' }}>{deleteModalCase.patient_ref || deleteModalCase.id.slice(0, 8)}</strong>?
            </p>

            <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-sm)', padding: '12px', marginBottom: '20px', fontSize: '0.82rem', color: 'var(--danger)' }}>
              <strong>Permanent Action Warning:</strong> This operation deletes the case record, all associated image uploads, preprocessed files, and Grad-CAM artifacts permanently from the database and storage. It cannot be recovered.
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setDeleteModalCase(null)}
                disabled={deletingCase}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-sm btn-danger"
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                onClick={handleConfirmDelete}
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
