import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import { Case, DashboardSummary } from '../types';
import { 
  Activity, AlertTriangle, CheckCircle, Clock, PlusCircle, 
  ArrowRight, Shield, Layers, FileText, ChevronRight 
} from 'lucide-react';
import { formatDateTimeIST } from '../utils/dateUtils';

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [cases, setCases] = useState<Case[]>([]);
  const [evalSummary, setEvalSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getDashboardSummary().then(setSummary).catch(() => {}),
      api.getCases().then(setCases).catch(() => {}),
      api.getModelEvaluationSummary().then(setEvalSummary).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed': return <span className="badge badge-success">Completed</span>;
      case 'requires_review': return <span className="badge badge-warning">Review Needed</span>;
      case 'processing': return <span className="badge badge-info">Processing</span>;
      case 'failed': return <span className="badge badge-danger">Failed</span>;
      default: return <span className="badge badge-info">Pending</span>;
    }
  };

  const getRiskBadge = (risk?: string) => {
    switch (risk) {
      case 'high': return <span className="badge badge-danger">High Risk</span>;
      case 'moderate': return <span className="badge badge-warning">Moderate</span>;
      case 'requires_review': return <span className="badge badge-warning">Review Flag</span>;
      case 'low': return <span className="badge badge-success">Low Risk</span>;
      default: return <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>N/A</span>;
    }
  };

  return (
    <div className="container" style={{ padding: '36px 16px' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', marginBottom: '4px', color: 'var(--text-main)' }}>
            {user?.role === 'clinician' ? 'Clinical Triage Dashboard' : (user?.role === 'admin' ? 'Administrative Operations Dashboard' : 'Dermatology Analysis Dashboard')}
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Welcome back, <b>{user?.full_name}</b> ({user?.role.toUpperCase()})
          </p>
        </div>
        <Link to="/new-case" className="btn btn-primary">
          <PlusCircle size={18} /> New Skin Analysis
        </Link>
      </div>

      {/* Metrics Stat Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '16px',
        marginBottom: '28px'
      }}>
        {/* Total Cases */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase' }}>Total Cases</span>
            <Activity size={18} color="var(--primary)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)' }}>
            {summary ? summary.total_cases : '...'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Analyzed or queued records</span>
        </div>

        {/* Requires Review */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase' }}>High Risk / Review</span>
            <AlertTriangle size={18} color="var(--warning)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--warning)' }}>
            {summary ? summary.cases_requiring_review : '...'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>High-risk or low-confidence triage</span>
        </div>

        {/* Completed */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase' }}>Completed</span>
            <CheckCircle size={18} color="var(--success)" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--success)' }}>
            {summary ? summary.completed_cases : '...'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Successful AI inferences</span>
        </div>

        {/* Active Model */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase' }}>Model Metrics</span>
            <Shield size={18} color="var(--primary)" />
          </div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary-dark)', marginTop: '4px', marginBottom: '6px' }}>
            {summary?.active_model?.name || 'EfficientNet-B0'}
          </div>
          {(() => {
            const tm = evalSummary?.test_metrics;
            const acc = tm?.overall_accuracy ?? summary?.active_model?.test_accuracy;
            const bal = tm?.balanced_accuracy ?? summary?.active_model?.test_balanced_acc;
            const f1 = tm?.macro_f1 ?? summary?.active_model?.test_macro_f1;
            return (
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                <div>Overall Acc: <b>{acc !== undefined && acc !== null ? `${(acc * 100).toFixed(2)}%` : '—'}</b></div>
                <div>Balanced Acc: <b>{bal !== undefined && bal !== null ? `${(bal * 100).toFixed(2)}%` : '—'}</b> | Macro F1: <b>{f1 !== undefined && f1 !== null ? `${(f1 * 100).toFixed(2)}%` : '—'}</b></div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>PAD-UFES-20 Test Split (N={tm?.sample_count ?? 329})</div>
              </div>
            );
          })()}
        </div>
      </div>

      {/* Disease Distribution Section (if cases exist) */}
      {summary && Object.keys(summary.disease_distribution).length > 0 && (
        <div className="glass-panel" style={{ padding: '20px', borderRadius: 'var(--radius-lg)', marginBottom: '28px' }}>
          <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)', marginBottom: '14px', fontWeight: 700 }}>
            Class Prediction Distribution in Platform
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
            {Object.entries(summary.disease_distribution).map(([cls, count]) => (
              <div key={cls} style={{
                background: '#f8fafc',
                border: '1px solid var(--border-subtle)',
                padding: '8px 16px',
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                borderLeft: '3px solid var(--primary)'
              }}>
                <b style={{ color: 'var(--text-main)' }}>{cls}:</b>
                <span style={{ color: 'var(--primary-dark)', fontWeight: 700 }}>{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Cases Table */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', margin: 0, fontWeight: 700 }}>
            Recent Consultation & Analysis Records
          </h3>
          <Link to="/history" style={{ fontSize: '0.85rem', color: 'var(--primary)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
            View All ({cases.length}) <ChevronRight size={14} />
          </Link>
        </div>

        {cases.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Activity size={36} style={{ margin: '0 auto 12px auto', display: 'block', color: 'var(--text-dim)' }} />
            <p style={{ marginBottom: '16px', fontWeight: 600 }}>
              {user?.role === 'patient' && 'No personal skin analyses recorded yet. Upload a lesion photo to start.'}
              {user?.role === 'clinician' && 'No patient consultation records found in clinical queue.'}
              {user?.role === 'researcher' && 'No research analysis records available in registry.'}
              {user?.role === 'admin' && 'No analysis cases recorded in platform database.'}
            </p>
            <Link to="/new-case" className="btn btn-primary btn-sm">
              <PlusCircle size={14} /> Start New Analysis
            </Link>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left', background: '#f8fafc' }}>
                  <th style={{ padding: '10px 14px' }}>Patient / Case Ref</th>
                  <th style={{ padding: '10px 14px' }}>Age</th>
                  <th style={{ padding: '10px 14px' }}>Site</th>
                  <th style={{ padding: '10px 14px' }}>Status</th>
                  <th style={{ padding: '10px 14px' }}>Predicted Condition</th>
                  <th style={{ padding: '10px 14px' }}>Risk Priority</th>
                  <th style={{ padding: '10px 14px' }}>Date (IST)</th>
                  <th style={{ padding: '10px 14px', textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {cases.slice(0, 8).map((c) => {
                  const pred = c.images[0]?.prediction;
                  return (
                    <tr key={c.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: 'var(--text-main)' }}>
                        {c.patient_ref || c.id.slice(0, 8)}
                      </td>
                      <td style={{ padding: '12px 14px', color: 'var(--text-muted)' }}>
                        {c.age !== null && c.age !== undefined ? `${c.age} yrs` : 'Not provided'}
                      </td>
                      <td style={{ padding: '12px 14px', color: 'var(--text-muted)' }}>
                        {c.anatomical_site || 'Unspecified'}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        {getStatusBadge(c.status)}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        {pred ? (
                          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>
                            {pred.predicted_class} <span style={{ color: 'var(--text-dim)', fontSize: '0.78rem' }}>({pred.confidence_percentage_str})</span>
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-dim)' }}>Pending analysis</span>
                        )}
                      </td>
                      <td style={{ padding: '12px 14px' }}>
                        {pred ? getRiskBadge(pred.risk_tier) : '—'}
                      </td>
                      <td style={{ padding: '12px 14px', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                        {formatDateTimeIST(pred?.created_at || c.created_at)}
                      </td>
                      <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                        <Link to={`/cases/${c.id}`} className="btn btn-secondary btn-sm" style={{ padding: '4px 10px' }}>
                          View Details
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
};
