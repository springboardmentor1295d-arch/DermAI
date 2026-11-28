import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { 
  BarChart2, CheckCircle2, AlertTriangle, ShieldCheck, 
  HelpCircle, Cpu, Table, FileText, Info 
} from 'lucide-react';

export const ModelEvaluationPage: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = () => {
    setLoading(true);
    setError(null);
    api.getModelEvaluationSummary()
      .then((res) => {
        if (!res || !res.test_metrics || !res.test_metrics.per_class) {
          setError(res?.message || 'Held-out test evaluation metrics are currently unavailable.');
          setData(null);
        } else {
          setData(res);
        }
      })
      .catch((err: any) => {
        setError(err?.message || 'Unable to load evaluation metrics from the backend API.');
        setData(null);
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const metrics = data?.test_metrics;
  const melMetrics = metrics?.per_class?.MEL;

  return (
    <div className="container" style={{ maxWidth: '1080px', padding: '36px 16px' }}>
      
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '2.2rem', marginBottom: '6px', color: 'var(--text-main)', fontWeight: 800 }}>
          Empirical Model Evaluation & Scientific Benchmark
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Active Deployed Model: <b>{data?.display_name || data?.active_model_name || 'EfficientNet-B0'} ({data?.model_version || 'v1.1.0-efficientnet_b0'})</b> • Dataset: <b>PAD-UFES-20 Held-Out Test Split</b> (N={metrics?.sample_count ?? 329} image records across 198 unique patients, zero patient leakage).
        </p>
      </div>

      {error && (
        <div style={{
          background: 'var(--danger-bg)',
          border: '1px solid var(--danger-border)',
          borderRadius: 'var(--radius-md)',
          padding: '14px 18px',
          marginBottom: '20px',
          color: 'var(--danger)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '12px',
          flexWrap: 'wrap'
        }}>
          <span><b>Metrics Load Error:</b> {error}</span>
          <button type="button" onClick={fetchMetrics} className="btn btn-secondary btn-sm">
            Retry
          </button>
        </div>
      )}

      {/* Top Benchmark KPI Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px',
        marginBottom: '28px'
      }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Overall Accuracy
          </span>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--accent)', marginTop: '4px' }}>
            {loading ? '...' : (metrics ? `${(metrics.overall_accuracy * 100).toFixed(2)}%` : '—')}
          </div>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Top-1 exact diagnosis match</span>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Balanced Accuracy
          </span>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--primary)', marginTop: '4px' }}>
            {loading ? '...' : (metrics ? `${(metrics.balanced_accuracy * 100).toFixed(2)}%` : '—')}
          </div>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Accounting for class imbalance</span>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Macro F1-Score
          </span>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--accent-purple)', marginTop: '4px' }}>
            {loading ? '...' : (metrics ? `${(metrics.macro_f1 * 100).toFixed(2)}%` : '—')}
          </div>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Unweighted mean across 6 classes</span>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase' }}>
            Held-Out Test Set
          </span>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)', marginTop: '4px' }}>
            {metrics?.sample_count ?? 329}
          </div>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>198 unique patients, 0 overlap</span>
        </div>
      </div>

      {/* Model Selection & Baseline Comparison */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '28px' }}>
        <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700 }}>
          <Cpu size={18} color="var(--primary)" /> Architecture Comparison (Held-Out Test Set)
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px', lineHeight: 1.5 }}>
          All candidate architectures were evaluated identically on the held-out PAD-UFES-20 test split (N=329) without patient leakage. The currently active administrator-selected classifier is highlighted below.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '16px', borderRadius: 'var(--radius-md)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <b style={{ color: 'var(--text-main)' }}>Baseline 4-Layer CNN</b>
              <span className="badge badge-info">Reference Baseline</span>
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>Validation Balanced Acc: <b>30.6%</b></div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '2px' }}>Trained from scratch without ImageNet transfer.</div>
          </div>

          {(['mobilenetv3_large', 'efficientnet_b0'] as const).map((archKey) => {
            const cmp = data?.comparison_models?.[archKey];
            const isActive = cmp?.is_active ?? (archKey === (data?.active_architecture || 'efficientnet_b0'));
            const title = cmp?.display_name || (archKey === 'mobilenetv3_large' ? 'MobileNetV3-Large' : 'EfficientNet-B0');
            return (
              <div
                key={archKey}
                style={{
                  background: isActive ? 'var(--primary-subtle)' : '#f8fafc',
                  border: isActive ? '1.5px solid var(--primary)' : '1px solid var(--border-subtle)',
                  padding: '16px',
                  borderRadius: 'var(--radius-md)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <b style={{ color: isActive ? 'var(--primary-dark)' : 'var(--text-main)' }}>{title}</b>
                  <span className={`badge ${isActive ? 'badge-success' : 'badge-info'}`}>
                    {isActive ? 'Active Production' : 'Standby'}
                  </span>
                </div>
                <div style={{ fontSize: '0.84rem', color: 'var(--text-main)' }}>
                  Test Accuracy: <b>{cmp?.overall_accuracy != null ? `${(cmp.overall_accuracy * 100).toFixed(2)}%` : '—'}</b> | Balanced Acc: <b>{cmp?.balanced_accuracy != null ? `${(cmp.balanced_accuracy * 100).toFixed(2)}%` : '—'}</b>
                </div>
                <div style={{ fontSize: '0.82rem', color: isActive ? 'var(--primary-dark)' : 'var(--text-muted)', fontWeight: 600, marginTop: '2px' }}>
                  Macro F1: <b>{cmp?.macro_f1 != null ? `${(cmp.macro_f1 * 100).toFixed(2)}%` : '—'}</b> | Cal. ECE: <b>{cmp?.calibrated_ece != null ? `${(cmp.calibrated_ece * 100).toFixed(2)}%` : '—'}</b>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Per-Class Metrics Table */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '28px' }}>
        <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '8px', fontWeight: 700 }}>
          Class-Wise Breakdown (Held-Out Test Set)
        </h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
          Per-class precision, recall (sensitivity), specificity, and F1-score across all six diagnostic classes.
        </p>

        {loading ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading evaluation metrics...
          </div>
        ) : error ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--danger)' }}>
            Unable to display class-wise metrics: {error}
          </div>
        ) : metrics?.per_class ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left', background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px' }}>Code</th>
                  <th style={{ padding: '10px 12px' }}>Diagnostic Condition</th>
                  <th style={{ padding: '10px 12px' }}>Precision</th>
                  <th style={{ padding: '10px 12px' }}>Recall (Sens.)</th>
                  <th style={{ padding: '10px 12px' }}>Specificity</th>
                  <th style={{ padding: '10px 12px' }}>F1-Score</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right' }}>Test Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_class).map(([code, m]: [string, any]) => (
                  <tr key={code} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '12px', fontWeight: 700, color: 'var(--primary-dark)' }}>{code}</td>
                    <td style={{ padding: '12px', color: 'var(--text-main)', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)' }}>{(m.precision * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px', color: 'var(--primary)', fontWeight: 600 }}>{(m.recall * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)' }}>{(m.specificity * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px', color: 'var(--accent)', fontWeight: 700 }}>{(m.f1_score * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: 'var(--text-main)', fontWeight: 600 }}>{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No per-class evaluation metrics found.
          </div>
        )}
      </div>

      {/* 6x6 Confusion Matrix Display */}
      {metrics?.confusion_matrix && (
        <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '28px' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '8px', fontWeight: 700 }}>
            Empirical 6×6 Confusion Matrix (Test Split)
          </h3>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
            Row: Ground Truth Label | Column: Model Prediction. Diagonal cells represent correct classifications.
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', maxWidth: '640px', margin: '0 auto', borderCollapse: 'collapse', textAlign: 'center', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px', color: 'var(--text-muted)', textAlign: 'left' }}>True \ Pred</th>
                  {metrics.classes.map((c: string) => (
                    <th key={c} style={{ padding: '8px', color: 'var(--primary-dark)', fontWeight: 700 }}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.confusion_matrix.map((row: number[], i: number) => {
                  const trueClass = metrics.classes[i];
                  return (
                    <tr key={trueClass}>
                      <td style={{ padding: '8px', fontWeight: 700, color: 'var(--text-main)', textAlign: 'left', background: '#f8fafc' }}>{trueClass}</td>
                      {row.map((val: number, j: number) => {
                        const isDiag = i === j;
                        return (
                          <td key={j} style={{
                            padding: '10px',
                            background: isDiag ? 'rgba(13, 148, 136, 0.12)' : (val > 0 ? 'rgba(239, 68, 68, 0.06)' : '#ffffff'),
                            border: '1px solid var(--border-subtle)',
                            fontWeight: isDiag ? 700 : 400,
                            color: isDiag ? 'var(--primary-dark)' : (val > 0 ? 'var(--danger)' : 'var(--text-dim)')
                          }}>
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Scientific Note on Melanoma Sample Size */}
      <div className="glass-panel" style={{
        padding: '22px',
        borderRadius: 'var(--radius-lg)',
        borderLeft: '4px solid var(--warning)',
        background: '#f8fafc'
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
          <Info size={22} color="var(--warning)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <h4 style={{ color: 'var(--text-main)', fontSize: '1.02rem', marginBottom: '6px', fontWeight: 700 }}>
              Research Integrity Note: Melanoma Sample Size & Evaluation Uncertainty
            </h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: '8px' }}>
              The complete PAD-UFES-20 dataset contains only <b>52 Melanoma (MEL)</b> images out of 2,298 records (2.26%), of which exactly <b>{melMetrics?.support ?? 7} images</b> are in the held-out test split.
              {melMetrics ? ` On the latest evaluation (${data?.display_name || 'Active Model'}), the model identified ${Math.round(melMetrics.recall * melMetrics.support)} of the ${melMetrics.support} test cases with ${(melMetrics.precision * 100).toFixed(1)}% precision, which still entails substantial statistical uncertainty due to small sample size.` : ' A small test sample size entails substantial statistical uncertainty.'}
            </p>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>
              In accordance with research ethics standards, the system triggers a mandatory <b>Clinical Review</b> flag whenever melanoma features or elevated uncertainty are detected.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
};
