import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Activity, Shield, Eye, BarChart3, Camera, 
  ArrowRight, CheckCircle2, AlertTriangle, Cpu, Layers, Sparkles 
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* Hero Section */}
      <section style={{
        padding: '70px 0 50px 0',
        textAlign: 'center',
        position: 'relative'
      }}>
        <div className="container" style={{ maxWidth: '960px' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: 'var(--primary-subtle)',
            border: '1px solid rgba(13, 148, 136, 0.3)',
            borderRadius: '9999px',
            padding: '6px 16px',
            fontSize: '0.82rem',
            color: 'var(--primary-dark)',
            fontWeight: 700,
            marginBottom: '20px'
          }}>
            <Cpu size={16} /> Clinical AI Decision-Support Platform
          </div>

          <h1 style={{
            fontSize: '2.8rem',
            lineHeight: 1.2,
            fontWeight: 800,
            marginBottom: '18px',
            color: 'var(--text-main)',
            letterSpacing: '-0.03em'
          }}>
            Skin Lesion Analysis & Screening from Mobile Images Using <span style={{ color: 'var(--primary)' }}>Deep Learning</span>
          </h1>

          <p style={{
            fontSize: '1.1rem',
            color: 'var(--text-muted)',
            lineHeight: 1.6,
            maxWidth: '760px',
            margin: '0 auto 28px auto'
          }}>
            DermAI is an artificial intelligence screening assistant designed to analyze dermatological lesions captured via mobile smartphone photography. Featuring automated image quality assessment, standardized preprocessing, EfficientNet-B0 transfer learning, post-hoc confidence calibration, and Grad-CAM visual interpretability.
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '14px', flexWrap: 'wrap', marginBottom: '32px' }}>
            <Link to="/new-case" className="btn btn-primary btn-lg">
              Start Analysis <ArrowRight size={18} />
            </Link>
            <Link to="/evaluation" className="btn btn-secondary btn-lg">
              <BarChart3 size={18} /> View Model Metrics
            </Link>
          </div>

        </div>
      </section>

      {/* 4-Step End-to-End Workflow Section */}
      <section style={{ padding: '40px 0 50px 0', background: '#ffffff', borderTop: '1px solid var(--border-subtle)', borderBottom: '1px solid var(--border-subtle)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '36px' }}>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '8px', color: 'var(--text-main)' }}>Standardized Analysis Workflow</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>From mobile image acquisition to verified clinical report generation</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
            <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--primary-subtle)', color: 'var(--primary-dark)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px auto', fontWeight: 800, fontSize: '1.1rem' }}>
                1
              </div>
              <h3 style={{ fontSize: '1rem', marginBottom: '6px', color: 'var(--text-main)' }}>Upload Lesion Image</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                Upload JPEG, PNG, or WebP smartphone images, or capture live using camera integration.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--info-bg)', color: 'var(--info)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px auto', fontWeight: 800, fontSize: '1.1rem' }}>
                2
              </div>
              <h3 style={{ fontSize: '1rem', marginBottom: '6px', color: 'var(--text-main)' }}>Quality Assessment</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                Automated Laplacian blur scoring, mean luminance, contrast analysis, and artifact detection.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--primary-subtle)', color: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px auto', fontWeight: 800, fontSize: '1.1rem' }}>
                3
              </div>
              <h3 style={{ fontSize: '1rem', marginBottom: '6px', color: 'var(--text-main)' }}>AI Model Inference</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                EfficientNet-B0 inference with post-hoc temperature scaling calibration and Grad-CAM saliency heatmaps.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--success-bg)', color: 'var(--success)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px auto', fontWeight: 800, fontSize: '1.1rem' }}>
                4
              </div>
              <h3 style={{ fontSize: '1rem', marginBottom: '6px', color: 'var(--text-main)' }}>Review & Report</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                Review un-numbered differential predictions, compare historical progression, and download clinical PDF/CSV/JSON reports.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 6 Diagnostic Classes Pathology Showcase */}
      <section style={{ padding: '50px 0' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '8px', color: 'var(--text-main)' }}>Target Diagnostic Pathology</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Trained on the patient-stratified PAD-UFES-20 dataset across six validated dermatological conditions
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
            {[
              { code: 'BCC', name: 'Basal Cell Carcinoma', type: 'Malignant (Non-Melanoma)', color: '#b91c1c' },
              { code: 'ACK', name: 'Actinic Keratosis', type: 'Pre-Malignant', color: '#b45309' },
              { code: 'NEV', name: 'Melanocytic Nevus', type: 'Benign Mole', color: '#15803d' },
              { code: 'SEK', name: 'Seborrheic Keratosis', type: 'Benign Growth', color: '#15803d' },
              { code: 'SCC', name: 'Squamous Cell Carcinoma', type: 'Malignant (Non-Melanoma)', color: '#b91c1c' },
              { code: 'MEL', name: 'Melanoma', type: 'Malignant Skin Cancer', color: '#991b1b' },
            ].map(cls => (
              <div key={cls.code} className="glass-panel" style={{ padding: '16px', textAlign: 'center', borderTop: `3px solid ${cls.color}` }}>
                <span style={{ fontSize: '1.4rem', fontWeight: 800, color: cls.color, display: 'block', marginBottom: '4px' }}>
                  {cls.code}
                </span>
                <b style={{ fontSize: '0.88rem', display: 'block', color: 'var(--text-main)', marginBottom: '4px' }}>
                  {cls.name}
                </b>
                <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                  {cls.type}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Core Technical Highlights */}
      <section style={{ padding: '50px 0', background: '#ffffff', borderTop: '1px solid var(--border-subtle)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '36px' }}>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '8px', color: 'var(--text-main)' }}>Platform Architecture & Scientific Validation</h2>
            <p style={{ color: 'var(--text-muted)' }}>Engineered for clinical decision support, explainability, and reproducible research</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
            <div className="glass-panel" style={{ padding: '24px' }}>
              <Cpu size={26} color="var(--primary)" style={{ marginBottom: '12px' }} />
              <h3 style={{ fontSize: '1.1rem', marginBottom: '6px', color: 'var(--text-main)' }}>EfficientNet-B0 Architecture</h3>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Trained via compound scaling transfer learning on standardized 224×224 tensors. Evaluated strictly on held-out patient-stratified test splits.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '24px' }}>
              <Sparkles size={26} color="var(--accent)" style={{ marginBottom: '12px' }} />
              <h3 style={{ fontSize: '1.1rem', marginBottom: '6px', color: 'var(--text-main)' }}>Post-Hoc Confidence Calibration</h3>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Temperature scaling (T = 1.060) fitted exclusively on validation logits aligns model scores with empirical accuracy, mitigating overconfidence.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '24px' }}>
              <Eye size={26} color="var(--primary)" style={{ marginBottom: '12px' }} />
              <h3 style={{ fontSize: '1.1rem', marginBottom: '6px', color: 'var(--text-main)' }}>Grad-CAM Interpretability</h3>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Gradient-weighted class activation mapping extracts spatial saliency maps from the final convolutional layer, highlighting regions driving predictions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ padding: '24px 0', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-subtle)' }}>
        <div className="container">
          <p>© 2026 DermAI Clinical Research Platform. Decision support system only — not for standalone diagnosis.</p>
        </div>
      </footer>
    </div>
  );
};
