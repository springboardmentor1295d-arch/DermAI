import React, { useState } from 'react';
import { Eye, Layers, Grid, Sparkles, Crosshair, Flame } from 'lucide-react';

interface ImageComparisonViewProps {
  originalUrl: string;
  preprocessedUrl?: string;
  gradcamUrl?: string;
  gradcamHeatmapUrl?: string;
  roiMaskUrl?: string;
  predictedClass: string;
}

export const ImageComparisonView: React.FC<ImageComparisonViewProps> = ({
  originalUrl,
  preprocessedUrl,
  gradcamUrl,
  gradcamHeatmapUrl,
  roiMaskUrl,
  predictedClass,
}) => {
  const [activeTab, setActiveTab] = useState<'sideBySide' | 'original' | 'heatmap' | 'gradcam' | 'roi'>('sideBySide');

  return (
    <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-main)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} color="var(--primary)" /> Multi-Modal Visual Analysis & Explainability (EfficientNet-B0)
          </h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Comparing Original Input, Classical Lesion ROI Mask, Grad-CAM Heatmap, and Decision Overlay
          </span>
        </div>

        {/* View Switcher Tabs */}
        <div style={{ display: 'flex', gap: '6px', background: 'var(--bg-tertiary)', padding: '4px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <button
            type="button"
            onClick={() => setActiveTab('sideBySide')}
            className={`btn btn-sm ${activeTab === 'sideBySide' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '5px 12px' }}
          >
            <Grid size={14} /> Multi-Panel Grid
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('original')}
            className={`btn btn-sm ${activeTab === 'original' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '5px 12px' }}
          >
            <Eye size={14} /> Original
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('heatmap')}
            className={`btn btn-sm ${activeTab === 'heatmap' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '5px 12px' }}
          >
            <Flame size={14} /> Heatmap
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('gradcam')}
            className={`btn btn-sm ${activeTab === 'gradcam' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '5px 12px' }}
          >
            <Sparkles size={14} /> Overlay
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('roi')}
            className={`btn btn-sm ${activeTab === 'roi' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '5px 12px' }}
          >
            <Crosshair size={14} /> Lesion ROI
          </button>
        </div>
      </div>

      {/* Side-by-Side Multi-Panel Grid View */}
      {activeTab === 'sideBySide' && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
        }}>
          {/* Panel 1: Original */}
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
              Raw Mobile Image
            </div>
            <img
              src={originalUrl}
              alt="Raw Mobile Acquisition"
              style={{ width: '100%', height: '220px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }}
            />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'block', marginTop: '6px' }}>
              Original RGB acquisition
            </span>
          </div>

          {/* Panel 2: Preprocessed (Standardized 224x224) */}
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
              Preprocessed (224×224)
            </div>
            {preprocessedUrl ? (
              <img
                src={preprocessedUrl}
                alt="Preprocessed Tensor"
                style={{ width: '100%', height: '220px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }}
              />
            ) : (
              <div style={{ height: '220px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.8rem', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <span>Preprocessed 224×224</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>Standardized RGB tensor</span>
              </div>
            )}
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'block', marginTop: '6px' }}>
              Standardized 224×224 RGB image
            </span>
          </div>

          {/* Panel 3: Lesion ROI Mask */}
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
              Lesion Localization (ROI)
            </div>
            {roiMaskUrl ? (
              <img
                src={roiMaskUrl}
                alt="Lesion Segmentation ROI"
                style={{ width: '100%', height: '220px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }}
              />
            ) : (
              <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.8rem', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                ROI mask unavailable
              </div>
            )}
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'block', marginTop: '6px' }}>
              Adaptive Otsu & CIE-Lab saliency
            </span>
          </div>

          {/* Panel 4: Grad-CAM Standalone Heatmap */}
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
              Grad-CAM Heatmap
            </div>
            {gradcamHeatmapUrl || gradcamUrl ? (
              <img
                src={gradcamHeatmapUrl || gradcamUrl}
                alt="Grad-CAM Raw Heatmap"
                style={{ width: '100%', height: '220px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }}
              />
            ) : (
              <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.8rem', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                Heatmap unavailable
              </div>
            )}
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'block', marginTop: '6px' }}>
              Normalized feature activations [0, 1]
            </span>
          </div>

          {/* Panel 5: Grad-CAM Overlay */}
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
              Decision Overlay ({predictedClass})
            </div>
            {gradcamUrl ? (
              <img
                src={gradcamUrl}
                alt="Grad-CAM Overlay"
                style={{ width: '100%', height: '220px', objectFit: 'contain', borderRadius: 'var(--radius-sm)', background: '#ffffff', border: '1px solid var(--border-subtle)' }}
              />
            ) : (
              <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.8rem', background: '#ffffff', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                Overlay unavailable
              </div>
            )}
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', display: 'block', marginTop: '6px' }}>
              Warm regions drove class decision
            </span>
          </div>
        </div>
      )}

      {/* Single View Modes */}
      {activeTab === 'original' && (
        <div style={{ textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <img src={originalUrl} alt="Original" style={{ maxHeight: '420px', maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
        </div>
      )}

      {activeTab === 'heatmap' && (
        <div style={{ textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          {gradcamHeatmapUrl || gradcamUrl ? (
            <img src={gradcamHeatmapUrl || gradcamUrl} alt="Grad-CAM Heatmap" style={{ maxHeight: '420px', maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
          ) : (
            <div style={{ padding: '40px', color: 'var(--text-muted)' }}>Grad-CAM heatmap not available for this analysis.</div>
          )}
        </div>
      )}

      {activeTab === 'gradcam' && (
        <div style={{ textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          {gradcamUrl ? (
            <img src={gradcamUrl} alt="Grad-CAM Overlay" style={{ maxHeight: '420px', maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
          ) : (
            <div style={{ padding: '40px', color: 'var(--text-muted)' }}>Grad-CAM overlay not available for this analysis.</div>
          )}
        </div>
      )}

      {activeTab === 'roi' && (
        <div style={{ textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          {roiMaskUrl ? (
            <img src={roiMaskUrl} alt="ROI Mask" style={{ maxHeight: '420px', maxWidth: '100%', borderRadius: 'var(--radius-md)', background: '#ffffff', border: '1px solid var(--border-subtle)' }} />
          ) : (
            <div style={{ padding: '40px', color: 'var(--text-muted)' }}>ROI mask not available for this analysis.</div>
          )}
        </div>
      )}
    </div>
  );
};
