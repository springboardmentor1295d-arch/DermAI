import React, { useRef, useState, useEffect } from 'react';
import { Upload, Image as ImageIcon, X, RefreshCw, Camera, AlertTriangle, CheckCircle2 } from 'lucide-react';

export interface ImageUploadDropzoneProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  onCameraClick?: () => void;
  disabled?: boolean;
  error?: string | null;
  onErrorClear?: () => void;
  onValidationError?: (errorMessage: string) => void;
}

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB

export const ImageUploadDropzone: React.FC<ImageUploadDropzoneProps> = ({
  selectedFile,
  previewUrl,
  onFileSelect,
  onFileRemove,
  onCameraClick,
  disabled = false,
  error,
  onErrorClear,
  onValidationError,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset dimensions when file changes
  useEffect(() => {
    if (!selectedFile) {
      setImageDimensions(null);
    }
  }, [selectedFile]);

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    // 1. Check extension
    const extension = `.${file.name.split('.').pop()?.toLowerCase()}`;
    const hasValidExtension = ALLOWED_EXTENSIONS.includes(extension);

    // 2. Check MIME type
    const hasValidMime = !file.type || ALLOWED_MIME_TYPES.includes(file.type.toLowerCase()) || file.type === 'image/jpg';

    if (!hasValidExtension || !hasValidMime) {
      return {
        valid: false,
        error: `Unsupported image format (${extension || 'unknown'}). Supported formats: ${ALLOWED_EXTENSIONS.join(', ')}`,
      };
    }

    // 3. Check 15MB limit
    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      return {
        valid: false,
        error: `File size (${sizeMb} MB) exceeds the 15 MB limit. Please choose a smaller image.`,
      };
    }

    // 4. Check for empty file
    if (file.size === 0) {
      return {
        valid: false,
        error: 'Uploaded file is empty. Please select a valid skin lesion image.',
      };
    }

    return { valid: true };
  };

  const processFile = (file: File) => {
    if (disabled) return;
    if (onErrorClear) onErrorClear();

    const validation = validateFile(file);
    if (!validation.valid) {
      if (onValidationError) {
        onValidationError(validation.error!);
      }
      return;
    }

    onFileSelect(file);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
    return `${(bytes / 1024).toFixed(1)} KB`;
  };

  return (
    <div style={{ marginBottom: '28px' }}>
      {/* Hidden native file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ALLOWED_EXTENSIONS.join(',')}
        onChange={handleInputChange}
        style={{ display: 'none' }}
        disabled={disabled}
      />

      <label className="form-label" style={{ fontSize: '1rem', color: 'var(--text-main)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>Lesion Photograph</span>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 400 }}>(JPEG, PNG, WebP up to 15MB)</span>
      </label>

      {/* Validation Error Banner */}
      {error && (
        <div
          id="dropzone-validation-error"
          style={{
            background: 'var(--danger-bg)',
            border: '1px solid var(--danger-border)',
            borderRadius: 'var(--radius-md)',
            padding: '12px 16px',
            color: 'var(--danger)',
            fontSize: '0.85rem',
            marginBottom: '14px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <AlertTriangle size={18} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}

      {selectedFile && previewUrl ? (
        /* Image Preview State */
        <div
          id="image-preview-container"
          style={{
            background: '#ffffff',
            border: '1px solid var(--border-medium)',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div
            style={{
              position: 'relative',
              background: '#f8fafc',
              padding: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '260px',
            }}
          >
            <img
              id="image-preview-img"
              src={previewUrl}
              alt="Selected lesion preview"
              onLoad={(e) => {
                setImageDimensions({
                  width: e.currentTarget.naturalWidth,
                  height: e.currentTarget.naturalHeight,
                });
              }}
              style={{
                maxWidth: '100%',
                maxHeight: '360px',
                objectFit: 'contain',
                borderRadius: 'var(--radius-md)',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                background: '#ffffff',
              }}
            />

            {/* Quick remove button */}
            <button
              type="button"
              id="remove-image-overlay-btn"
              onClick={onFileRemove}
              disabled={disabled}
              title="Remove this image"
              style={{
                position: 'absolute',
                top: '14px',
                right: '14px',
                background: 'var(--danger)',
                border: 'none',
                color: '#fff',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: disabled ? 'not-allowed' : 'pointer',
                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
              }}
            >
              <X size={16} />
            </button>
          </div>

          {/* Metadata bar & action controls */}
          <div
            style={{
              padding: '14px 20px',
              borderTop: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '12px',
              background: '#ffffff',
            }}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)' }}>
                  {selectedFile.name}
                </span>
                <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>
                  <CheckCircle2 size={12} /> Valid Format
                </span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', gap: '14px' }}>
                <span>Size: <b>{formatFileSize(selectedFile.size)}</b></span>
                <span>Type: <b>{selectedFile.type || 'image'}</b></span>
                {imageDimensions && (
                  <span>Resolution: <b>{imageDimensions.width} × {imageDimensions.height} px</b></span>
                )}
              </div>
            </div>

            {/* Replace & Remove Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <button
                type="button"
                id="replace-image-btn"
                className="btn btn-secondary btn-sm"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <RefreshCw size={14} /> Replace
              </button>
              <button
                type="button"
                id="remove-image-btn"
                className="btn btn-danger btn-sm"
                onClick={onFileRemove}
                disabled={disabled}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <X size={14} /> Remove
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* Empty / Dropzone State */
        <div
          id="image-dropzone"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          style={{
            border: isDragging
              ? '2px dashed var(--primary)'
              : '2px dashed var(--border-medium)',
            borderRadius: 'var(--radius-lg)',
            padding: '42px 24px',
            textAlign: 'center',
            background: isDragging ? 'var(--primary-subtle)' : '#ffffff',
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            boxShadow: isDragging ? '0 0 12px rgba(13, 148, 136, 0.2)' : 'var(--shadow-sm)',
          }}
        >
          <div
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '50%',
              background: 'var(--primary-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 14px auto',
              color: 'var(--primary)',
            }}
          >
            <Upload size={26} />
          </div>

          <h3 style={{ color: 'var(--text-main)', fontSize: '1.05rem', marginBottom: '6px', fontWeight: 700 }}>
            Drag & drop lesion image here, or click to browse
          </h3>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', maxWidth: '460px', margin: '0 auto 12px auto' }}>
            Supports PNG, JPEG, and WebP (up to 15MB). The image will be validated, securely stored, and prepared for quality check & ML inference.
          </p>
          <p style={{ fontSize: '0.74rem', color: 'var(--text-dim)', maxWidth: '520px', margin: '0 auto 18px auto' }}>
            Tips: Keep the camera steady • Use natural or sufficient lighting • Keep the affected area in focus • Avoid shadows and flash glare • Capture the lesion clearly.
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <button
              type="button"
              id="choose-file-btn"
              className="btn btn-secondary btn-sm"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
              disabled={disabled}
            >
              <ImageIcon size={15} /> Choose File
            </button>
            {onCameraClick && (
              <button
                type="button"
                id="use-camera-btn"
                className="btn btn-primary btn-sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onCameraClick();
                }}
                disabled={disabled}
              >
                <Camera size={15} /> Capture via Camera
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
