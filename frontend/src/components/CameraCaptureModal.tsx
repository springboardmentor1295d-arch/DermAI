import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, RefreshCw, X, Check, AlertTriangle, ShieldAlert, VideoOff, Lock } from 'lucide-react';

export type CameraState =
  | 'CLOSED'
  | 'IDLE_WAITING_FOR_TRIGGER'
  | 'REQUESTING_PERMISSION'
  | 'PERMISSION_GRANTED'
  | 'PREVIEW_ACTIVE'
  | 'IMAGE_CAPTURED'
  | 'PERMISSION_DENIED'
  | 'NO_DEVICE_FOUND'
  | 'DEVICE_BUSY'
  | 'BROWSER_UNSUPPORTED'
  | 'INSECURE_CONTEXT';

interface CameraCaptureModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

export const CameraCaptureModal: React.FC<CameraCaptureModalProps> = ({
  isOpen,
  onClose,
  onCapture,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>('IDLE_WAITING_FOR_TRIGGER');
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Clean up all media stream tracks safely
  const stopAllTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore error on stop
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Request camera access strictly following mentor's specification:
  // navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false })
  const requestCameraAccess = useCallback(async (preferredFacing: 'user' | 'environment' = 'environment') => {
    // 1. Check secure context (HTTPS or localhost)
    if (typeof window !== 'undefined' && !window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      setCameraState('INSECURE_CONTEXT');
      setErrorMessage(
        'Camera access requires a secure HTTPS connection or localhost. Please access DermAI via https:// or localhost.'
      );
      return;
    }

    // 2. Check browser API support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraState('BROWSER_UNSUPPORTED');
      setErrorMessage(
        'Your web browser does not support the MediaDevices camera API. Please use a modern browser such as Chrome, Edge, Safari, or Firefox.'
      );
      return;
    }

    stopAllTracks();
    setCameraState('REQUESTING_PERMISSION');
    setErrorMessage(null);

    let mediaStream: MediaStream | null = null;

    try {
      // Direct user gesture camera request matching exact requirement:
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: preferredFacing } },
        audio: false
      });
    } catch (primaryErr: any) {
      // Fallback if ideal constraint not satisfied
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
      } catch (finalErr: any) {
        const err = finalErr || primaryErr;
        const errorName = err?.name || '';
        const errorMsgStr = err?.message || '';

        if (errorName === 'NotAllowedError' || errorName === 'PermissionDeniedError' || errorMsgStr.includes('denied')) {
          setCameraState('PERMISSION_DENIED');
          setErrorMessage(
            'Camera permission was denied. Please allow camera permissions in your browser URL bar and click "Open Camera".'
          );
        } else if (errorName === 'NotFoundError' || errorName === 'DevicesNotFoundError') {
          setCameraState('NO_DEVICE_FOUND');
          setErrorMessage(
            'No physical camera device was detected on your device. Please connect a webcam or use standard file upload.'
          );
        } else if (errorName === 'NotReadableError' || errorName === 'TrackStartError') {
          setCameraState('DEVICE_BUSY');
          setErrorMessage(
            'The camera is currently in use by another application. Please close other video tools and click "Retry".'
          );
        } else if (errorName === 'SecurityError') {
          setCameraState('INSECURE_CONTEXT');
          setErrorMessage('Camera access was blocked due to an insecure context. HTTPS or localhost is required.');
        } else {
          setCameraState('PERMISSION_DENIED');
          setErrorMessage(
            `Camera initialization error (${errorName || 'Unknown'}). Please check permissions or upload a file directly.`
          );
        }
        return;
      }
    }

    if (!mediaStream) {
      setCameraState('PERMISSION_DENIED');
      setErrorMessage('Could not establish video stream. Please retry.');
      return;
    }

    streamRef.current = mediaStream;
    setCameraState('PERMISSION_GRANTED');

    if (videoRef.current) {
      videoRef.current.srcObject = mediaStream;
      try {
        await videoRef.current.play();
        setCameraState('PREVIEW_ACTIVE');
      } catch {
        setCameraState('PREVIEW_ACTIVE');
      }
    } else {
      setCameraState('PREVIEW_ACTIVE');
    }
  }, [stopAllTracks]);

  // Modal open / close lifecycle
  useEffect(() => {
    if (isOpen) {
      // Attempt immediate user-context access
      requestCameraAccess(facingMode);
    } else {
      stopAllTracks();
      setCameraState('CLOSED');
      setCapturedBlob(null);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setErrorMessage(null);
    }

    return () => {
      stopAllTracks();
    };
  }, [isOpen, facingMode, requestCameraAccess, stopAllTracks]);

  // Switch facing mode (environment <-> user)
  const handleToggleFacingMode = () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    requestCameraAccess(nextMode);
  };

  // Capture centered square frame matching 1:1 EfficientNet-B0 lesion framing
  const handleTakePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    const side = Math.min(vw, vh);
    const sx = Math.floor((vw - side) / 2);
    const sy = Math.floor((vh - side) / 2);

    const canvas = document.createElement('canvas');
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, sx, sy, side, side, 0, 0, side, side);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          setCapturedBlob(blob);
          const url = URL.createObjectURL(blob);
          setPreviewUrl(url);
          setCameraState('IMAGE_CAPTURED');
          stopAllTracks();
        }
      },
      'image/jpeg',
      0.95
    );
  };

  // Retake photo
  const handleRetake = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setCapturedBlob(null);
    requestCameraAccess(facingMode);
  };

  // Confirm and deliver File object
  const handleConfirm = () => {
    if (capturedBlob) {
      const file = new File([capturedBlob], `camera_lesion_${Date.now()}.jpg`, {
        type: 'image/jpeg',
      });
      onCapture(file);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '560px',
          padding: '24px',
          borderRadius: 'var(--radius-lg)',
          position: 'relative',
          boxShadow: 'var(--shadow-lg)',
          background: '#ffffff',
          border: '1px solid var(--border-medium)',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '16px',
            borderBottom: '1px solid var(--border-subtle)',
            paddingBottom: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Camera size={20} color="var(--primary)" />
            <h3 style={{ margin: 0, fontSize: '1.15rem', color: 'var(--text-main)' }}>
              Lesion Camera Acquisition
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px',
              borderRadius: '4px',
            }}
            aria-label="Close camera"
          >
            <X size={20} />
          </button>
        </div>

        {/* Viewport / Error / Preview Container */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: '360px',
            background: '#f8fafc',
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid var(--border-subtle)',
          }}
        >
          {/* State: REQUESTING_PERMISSION */}
          {cameraState === 'REQUESTING_PERMISSION' && (
            <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
              <RefreshCw
                size={36}
                color="var(--primary)"
                style={{ animation: 'spin 1.5s linear infinite', margin: '0 auto 12px auto' }}
              />
              <h4 style={{ color: 'var(--text-main)', marginBottom: '6px' }}>Requesting Camera Permission...</h4>
              <p style={{ fontSize: '0.85rem', maxWidth: '340px', margin: '0 auto', color: 'var(--text-muted)' }}>
                Please look for your browser’s permission popup and click <b>Allow</b> to enable live video acquisition.
              </p>
            </div>
          )}

          {/* State: IMAGE_CAPTURED (Still Frame Review) */}
          {cameraState === 'IMAGE_CAPTURED' && previewUrl && (
            <img
              src={previewUrl}
              alt="Captured clinical lesion preview"
              style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
            />
          )}

          {/* State: PREVIEW_ACTIVE or PERMISSION_GRANTED */}
          {(cameraState === 'PREVIEW_ACTIVE' || cameraState === 'PERMISSION_GRANTED') && (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />

              {/* Centered Square Lesion Framing Guide */}
              <div
                style={{
                  position: 'absolute',
                  width: '240px',
                  height: '240px',
                  border: '2px dashed rgba(20, 184, 166, 0.9)',
                  borderRadius: '12px',
                  boxShadow: '0 0 0 9999px rgba(15, 23, 42, 0.32)',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'flex-end',
                  justifyContent: 'center',
                  paddingBottom: '8px',
                }}
              >
                <span style={{
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  color: '#ffffff',
                  background: 'rgba(15, 23, 42, 0.65)',
                  padding: '2px 8px',
                  borderRadius: '4px',
                }}>
                  Center Lesion Here
                </span>
              </div>

              {/* Camera Switcher Button */}
              <button
                type="button"
                onClick={handleToggleFacingMode}
                style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '1px solid var(--border-medium)',
                  color: 'var(--text-main)',
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  boxShadow: 'var(--shadow-sm)'
                }}
                title={`Switch camera (current: ${facingMode})`}
              >
                <RefreshCw size={14} />
                <span>{facingMode === 'environment' ? 'Rear (Env)' : 'Front (Selfie)'}</span>
              </button>
            </>
          )}

          {/* States: ERROR / DENIED / INSECURE / BUSY */}
          {(cameraState === 'PERMISSION_DENIED' ||
            cameraState === 'NO_DEVICE_FOUND' ||
            cameraState === 'DEVICE_BUSY' ||
            cameraState === 'BROWSER_UNSUPPORTED' ||
            cameraState === 'INSECURE_CONTEXT' ||
            cameraState === 'IDLE_WAITING_FOR_TRIGGER') && (
            <div style={{ textAlign: 'center', padding: '24px', maxWidth: '420px' }}>
              {cameraState === 'INSECURE_CONTEXT' ? (
                <Lock size={36} color="var(--warning)" style={{ margin: '0 auto 12px auto' }} />
              ) : cameraState === 'NO_DEVICE_FOUND' ? (
                <VideoOff size={36} color="var(--danger)" style={{ margin: '0 auto 12px auto' }} />
              ) : (
                <ShieldAlert size={36} color="var(--primary)" style={{ margin: '0 auto 12px auto' }} />
              )}
              <h4 style={{ color: 'var(--text-main)', marginBottom: '8px', fontSize: '1.05rem' }}>
                {cameraState === 'PERMISSION_DENIED' && 'Camera Access Permission Required'}
                {cameraState === 'NO_DEVICE_FOUND' && 'No Camera Hardware Detected'}
                {cameraState === 'DEVICE_BUSY' && 'Camera Device Busy'}
                {cameraState === 'BROWSER_UNSUPPORTED' && 'Browser Not Supported'}
                {cameraState === 'INSECURE_CONTEXT' && 'Insecure Context (HTTPS Required)'}
                {cameraState === 'IDLE_WAITING_FOR_TRIGGER' && 'Ready to Open Camera'}
              </h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem', marginBottom: '18px', lineHeight: '1.5' }}>
                {errorMessage || 'Click below to grant camera access and start live preview.'}
              </p>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() => requestCameraAccess(facingMode)}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                <Camera size={14} /> Open Camera & Allow Permission
              </button>
            </div>
          )}
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {cameraState === 'IMAGE_CAPTURED' ? (
            <>
              <button type="button" onClick={handleRetake} className="btn btn-secondary">
                <RefreshCw size={16} /> Retake Photo
              </button>
              <button type="button" onClick={handleConfirm} className="btn btn-primary">
                <Check size={16} /> Use This Photo
              </button>
            </>
          ) : (
            <button
              type="button"
              id="capture-photo-btn"
              onClick={handleTakePhoto}
              className="btn btn-primary btn-lg"
              disabled={cameraState !== 'PREVIEW_ACTIVE'}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                opacity: cameraState === 'PREVIEW_ACTIVE' ? 1 : 0.6,
              }}
            >
              <Camera size={18} /> Capture Lesion
            </button>
          )}
        </div>

        {/* Guidance Note */}
        <div
          style={{
            background: '#f8fafc',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 14px',
            marginTop: '14px',
            fontSize: '0.76rem',
            color: 'var(--text-muted)',
            lineHeight: '1.5',
          }}
        >
          <strong style={{ color: 'var(--text-main)', display: 'block', marginBottom: '4px' }}>
            Camera Capture Guidance:
          </strong>
          <span>
            Keep the camera steady • Use natural or sufficient lighting • Keep the affected area in focus • Avoid shadows and flash glare • Capture the lesion clearly inside the square frame.
          </span>
        </div>
      </div>
    </div>
  );
};
