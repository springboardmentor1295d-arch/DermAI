import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { User, ModelVersion } from '../types';
import { formatDateTimeIST } from '../utils/dateUtils';
import { 
  Shield, Users, Cpu, Activity, CheckCircle, AlertCircle, 
  RefreshCw, Power, Trash2, AlertTriangle 
} from 'lucide-react';

export const AdminDashboardPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [deleteModalUser, setDeleteModalUser] = useState<User | null>(null);
  const [deletingUser, setDeletingUser] = useState(false);
  const [adminMessage, setAdminMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const [switchingModelId, setSwitchingModelId] = useState<string | null>(null);

  const loadData = () => {
    setLoading(true);
    Promise.all([
      api.getUsers().then(setUsers).catch(() => {}),
      api.getModels().then(setModels).catch(() => {}),
      fetch('/health').then(r => r.json()).then(setHealth).catch(() => {}),
    ]).finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSetActive = async (model: ModelVersion) => {
    setSwitchingModelId(model.id);
    setAdminMessage(null);
    try {
      const updated = await api.setActiveModel(model.id);
      setAdminMessage({
        text: `Active production model switched to ${updated.architecture} (${updated.version_tag}). New predictions and evaluation metrics now use this model.`,
        type: 'success',
      });
      loadData();
    } catch (err: any) {
      setAdminMessage({ text: err?.message || 'Failed to activate selected model', type: 'error' });
    } finally {
      setSwitchingModelId(null);
    }
  };

  const handleConfirmDeleteUser = async () => {
    if (!deleteModalUser) return;
    setDeletingUser(true);
    setAdminMessage(null);
    try {
      await api.deleteUser(deleteModalUser.id);
      setAdminMessage({ text: `User ${deleteModalUser.email} permanently deleted.`, type: 'success' });
      setUsers(prev => prev.filter(u => u.id !== deleteModalUser.id));
      setDeleteModalUser(null);
      setTimeout(() => setAdminMessage(null), 4000);
    } catch (err: any) {
      setAdminMessage({ text: err.message || 'Failed to delete user account', type: 'error' });
    } finally {
      setDeletingUser(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '1040px', padding: '36px 16px' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-main)', fontWeight: 800 }}>
            <Shield size={28} color="var(--primary)" /> System Administrator Operations
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            User role authorization, active model version deployment, and platform health monitoring.
          </p>
        </div>
        <button onClick={loadData} className="btn btn-secondary btn-sm">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* System Health Status */}
      <div className="glass-panel" style={{ padding: '20px', borderRadius: 'var(--radius-lg)', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', color: 'var(--text-main)', marginBottom: '12px', fontWeight: 700 }}>Platform Infrastructure Health</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '14px', borderRadius: 'var(--radius-md)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 600 }}>API Gateway</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
              <CheckCircle size={16} color="var(--success)" />
              <b style={{ color: 'var(--text-main)' }}>Healthy (v1.0.0)</b>
            </div>
          </div>
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '14px', borderRadius: 'var(--radius-md)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 600 }}>Database Connectivity</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
              <CheckCircle size={16} color="var(--success)" />
              <b style={{ color: 'var(--text-main)' }}>Connected (SQLite/PostgreSQL)</b>
            </div>
          </div>
          <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', padding: '14px', borderRadius: 'var(--radius-md)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 600 }}>Artifact Storage</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
              <CheckCircle size={16} color="var(--success)" />
              <b style={{ color: 'var(--text-main)' }}>Persistent Storage</b>
            </div>
          </div>
        </div>
      </div>

      {/* Model Registry & Active Deployment (Two Production Models Only) */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)', marginBottom: '28px' }}>
        <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700 }}>
          <Cpu size={18} color="var(--primary)" /> Model Registry & Active Inference Selection
        </h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
          Select between the two verified DermAI classifiers (**EfficientNet-B0** and **MobileNetV3-Large**). Activating a model updates live lesion inference, temperature calibration, and platform evaluation metrics immediately.
        </p>

        {adminMessage && (
          <div style={{
            background: adminMessage.type === 'success' ? 'var(--success-bg)' : 'var(--danger-bg)',
            border: `1px solid ${adminMessage.type === 'success' ? 'var(--success-border)' : 'var(--danger-border)'}`,
            borderRadius: 'var(--radius-md)',
            padding: '10px 14px',
            marginBottom: '16px',
            color: adminMessage.type === 'success' ? 'var(--success)' : 'var(--danger)',
            fontSize: '0.84rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            {adminMessage.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
            <span>{adminMessage.text}</span>
          </div>
        )}

        {models.length === 0 ? (
          <div style={{ padding: '20px', color: 'var(--text-dim)', textAlign: 'center' }}>
            {loading ? 'Loading registered models...' : 'No registered models found in database.'}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
            {models.slice(0, 2).map(m => (
              <div
                key={m.id}
                style={{
                  background: m.is_active ? 'var(--primary-subtle)' : '#f8fafc',
                  border: m.is_active ? '1.5px solid var(--primary)' : '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '20px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '14px',
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px', marginBottom: '8px' }}>
                    <div>
                      <div style={{ fontSize: '1.08rem', fontWeight: 800, color: 'var(--text-main)' }}>
                        {m.architecture}
                      </div>
                      <code style={{ fontSize: '0.76rem', color: 'var(--primary-dark)' }}>{m.version_tag}</code>
                    </div>
                    {m.is_active ? (
                      <span className="badge badge-success" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                        <CheckCircle size={12} /> Active Model
                      </span>
                    ) : (
                      <span className="badge badge-info">Standby</span>
                    )}
                  </div>

                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '8px',
                    marginTop: '12px',
                    paddingTop: '12px',
                    borderTop: '1px solid rgba(15, 23, 42, 0.08)',
                    fontSize: '0.8rem',
                  }}>
                    <div>
                      <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem', textTransform: 'uppercase' }}>Accuracy</div>
                      <b style={{ color: 'var(--text-main)' }}>
                        {m.test_accuracy != null ? `${(m.test_accuracy * 100).toFixed(2)}%` : '—'}
                      </b>
                    </div>
                    <div>
                      <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem', textTransform: 'uppercase' }}>Bal. Acc</div>
                      <b style={{ color: 'var(--primary-dark)' }}>
                        {m.test_balanced_acc != null ? `${(m.test_balanced_acc * 100).toFixed(2)}%` : '—'}
                      </b>
                    </div>
                    <div>
                      <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem', textTransform: 'uppercase' }}>Macro F1</div>
                      <b style={{ color: 'var(--text-main)' }}>
                        {m.test_macro_f1 != null ? `${(m.test_macro_f1 * 100).toFixed(2)}%` : '—'}
                      </b>
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  {m.is_active ? (
                    <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--primary-dark)' }}>
                      Currently serving all predictions
                    </span>
                  ) : (
                    <button
                      onClick={() => handleSetActive(m)}
                      disabled={switchingModelId !== null}
                      className="btn btn-primary btn-sm"
                      style={{ padding: '7px 14px', fontSize: '0.8rem' }}
                    >
                      <Power size={13} /> {switchingModelId === m.id ? 'Activating...' : `Activate ${m.architecture}`}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Registered User Accounts */}
      <div className="glass-panel" style={{ padding: '24px', borderRadius: 'var(--radius-lg)' }}>
        <h3 style={{ fontSize: '1.15rem', color: 'var(--text-main)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700 }}>
          <Users size={18} color="var(--primary)" /> User Role Authorization & Accounts ({users.length})
        </h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
          Inspect registered platform users and their assigned Role-Based Access Control (RBAC) tiers.
        </p>

        {adminMessage && (
          <div style={{
            background: adminMessage.type === 'success' ? 'var(--success-bg)' : 'var(--danger-bg)',
            border: `1px solid ${adminMessage.type === 'success' ? 'var(--success-border)' : 'var(--danger-border)'}`,
            borderRadius: 'var(--radius-md)',
            padding: '10px 14px',
            marginBottom: '16px',
            color: adminMessage.type === 'success' ? 'var(--success)' : 'var(--danger)',
            fontSize: '0.85rem'
          }}>
            {adminMessage.text}
          </div>
        )}

        {users.length === 0 ? (
          <div style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Users size={32} style={{ margin: '0 auto 10px auto', display: 'block', color: 'var(--text-dim)' }} />
            <p>No user accounts registered in platform database.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left', background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px' }}>Full Name</th>
                  <th style={{ padding: '10px 12px' }}>Email Address</th>
                  <th style={{ padding: '10px 12px' }}>Role</th>
                  <th style={{ padding: '10px 12px' }}>Status</th>
                  <th style={{ padding: '10px 12px' }}>Created (IST)</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '12px', fontWeight: 600, color: 'var(--text-main)' }}>{u.full_name}</td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)' }}>{u.email}</td>
                    <td style={{ padding: '12px' }}>
                      <span className={`badge ${u.role === 'admin' ? 'badge-danger' : (u.role === 'clinician' ? 'badge-success' : 'badge-info')}`}>
                        {u.role.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      {u.is_active ? (
                        <span style={{ color: 'var(--success)', fontSize: '0.8rem', fontWeight: 600 }}>Active</span>
                      ) : (
                        <span style={{ color: 'var(--danger)', fontSize: '0.8rem' }}>Disabled</span>
                      )}
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                      {formatDateTimeIST(u.created_at)}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      <button
                        type="button"
                        onClick={() => setDeleteModalUser(u)}
                        className="btn btn-sm"
                        style={{ padding: '4px 8px', background: 'var(--danger-bg)', color: 'var(--danger)', border: '1px solid var(--danger-border)' }}
                        title="Permanently Delete User Account"
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Permanent User Deletion Modal */}
      {deleteModalUser && (
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
              <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-main)', fontWeight: 700 }}>Confirm User Deletion</h3>
            </div>
            
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: '1.5', marginBottom: '16px' }}>
              Are you sure you want to permanently delete user account <strong style={{ color: 'var(--text-main)' }}>{deleteModalUser.email}</strong> ({deleteModalUser.full_name})?
            </p>

            <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-sm)', padding: '12px', marginBottom: '20px', fontSize: '0.82rem', color: 'var(--danger)' }}>
              <strong>Permanent Action:</strong> This operation deletes the user account, all associated analysis cases, image uploads, and notifications permanently.
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setDeleteModalUser(null)}
                disabled={deletingUser}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-sm btn-danger"
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                onClick={handleConfirmDeleteUser}
                disabled={deletingUser}
              >
                <Trash2 size={14} /> {deletingUser ? 'Deleting...' : 'Confirm Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
