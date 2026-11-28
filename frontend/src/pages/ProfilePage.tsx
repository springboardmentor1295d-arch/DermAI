import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import { User, Shield, Key, Save, CheckCircle2 } from 'lucide-react';

export const ProfilePage: React.FC = () => {
  const { user, refreshUser } = useAuth();
  const [fullName, setFullName] = useState(user?.full_name || '');
  const [email, setEmail] = useState(user?.email || '');
  const [password, setPassword] = useState('');
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const res = await fetch('/api/v1/users/me', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('dermai_token')}`
        },
        body: JSON.stringify({
          full_name: fullName || undefined,
          email: email || undefined,
          password: password || undefined,
        })
      });
      if (!res.ok) throw new Error('Failed to update profile');
      await refreshUser();
      setSaved(true);
      setPassword('');
      setTimeout(() => setSaved(false), 2500);
    } catch (err: any) {
      setError(err.message || 'Update failed');
    }
  };

  return (
    <div className="container" style={{ maxWidth: '640px', padding: '40px 16px' }}>
      <div className="glass-panel" style={{ padding: '32px', borderRadius: 'var(--radius-xl)', background: '#ffffff', boxShadow: 'var(--shadow-md)' }}>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '24px' }}>
          <div style={{
            width: '50px',
            height: '50px',
            borderRadius: '12px',
            background: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff'
          }}>
            <User size={26} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.5rem', margin: 0, color: 'var(--text-main)', fontWeight: 800 }}>User Profile & Preferences</h2>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Assigned Role: <b style={{ color: 'var(--primary-dark)' }}>{user?.role.toUpperCase()}</b>
            </span>
          </div>
        </div>

        {saved && (
          <div style={{ background: 'var(--success-bg)', border: '1px solid var(--success-border)', color: 'var(--success)', padding: '12px', borderRadius: 'var(--radius-md)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={16} /> Profile successfully updated.
          </div>
        )}

        {error && (
          <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', color: 'var(--danger)', padding: '12px', borderRadius: 'var(--radius-md)', marginBottom: '16px' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Full Name</label>
            <input
              type="text"
              className="form-input"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              type="email"
              className="form-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Change Password (leave blank to keep current)</label>
            <input
              type="password"
              className="form-input"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '10px' }}>
            <Save size={16} /> Update Profile
          </button>
        </form>

      </div>

    </div>
  );
};
