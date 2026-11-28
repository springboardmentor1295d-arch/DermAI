import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import { NotificationItem } from '../types';
import { 
  Activity, PlusCircle, History, BarChart2, Shield, LogOut, 
  Bell, User, Menu, X, CheckCircle 
} from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [showNotifMenu, setShowNotifMenu] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    if (user) {
      api.getNotifications().then(setNotifications).catch(() => {});
    }
  }, [user, location.pathname]);

  const unreadCount = notifications.filter(n => !n.is_read).length;

  const handleMarkRead = async (id: string, link?: string) => {
    await api.markNotificationRead(id);
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, is_read: true } : n));
    if (link) {
      setShowNotifMenu(false);
      navigate(link);
    }
  };

  const getRoleBadgeColor = (role?: string) => {
    switch (role) {
      case 'admin': return 'badge-danger';
      case 'clinician': return 'badge-success';
      case 'researcher': return 'badge-info';
      case 'ml_engineer': return 'badge-warning';
      default: return 'badge-info';
    }
  };

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 50,
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border-subtle)',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
    }}>
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '64px' }}>
        
        {/* Brand Logo */}
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 6px rgba(13, 148, 136, 0.25)'
          }}>
            <Activity size={20} color="#ffffff" />
          </div>
          <div>
            <span style={{ fontSize: '1.2rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-main)' }}>
              Derm<span style={{ color: 'var(--primary)' }}>AI</span>
            </span>
            <span style={{ display: 'block', fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginTop: '-3px' }}>
              Dermatological AI Assistant
            </span>
          </div>
        </Link>

        {/* Desktop Nav Links */}
        {user ? (
          <nav style={{ display: 'flex', alignItems: 'center', gap: '8px' }} className="desktop-nav">
            <Link to="/dashboard" className={`btn btn-sm ${location.pathname === '/dashboard' ? 'btn-primary' : 'btn-secondary'}`}>
              <Activity size={15} /> Dashboard
            </Link>
            <Link to="/new-case" className={`btn btn-sm ${location.pathname === '/new-case' ? 'btn-primary' : 'btn-secondary'}`}>
              <PlusCircle size={15} /> New Analysis
            </Link>
            <Link to="/history" className={`btn btn-sm ${location.pathname === '/history' ? 'btn-primary' : 'btn-secondary'}`}>
              <History size={15} /> Case History
            </Link>
            <Link to="/evaluation" className={`btn btn-sm ${location.pathname === '/evaluation' ? 'btn-primary' : 'btn-secondary'}`}>
              <BarChart2 size={15} /> Model Metrics
            </Link>
            {user.role === 'admin' && (
              <Link to="/admin" className={`btn btn-sm ${location.pathname === '/admin' ? 'btn-danger' : 'btn-secondary'}`}>
                <Shield size={15} /> Admin Portal
              </Link>
            )}
          </nav>
        ) : (
          <nav style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Link to="/evaluation" className="btn btn-secondary btn-sm">
              <BarChart2 size={15} /> Research Metrics
            </Link>
            <Link to="/login" className="btn btn-secondary btn-sm">
              Sign In
            </Link>
            <Link to="/register" className="btn btn-primary btn-sm">
              Get Started
            </Link>
          </nav>
        )}

        {/* User & Notifications */}
        {user && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* Notification Bell */}
            <div style={{ position: 'relative' }}>
              <button
                type="button"
                onClick={() => setShowNotifMenu(!showNotifMenu)}
                style={{
                  background: 'none',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '6px',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  position: 'relative'
                }}
                aria-label="Notifications"
              >
                <Bell size={18} />
                {unreadCount > 0 && (
                  <span style={{
                    position: 'absolute',
                    top: '-4px',
                    right: '-4px',
                    width: '16px',
                    height: '16px',
                    background: '#ef4444',
                    color: 'white',
                    borderRadius: '50%',
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {unreadCount}
                  </span>
                )}
              </button>

              {/* Notification Dropdown */}
              {showNotifMenu && (
                <div style={{
                  position: 'absolute',
                  right: 0,
                  top: '100%',
                  marginTop: '8px',
                  width: '320px',
                  background: '#ffffff',
                  border: '1px solid var(--border-medium)',
                  borderRadius: 'var(--radius-md)',
                  boxShadow: 'var(--shadow-lg)',
                  padding: '12px',
                  zIndex: 60
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '6px' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-main)' }}>Notifications</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{unreadCount} unread</span>
                  </div>

                  {notifications.length === 0 ? (
                    <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                      No notifications
                    </div>
                  ) : (
                    <div style={{ maxHeight: '240px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {notifications.map(n => (
                        <div
                          key={n.id}
                          onClick={() => handleMarkRead(n.id, n.link)}
                          style={{
                            padding: '8px 10px',
                            borderRadius: 'var(--radius-sm)',
                            background: n.is_read ? '#ffffff' : 'var(--primary-subtle)',
                            cursor: 'pointer',
                            fontSize: '0.78rem',
                            border: '1px solid',
                            borderColor: n.is_read ? 'transparent' : 'var(--border-subtle)'
                          }}
                        >
                          <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '2px' }}>{n.title}</div>
                          <div style={{ color: 'var(--text-muted)' }}>{n.message}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Profile Link & Logout */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Link to="/profile" style={{ display: 'flex', alignItems: 'center', gap: '6px', textDecoration: 'none', color: 'var(--text-main)', fontSize: '0.85rem', fontWeight: 600 }}>
                <div style={{
                  width: '30px',
                  height: '30px',
                  borderRadius: '50%',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-medium)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-dark)'
                }}>
                  <User size={16} />
                </div>
                <span className="desktop-nav">{user.full_name.split(' ')[0]}</span>
              </Link>

              <button
                type="button"
                onClick={logout}
                className="btn btn-secondary btn-sm"
                title="Sign Out"
                style={{ padding: '6px 10px' }}
              >
                <LogOut size={14} />
              </button>
            </div>
          </div>
        )}

      </div>
    </header>
  );
};
