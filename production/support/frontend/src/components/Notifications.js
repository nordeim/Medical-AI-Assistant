import React, { useState, useEffect } from 'react';
import { FaBell, FaCheck, FaTrash, FaExclamationTriangle, FaInfoCircle, FaCheckCircle } from 'react-icons/fa';

const Notifications = () => {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [selectedNotifications, setSelectedNotifications] = useState([]);

  useEffect(() => {
    fetchNotifications();
  }, [filter]);

  const fetchNotifications = async () => {
    try {
      const response = await fetch('/api/notifications', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        let filtered = data;
        
        if (filter !== 'all') {
          filtered = data.filter(n => n.type === filter);
        }
        
        setNotifications(filtered);
      }
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    } finally {
      setLoading(false);
    }
  };

  const markAsRead = async (notificationId) => {
    try {
      await fetch(`/api/notifications/${notificationId}/read`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      fetchNotifications();
    } catch (error) {
      console.error('Failed to mark as read:', error);
    }
  };

  const markAllAsRead = async () => {
    try {
      await fetch('/api/notifications/mark-all-read', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      fetchNotifications();
    } catch (error) {
      console.error('Failed to mark all as read:', error);
    }
  };

  const deleteNotification = async (notificationId) => {
    try {
      await fetch(`/api/notifications/${notificationId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      fetchNotifications();
    } catch (error) {
      console.error('Failed to delete notification:', error);
    }
  };

  const deleteSelected = async () => {
    try {
      await Promise.all(
        selectedNotifications.map(id =>
          fetch(`/api/notifications/${id}`, {
            method: 'DELETE',
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          })
        )
      );
      
      setSelectedNotifications([]);
      fetchNotifications();
    } catch (error) {
      console.error('Failed to delete selected notifications:', error);
    }
  };

  const handleSelectNotification = (notificationId) => {
    setSelectedNotifications(prev => 
      prev.includes(notificationId)
        ? prev.filter(id => id !== notificationId)
        : [...prev, notificationId]
    );
  };

  const handleSelectAll = () => {
    if (selectedNotifications.length === notifications.length) {
      setSelectedNotifications([]);
    } else {
      setSelectedNotifications(notifications.map(n => n.id));
    }
  };

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'emergency':
        return <FaExclamationTriangle className="icon-emergency" />;
      case 'success':
        return <FaCheckCircle className="icon-success" />;
      case 'warning':
        return <FaExclamationTriangle className="icon-warning" />;
      case 'info':
      default:
        return <FaInfoCircle className="icon-info" />;
    }
  };

  const getNotificationColor = (type) => {
    switch (type) {
      case 'emergency':
        return '#e74c3c';
      case 'success':
        return '#27ae60';
      case 'warning':
        return '#f39c12';
      case 'info':
      default:
        return '#3498db';
    }
  };

  const filters = [
    { value: 'all', label: 'All Notifications' },
    { value: 'ticket', label: 'Tickets' },
    { value: 'system', label: 'System' },
    { value: 'training', label: 'Training' },
    { value: 'security', label: 'Security' },
    { value: 'medical', label: 'Medical' }
  ];

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="notifications-container">
      <div className="page-header">
        <h1>
          <FaBell />
          Notifications
        </h1>
        <div className="header-actions">
          {unreadCount > 0 && (
            <button onClick={markAllAsRead} className="btn-secondary">
              <FaCheck />
              Mark All Read
            </button>
          )}
          {selectedNotifications.length > 0 && (
            <button onClick={deleteSelected} className="btn-danger">
              <FaTrash />
              Delete Selected ({selectedNotifications.length})
            </button>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="notification-filters">
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          {filters.map(filter => (
            <option key={filter.value} value={filter.value}>
              {filter.label}
            </option>
          ))}
        </select>
      </div>

      {/* Notifications List */}
      <div className="notifications-list">
        {loading ? (
          <div className="loading">Loading notifications...</div>
        ) : notifications.length === 0 ? (
          <div className="no-notifications">
            <FaBell />
            <h3>No notifications</h3>
            <p>You're all caught up!</p>
          </div>
        ) : (
          <div className="notifications-table">
            <div className="table-header">
              <input
                type="checkbox"
                checked={selectedNotifications.length === notifications.length}
                onChange={handleSelectAll}
              />
              <span>Type</span>
              <span>Message</span>
              <span>Date</span>
              <span>Actions</span>
            </div>
            
            {notifications.map(notification => (
              <div 
                key={notification.id} 
                className={`notification-row ${!notification.read ? 'unread' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={selectedNotifications.includes(notification.id)}
                  onChange={() => handleSelectNotification(notification.id)}
                />
                
                <div className="notification-type">
                  {getNotificationIcon(notification.type)}
                </div>
                
                <div className="notification-content">
                  <div className="notification-message">
                    {notification.message}
                  </div>
                  <div className="notification-meta">
                    <span className="notification-source">
                      {notification.source}
                    </span>
                    {notification.priority && (
                      <span className={`priority-badge ${notification.priority}`}>
                        {notification.priority}
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="notification-date">
                  {new Date(notification.created_at).toLocaleDateString()}
                  <br />
                  <small>{new Date(notification.created_at).toLocaleTimeString()}</small>
                </div>
                
                <div className="notification-actions">
                  {!notification.read && (
                    <button
                      onClick={() => markAsRead(notification.id)}
                      className="btn-mark-read"
                      title="Mark as read"
                    >
                      <FaCheck />
                    </button>
                  )}
                  <button
                    onClick={() => deleteNotification(notification.id)}
                    className="btn-delete"
                    title="Delete notification"
                  >
                    <FaTrash />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Notification Settings */}
      <div className="notification-settings">
        <h2>Notification Settings</h2>
        <div className="settings-grid">
          <div className="setting-group">
            <h3>Email Notifications</h3>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Receive email notifications for new tickets</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>System maintenance notifications</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" />
              <span>Product updates and announcements</span>
            </label>
          </div>

          <div className="setting-group">
            <h3>Medical Emergency Alerts</h3>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Immediate SMS for medical emergencies</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Email alerts for SLA breaches</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Push notifications for critical issues</span>
            </label>
          </div>

          <div className="setting-group">
            <h3>Training & Certification</h3>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Course completion notifications</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Certificate renewal reminders</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" />
              <span>New training materials available</span>
            </label>
          </div>

          <div className="setting-group">
            <h3>Security Notifications</h3>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Failed login attempts</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Password changes</span>
            </label>
            <label className="toggle-item">
              <input type="checkbox" defaultChecked />
              <span>Security alerts</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Notifications;