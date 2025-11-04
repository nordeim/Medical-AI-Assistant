import React, { useState, useEffect } from 'react';
import { 
  FaHeartbeat, 
  FaServer, 
  FaDatabase, 
  FaExclamationTriangle,
  FaCheckCircle,
  FaClock,
  FaBell,
  FaCog
} from 'react-icons/fa';

const HealthMonitor = () => {
  const [healthData, setHealthData] = useState(null);
  const [recentChecks, setRecentChecks] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    fetchHealthData();
    const interval = setInterval(fetchHealthData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchHealthData = async () => {
    try {
      const response = await fetch('/api/health', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setHealthData(data);
        setLastUpdate(new Date());
        processHealthData(data);
      }
    } catch (error) {
      console.error('Failed to fetch health data:', error);
    } finally {
      setLoading(false);
    }
  };

  const processHealthData = (data) => {
    const services = data.services || [];
    
    // Generate recent checks
    const recent = services.map(service => ({
      service: service.service,
      status: service.status,
      response_time: service.response_time_ms,
      timestamp: service.checked_at,
      error: service.error
    }));
    
    setRecentChecks(recent);
    
    // Generate alerts for unhealthy services
    const unhealthy = services.filter(service => 
      service.status === 'unhealthy' || service.status === 'degraded'
    );
    
    const newAlerts = unhealthy.map(service => ({
      id: Date.now() + Math.random(),
      service: service.service,
      severity: service.status === 'unhealthy' ? 'critical' : 'warning',
      message: service.error || 'Service degradation detected',
      timestamp: service.checked_at
    }));
    
    setAlerts(newAlerts);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return '#27ae60';
      case 'degraded': return '#f39c12';
      case 'unhealthy': return '#e74c3c';
      default: return '#95a5a6';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <FaCheckCircle className="status-healthy" />;
      case 'degraded': return <FaExclamationTriangle className="status-warning" />;
      case 'unhealthy': return <FaExclamationTriangle className="status-critical" />;
      default: return <FaClock className="status-unknown" />;
    }
  };

  const formatResponseTime = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  if (loading) {
    return <div className="loading">Loading health monitor...</div>;
  }

  return (
    <div className="health-monitor-container">
      <div className="page-header">
        <h1>System Health Monitor</h1>
        <div className="last-update">
          Last updated: {lastUpdate?.toLocaleTimeString()}
        </div>
      </div>

      {/* Overall System Status */}
      <div className="overall-status">
        <div className="status-card">
          <div className="status-header">
            <FaHeartbeat className="system-icon" />
            <h2>Overall System Status</h2>
          </div>
          <div className="status-content">
            <div className={`system-status ${healthData?.status}`}>
              {getStatusIcon(healthData?.status)}
              <span className="status-text">
                {healthData?.status?.toUpperCase() || 'UNKNOWN'}
              </span>
            </div>
            <div className="system-metrics">
              <div className="metric">
                <span className="metric-label">Uptime</span>
                <span className="metric-value">
                  {Math.floor((healthData?.uptime || 0) / 3600)}h {(Math.floor((healthData?.uptime || 0) % 3600) / 60).toFixed(0)}m
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Services</span>
                <span className="metric-value">
                  {healthData?.services?.filter(s => s.status === 'healthy').length || 0} / {healthData?.services?.length || 0} healthy
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="alerts-section">
          <h2>
            <FaBell />
            Active Alerts ({alerts.length})
          </h2>
          <div className="alerts-list">
            {alerts.map(alert => (
              <div key={alert.id} className={`alert alert-${alert.severity}`}>
                <div className="alert-header">
                  <span className="alert-service">{alert.service}</span>
                  <span className="alert-time">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="alert-message">
                  {alert.message}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Services Status */}
      <div className="services-status">
        <h2>Service Status</h2>
        <div className="services-grid">
          {healthData?.services?.map(service => (
            <div key={service.service} className="service-card">
              <div className="service-header">
                <div className="service-info">
                  <FaServer className="service-icon" />
                  <h3>{service.service.replace('_', ' ')}</h3>
                </div>
                <div className={`service-status ${service.status}`}>
                  {getStatusIcon(service.status)}
                  <span>{service.status}</span>
                </div>
              </div>
              
              <div className="service-metrics">
                <div className="service-metric">
                  <span className="metric-label">Response Time</span>
                  <span className="metric-value">
                    {service.response_time_ms ? formatResponseTime(service.response_time_ms) : 'N/A'}
                  </span>
                </div>
                <div className="service-metric">
                  <span className="metric-label">Last Check</span>
                  <span className="metric-value">
                    {new Date(service.checked_at).toLocaleTimeString()}
                  </span>
                </div>
              </div>
              
              {service.error && (
                <div className="service-error">
                  <FaExclamationTriangle />
                  <span>{service.error}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Recent Checks Timeline */}
      <div className="recent-checks">
        <h2>Recent Health Checks</h2>
        <div className="checks-timeline">
          {recentChecks.map((check, index) => (
            <div key={index} className={`check-item ${check.status}`}>
              <div className="check-icon">
                {getStatusIcon(check.status)}
              </div>
              <div className="check-content">
                <div className="check-header">
                  <span className="check-service">{check.service.replace('_', ' ')}</span>
                  <span className="check-time">
                    {new Date(check.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="check-details">
                  <span className={`status-badge ${check.status}`}>
                    {check.status}
                  </span>
                  {check.response_time && (
                    <span className="response-time">
                      {formatResponseTime(check.response_time)}
                    </span>
                  )}
                </div>
                {check.error && (
                  <div className="check-error">
                    {check.error}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Resources */}
      {healthData?.memory && (
        <div className="system-resources">
          <h2>System Resources</h2>
          <div className="resources-grid">
            <div className="resource-card">
              <h3>Memory Usage</h3>
              <div className="resource-metric">
                <span className="metric-label">Used</span>
                <span className="metric-value">
                  {(healthData.memory.heapUsed / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
              <div className="resource-metric">
                <span className="metric-label">Total</span>
                <span className="metric-value">
                  {(healthData.memory.heapTotal / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
              <div className="resource-bar">
                <div 
                  className="resource-fill" 
                  style={{
                    width: `${(healthData.memory.heapUsed / healthData.memory.heapTotal) * 100}%`
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Health Check Controls */}
      <div className="health-controls">
        <h2>Health Check Settings</h2>
        <div className="controls-grid">
          <div className="control-group">
            <label>Check Interval</label>
            <select defaultValue="60">
              <option value="30">30 seconds</option>
              <option value="60">1 minute</option>
              <option value="300">5 minutes</option>
              <option value="900">15 minutes</option>
            </select>
          </div>
          <div className="control-group">
            <label>Alert Thresholds</label>
            <div className="threshold-controls">
              <input type="number" placeholder="Response time (ms)" defaultValue="5000" />
              <input type="number" placeholder="Error rate (%)" defaultValue="10" />
            </div>
          </div>
          <div className="control-group">
            <label>Notifications</label>
            <div className="notification-toggles">
              <label>
                <input type="checkbox" defaultChecked />
                Email alerts
              </label>
              <label>
                <input type="checkbox" defaultChecked />
                Slack notifications
              </label>
              <label>
                <input type="checkbox" defaultChecked />
                SMS for critical issues
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HealthMonitor;