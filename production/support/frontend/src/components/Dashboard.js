import React, { useState, useEffect } from 'react';
import { 
  FaTicketAlt, 
  FaHeartbeat, 
  FaComments, 
  FaChartLine, 
  FaClock,
  FaExclamationTriangle,
  FaCheckCircle,
  FaUsers
} from 'react-icons/fa';

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [recentTickets, setRecentTickets] = useState([]);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [metricsRes, ticketsRes, healthRes] = await Promise.all([
        fetch('/api/metrics', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }),
        fetch('/api/tickets?limit=10', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        }),
        fetch('/api/health', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
        })
      ]);

      if (metricsRes.ok) setMetrics(await metricsRes.json());
      if (ticketsRes.ok) setRecentTickets(await ticketsRes.json());
      if (healthRes.ok) setHealthStatus(await healthRes.json());
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'open': return '#e74c3c';
      case 'in_progress': return '#f39c12';
      case 'resolved': return '#27ae60';
      case 'closed': return '#95a5a6';
      default: return '#7f8c8d';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'emergency': return '#e74c3c';
      case 'critical': return '#c0392b';
      case 'high': return '#e67e22';
      case 'medium': return '#f39c12';
      case 'low': return '#27ae60';
      default: return '#7f8c8d';
    }
  };

  if (!metrics || !healthStatus) {
    return <div className="loading">Loading dashboard...</div>;
  }

  const activeTickets = recentTickets.filter(t => t.status !== 'closed' && t.status !== 'resolved').length;
  const criticalIssues = recentTickets.filter(t => t.priority === 'critical' || t.priority === 'emergency').length;

  return (
    <div className="dashboard">
      <h1>Support Dashboard</h1>
      
      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">
            <FaTicketAlt />
          </div>
          <div className="metric-content">
            <h3>{activeTickets}</h3>
            <p>Active Tickets</p>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon" style={{ color: '#e74c3c' }}>
            <FaExclamationTriangle />
          </div>
          <div className="metric-content">
            <h3>{criticalIssues}</h3>
            <p>Critical Issues</p>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon" style={{ color: '#27ae60' }}>
            <FaHeartbeat />
          </div>
          <div className="metric-content">
            <h3>{healthStatus.status}</h3>
            <p>System Health</p>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon" style={{ color: '#3498db' }}>
            <FaUsers />
          </div>
          <div className="metric-content">
            <h3>24/7</h3>
            <p>Support Available</p>
          </div>
        </div>
      </div>

      {/* System Health Status */}
      <div className="health-overview">
        <h2>System Health</h2>
        <div className="health-status">
          <div className={`health-indicator ${healthStatus.status}`}>
            <FaHeartbeat className="health-icon" />
            <span>{healthStatus.status.toUpperCase()}</span>
          </div>
          <div className="health-details">
            <p>Uptime: {Math.floor(healthStatus.uptime / 3600)} hours</p>
            <p>Services monitored: {healthStatus.services?.length || 0}</p>
          </div>
        </div>
      </div>

      {/* Recent Tickets */}
      <div className="recent-tickets">
        <div className="section-header">
          <h2>Recent Tickets</h2>
          <a href="/tickets" className="view-all">View All</a>
        </div>
        
        <div className="tickets-list">
          {recentTickets.length === 0 ? (
            <p>No recent tickets</p>
          ) : (
            recentTickets.map(ticket => (
              <div key={ticket.id} className="ticket-item">
                <div className="ticket-info">
                  <div className="ticket-header">
                    <span 
                      className="status-badge" 
                      style={{ backgroundColor: getStatusColor(ticket.status) }}
                    >
                      {ticket.status.replace('_', ' ')}
                    </span>
                    <span 
                      className="priority-badge" 
                      style={{ backgroundColor: getPriorityColor(ticket.priority) }}
                    >
                      {ticket.priority}
                    </span>
                    <span className="ticket-number">#{ticket.ticket_number}</span>
                  </div>
                  <h4>{ticket.title}</h4>
                  <p>{ticket.category} • {ticket.user_name}</p>
                </div>
                <div className="ticket-time">
                  <FaClock />
                  <span>{new Date(ticket.created_at).toLocaleString()}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <h2>Quick Actions</h2>
        <div className="action-buttons">
          <a href="/tickets/new" className="action-btn primary">
            <FaTicketAlt />
            Create New Ticket
          </a>
          <a href="/feedback" className="action-btn">
            <FaComments />
            Submit Feedback
          </a>
          <a href="/knowledge" className="action-btn">
            <FaChartLine />
            Browse Knowledge Base
          </a>
          <a href="/training" className="action-btn">
            <FaComments />
            Access Training
          </a>
        </div>
      </div>

      {/* Emergency Contact */}
      <div className="emergency-contact">
        <div className="emergency-header">
          <FaExclamationTriangle className="emergency-icon" />
          <h2>Medical Emergency</h2>
        </div>
        <div className="emergency-content">
          <p>For medical emergencies requiring immediate assistance:</p>
          <div className="emergency-contacts">
            <div className="contact-item">
              <strong>Phone:</strong> +1-800-MEDICAL
            </div>
            <div className="contact-item">
              <strong>Email:</strong> medical@yourdomain.com
            </div>
            <div className="contact-item">
              <strong>SLA:</strong> 30 minutes response time
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;