import React, { useState, useEffect } from 'react';
import { FaPlus, FaSearch, FaFilter, FaClock, FaUser, FaTag } from 'react-icons/fa';
import { Link } from 'react-router-dom';

const TicketsList = () => {
  const [tickets, setTickets] = useState([]);
  const [filteredTickets, setFilteredTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    search: '',
    status: '',
    priority: '',
    category: '',
    page: 1
  });
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 20,
    total: 0
  });

  useEffect(() => {
    fetchTickets();
  }, [filters]);

  const fetchTickets = async () => {
    try {
      setLoading(true);
      const queryParams = new URLSearchParams({
        page: filters.page,
        limit: pagination.limit,
        ...Object.fromEntries(Object.entries(filters).filter(([_, value]) => value !== ''))
      });

      const response = await fetch(`/api/tickets?${queryParams}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setTickets(data);
        setFilteredTickets(data);
        setPagination(prev => ({ ...prev, total: data.length }));
      }
    } catch (error) {
      console.error('Failed to fetch tickets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value, page: 1 }));
  };

  const handleSearch = (searchTerm) => {
    setFilters(prev => ({ ...prev, search: searchTerm }));
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'open': return '#e74c3c';
      case 'in_progress': return '#f39c12';
      case 'resolved': return '#27ae60';
      case 'closed': return '#95a5a6';
      case 'escalated': return '#8e44ad';
      default: return '#7f8c8d';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'emergency': return '#c0392b';
      case 'critical': return '#e74c3c';
      case 'high': return '#e67e22';
      case 'medium': return '#f39c12';
      case 'low': return '#27ae60';
      default: return '#7f8c8d';
    }
  };

  const getTimeAgo = (date) => {
    const now = new Date();
    const ticketDate = new Date(date);
    const diffMs = now - ticketDate;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) return `${diffDays}d ago`;
    if (diffHours > 0) return `${diffHours}h ago`;
    return 'Just now';
  };

  return (
    <div className="tickets-list-container">
      <div className="page-header">
        <h1>Support Tickets</h1>
        <Link to="/tickets/new" className="btn-primary">
          <FaPlus />
          Create Ticket
        </Link>
      </div>

      {/* Filters and Search */}
      <div className="filters-section">
        <div className="search-box">
          <FaSearch className="search-icon" />
          <input
            type="text"
            placeholder="Search tickets..."
            value={filters.search}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>

        <div className="filter-controls">
          <select
            value={filters.status}
            onChange={(e) => handleFilterChange('status', e.target.value)}
          >
            <option value="">All Statuses</option>
            <option value="open">Open</option>
            <option value="in_progress">In Progress</option>
            <option value="resolved">Resolved</option>
            <option value="closed">Closed</option>
            <option value="escalated">Escalated</option>
          </select>

          <select
            value={filters.priority}
            onChange={(e) => handleFilterChange('priority', e.target.value)}
          >
            <option value="">All Priorities</option>
            <option value="emergency">Emergency</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>

          <select
            value={filters.category}
            onChange={(e) => handleFilterChange('category', e.target.value)}
          >
            <option value="">All Categories</option>
            <option value="technical">Technical</option>
            <option value="billing">Billing</option>
            <option value="training">Training</option>
            <option value="medical">Medical</option>
            <option value="account">Account</option>
            <option value="feature_request">Feature Request</option>
          </select>
        </div>
      </div>

      {/* Tickets Table */}
      <div className="tickets-table">
        {loading ? (
          <div className="loading">Loading tickets...</div>
        ) : filteredTickets.length === 0 ? (
          <div className="no-tickets">
            <p>No tickets found matching your criteria.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Ticket</th>
                <th>Status</th>
                <th>Priority</th>
                <th>Category</th>
                <th>Created By</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTickets.map(ticket => (
                <tr key={ticket.id}>
                  <td className="ticket-info-cell">
                    <div className="ticket-title">
                      <Link to={`/tickets/${ticket.id}`}>
                        #{ticket.ticket_number}
                      </Link>
                    </div>
                    <div className="ticket-description">
                      <p>{ticket.title}</p>
                    </div>
                    {ticket.medical_case_id && (
                      <div className="medical-badge">
                        <FaTag />
                        Medical Case
                      </div>
                    )}
                  </td>
                  
                  <td>
                    <span 
                      className="status-badge"
                      style={{ backgroundColor: getStatusColor(ticket.status) }}
                    >
                      {ticket.status.replace('_', ' ')}
                    </span>
                  </td>
                  
                  <td>
                    <span 
                      className="priority-badge"
                      style={{ backgroundColor: getPriorityColor(ticket.priority) }}
                    >
                      {ticket.priority}
                    </span>
                  </td>
                  
                  <td>
                    <span className="category-badge">
                      {ticket.category}
                    </span>
                  </td>
                  
                  <td>
                    <div className="user-info">
                      <FaUser className="user-icon" />
                      <span>{ticket.user_name}</span>
                    </div>
                  </td>
                  
                  <td>
                    <div className="created-info">
                      <FaClock className="time-icon" />
                      <div>
                        <div>{getTimeAgo(ticket.created_at)}</div>
                        <small>{new Date(ticket.created_at).toLocaleDateString()}</small>
                      </div>
                    </div>
                  </td>
                  
                  <td>
                    <div className="action-buttons">
                      <Link 
                        to={`/tickets/${ticket.id}`}
                        className="btn-view"
                      >
                        View
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Summary Statistics */}
      <div className="tickets-summary">
        <h3>Summary</h3>
        <div className="summary-stats">
          <div className="stat">
            <span className="stat-number">
              {filteredTickets.filter(t => t.status === 'open').length}
            </span>
            <span className="stat-label">Open</span>
          </div>
          <div className="stat">
            <span className="stat-number">
              {filteredTickets.filter(t => t.status === 'in_progress').length}
            </span>
            <span className="stat-label">In Progress</span>
          </div>
          <div className="stat">
            <span className="stat-number">
              {filteredTickets.filter(t => t.priority === 'critical' || t.priority === 'emergency').length}
            </span>
            <span className="stat-label">Critical</span>
          </div>
          <div className="stat">
            <span className="stat-number">
              {filteredTickets.filter(t => t.category === 'medical').length}
            </span>
            <span className="stat-label">Medical</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TicketsList;