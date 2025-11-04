import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  FaArrowLeft, 
  FaClock, 
  FaUser, 
  FaTag, 
  FaComment, 
  FaFileAlt,
  FaCheckCircle,
  FaExclamationTriangle,
  FaEdit
} from 'react-icons/fa';

const TicketDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [ticket, setTicket] = useState(null);
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newComment, setNewComment] = useState('');
  const [showResolutionForm, setShowResolutionForm] = useState(false);
  const [satisfactionRating, setSatisfactionRating] = useState(5);
  const [satisfactionFeedback, setSatisfactionFeedback] = useState('');

  useEffect(() => {
    fetchTicketDetails();
    fetchComments();
  }, [id]);

  const fetchTicketDetails = async () => {
    try {
      const response = await fetch(`/api/tickets/${id}`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setTicket(data);
      }
    } catch (error) {
      console.error('Failed to fetch ticket:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchComments = async () => {
    try {
      const response = await fetch(`/api/tickets/${id}/comments`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setComments(data);
      }
    } catch (error) {
      console.error('Failed to fetch comments:', error);
    }
  };

  const handleAddComment = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch(`/api/tickets/${id}/comments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          comment: newComment,
          is_internal: false
        })
      });

      if (response.ok) {
        setNewComment('');
        fetchComments();
      }
    } catch (error) {
      console.error('Failed to add comment:', error);
    }
  };

  const handleCloseTicket = async () => {
    try {
      const response = await fetch(`/api/tickets/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          status: 'closed',
          satisfaction_rating: satisfactionRating,
          satisfaction_feedback: satisfactionFeedback
        })
      });

      if (response.ok) {
        fetchTicketDetails();
        setShowResolutionForm(false);
      }
    } catch (error) {
      console.error('Failed to close ticket:', error);
    }
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

  const getTimeElapsed = (startTime) => {
    const start = new Date(startTime);
    const now = new Date();
    const diffMs = now - start;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

    if (diffHours > 24) {
      const diffDays = Math.floor(diffHours / 24);
      return `${diffDays}d ${diffHours % 24}h`;
    }
    return `${diffHours}h ${diffMinutes}m`;
  };

  if (loading) {
    return <div className="loading">Loading ticket details...</div>;
  }

  if (!ticket) {
    return (
      <div className="ticket-not-found">
        <h2>Ticket Not Found</h2>
        <p>The requested ticket could not be found.</p>
        <button onClick={() => navigate('/tickets')} className="btn-primary">
          Back to Tickets
        </button>
      </div>
    );
  }

  return (
    <div className="ticket-detail-container">
      <div className="page-header">
        <div className="header-nav">
          <button onClick={() => navigate('/tickets')} className="btn-back">
            <FaArrowLeft />
            Back to Tickets
          </button>
        </div>
        <div className="ticket-meta">
          <h1>#{ticket.ticket_number}</h1>
          <div className="ticket-badges">
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
              {ticket.priority} priority
            </span>
          </div>
        </div>
      </div>

      {/* Emergency Alert for Medical Tickets */}
      {(ticket.priority === 'emergency' || ticket.category === 'medical') && (
        <div className="emergency-banner">
          <FaExclamationTriangle />
          <div>
            <h3>Medical Emergency Ticket</h3>
            <p>This is a medical emergency requiring immediate attention. Our medical support team has been notified.</p>
          </div>
        </div>
      )}

      <div className="ticket-content">
        <div className="ticket-main">
          <div className="ticket-header">
            <h2>{ticket.title}</h2>
            <div className="ticket-info-grid">
              <div className="info-item">
                <FaTag className="info-icon" />
                <div>
                  <span className="info-label">Category</span>
                  <span className="info-value">{ticket.category}</span>
                </div>
              </div>
              <div className="info-item">
                <FaUser className="info-icon" />
                <div>
                  <span className="info-label">Created By</span>
                  <span className="info-value">{ticket.user_name}</span>
                </div>
              </div>
              <div className="info-item">
                <FaClock className="info-icon" />
                <div>
                  <span className="info-label">Created</span>
                  <span className="info-value">
                    {new Date(ticket.created_at).toLocaleString()}
                  </span>
                </div>
              </div>
              <div className="info-item">
                <FaClock className="info-icon" />
                <div>
                  <span className="info-label">Time Elapsed</span>
                  <span className="info-value">{getTimeElapsed(ticket.created_at)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="ticket-description">
            <h3>Description</h3>
            <p>{ticket.description}</p>
          </div>

          {ticket.medical_case_id && (
            <div className="medical-info">
              <h3>Medical Case Information</h3>
              <p><strong>Case ID:</strong> {ticket.medical_case_id}</p>
            </div>
          )}
        </div>

        <div className="ticket-sidebar">
          <div className="ticket-actions">
            {ticket.status !== 'closed' && (
              <button 
                onClick={() => setShowResolutionForm(true)}
                className="btn-success"
              >
                <FaCheckCircle />
                Close Ticket
              </button>
            )}
            <button className="btn-secondary">
              <FaEdit />
              Edit Ticket
            </button>
          </div>

          {ticket.satisfaction_rating && (
            <div className="satisfaction-info">
              <h4>Customer Satisfaction</h4>
              <div className="rating-display">
                <span className="rating-stars">
                  {'★'.repeat(ticket.satisfaction_rating)}{'☆'.repeat(5 - ticket.satisfaction_rating)}
                </span>
                <p>{ticket.satisfaction_feedback}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Comments Section */}
      <div className="comments-section">
        <h3>Comments & Updates</h3>
        
        <div className="comments-list">
          {comments.length === 0 ? (
            <p>No comments yet. Be the first to comment.</p>
          ) : (
            comments.map(comment => (
              <div key={comment.id} className={`comment ${comment.is_internal ? 'internal' : ''}`}>
                <div className="comment-header">
                  <div className="comment-author">
                    <FaUser className="author-icon" />
                    <span>{comment.user_name}</span>
                    {comment.is_internal && (
                      <span className="internal-badge">Internal</span>
                    )}
                  </div>
                  <span className="comment-time">
                    {new Date(comment.created_at).toLocaleString()}
                  </span>
                </div>
                <div className="comment-body">
                  <p>{comment.comment}</p>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Add Comment Form */}
        {ticket.status !== 'closed' && (
          <form onSubmit={handleAddComment} className="comment-form">
            <textarea
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              placeholder="Add a comment or update..."
              rows={3}
              required
            />
            <button type="submit" className="btn-primary">
              <FaComment />
              Add Comment
            </button>
          </form>
        )}
      </div>

      {/* Resolution Form Modal */}
      {showResolutionForm && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Close Ticket</h3>
            <p>Please rate your experience and provide feedback:</p>
            
            <div className="form-group">
              <label>Satisfaction Rating</label>
              <div className="rating-input">
                {[1, 2, 3, 4, 5].map(rating => (
                  <button
                    key={rating}
                    type="button"
                    onClick={() => setSatisfactionRating(rating)}
                    className={rating <= satisfactionRating ? 'star-filled' : 'star-empty'}
                  >
                    ★
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Feedback (Optional)</label>
              <textarea
                value={satisfactionFeedback}
                onChange={(e) => setSatisfactionFeedback(e.target.value)}
                placeholder="How can we improve our service?"
                rows={3}
              />
            </div>

            <div className="modal-actions">
              <button 
                onClick={() => setShowResolutionForm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button 
                onClick={handleCloseTicket}
                className="btn-success"
              >
                Close Ticket
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TicketDetail;