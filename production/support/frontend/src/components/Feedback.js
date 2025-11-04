import React, { useState, useEffect } from 'react';
import { FaStar, FaPaperPlane, FaSmile, FaFrown, FaMeh } from 'react-icons/fa';

const Feedback = () => {
  const [feedbackList, setFeedbackList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    feedback_type: 'general_feedback',
    rating: 5,
    title: '',
    description: '',
    feature_area: '',
    is_anonymous: false
  });

  useEffect(() => {
    fetchFeedback();
  }, []);

  const fetchFeedback = async () => {
    try {
      const response = await fetch('/api/feedback', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setFeedbackList(data);
      }
    } catch (error) {
      console.error('Failed to fetch feedback:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitFeedback = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        setShowForm(false);
        setFormData({
          feedback_type: 'general_feedback',
          rating: 5,
          title: '',
          description: '',
          feature_area: '',
          is_anonymous: false
        });
        fetchFeedback();
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  const getSentimentIcon = (sentimentScore) => {
    if (sentimentScore > 0.3) return <FaSmile className="sentiment-positive" />;
    if (sentimentScore < -0.3) return <FaFrown className="sentiment-negative" />;
    return <FaMeh className="sentiment-neutral" />;
  };

  const getSentimentColor = (sentimentScore) => {
    if (sentimentScore > 0.3) return '#27ae60';
    if (sentimentScore < -0.3) return '#e74c3c';
    return '#f39c12';
  };

  const feedbackTypes = [
    { value: 'general_feedback', label: 'General Feedback' },
    { value: 'feature_request', label: 'Feature Request' },
    { value: 'bug_report', label: 'Bug Report' },
    { value: 'suggestion', label: 'Suggestion' },
    { value: 'complaint', label: 'Complaint' }
  ];

  const featureAreas = [
    { value: '', label: 'Select feature area' },
    { value: 'ui', label: 'User Interface' },
    { value: 'api', label: 'API & Integration' },
    { value: 'mobile_app', label: 'Mobile Application' },
    { value: 'web_dashboard', label: 'Web Dashboard' },
    { value: 'medical_features', label: 'Medical Features' },
    { value: 'reporting', label: 'Reporting & Analytics' },
    { value: 'support', label: 'Customer Support' }
  ];

  return (
    <div className="feedback-container">
      <div className="page-header">
        <h1>Feedback & Suggestions</h1>
        <button 
          onClick={() => setShowForm(true)}
          className="btn-primary"
        >
          <FaPaperPlane />
          Submit Feedback
        </button>
      </div>

      {/* Feedback Form Modal */}
      {showForm && (
        <div className="modal-overlay">
          <div className="modal feedback-modal">
            <h3>Submit Feedback</h3>
            
            <form onSubmit={handleSubmitFeedback}>
              <div className="form-group">
                <label>Feedback Type</label>
                <select
                  value={formData.feedback_type}
                  onChange={(e) => setFormData({...formData, feedback_type: e.target.value})}
                >
                  {feedbackTypes.map(type => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Rating</label>
                <div className="rating-input">
                  {[1, 2, 3, 4, 5].map(rating => (
                    <button
                      key={rating}
                      type="button"
                      onClick={() => setFormData({...formData, rating})}
                      className={rating <= formData.rating ? 'star-filled' : 'star-empty'}
                    >
                      <FaStar />
                    </button>
                  ))}
                  <span className="rating-label">{formData.rating}/5</span>
                </div>
              </div>

              <div className="form-group">
                <label>Feature Area</label>
                <select
                  value={formData.feature_area}
                  onChange={(e) => setFormData({...formData, feature_area: e.target.value})}
                >
                  {featureAreas.map(area => (
                    <option key={area.value} value={area.value}>
                      {area.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  value={formData.title}
                  onChange={(e) => setFormData({...formData, title: e.target.value})}
                  placeholder="Brief summary of your feedback"
                  required
                />
              </div>

              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({...formData, description: e.target.value})}
                  placeholder="Detailed feedback, suggestions, or bug reports"
                  rows={4}
                  required
                />
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={formData.is_anonymous}
                    onChange={(e) => setFormData({...formData, is_anonymous: e.target.checked})}
                  />
                  Submit anonymously
                </label>
              </div>

              <div className="modal-actions">
                <button 
                  type="button"
                  onClick={() => setShowForm(false)}
                  className="btn-secondary"
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Submit Feedback
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Feedback List */}
      <div className="feedback-list">
        {loading ? (
          <div className="loading">Loading feedback...</div>
        ) : feedbackList.length === 0 ? (
          <div className="no-feedback">
            <p>No feedback submitted yet. Be the first to share your thoughts!</p>
          </div>
        ) : (
          feedbackList.map(feedback => (
            <div key={feedback.id} className="feedback-item">
              <div className="feedback-header">
                <div className="feedback-meta">
                  <span className="feedback-type">{feedback.feedback_type.replace('_', ' ')}</span>
                  <div className="rating-display">
                    {'★'.repeat(feedback.rating)}{'☆'.repeat(5 - feedback.rating)}
                  </div>
                  {feedback.sentiment_score !== null && (
                    <div className="sentiment-indicator">
                      {getSentimentIcon(feedback.sentiment_score)}
                      <span style={{ color: getSentimentColor(feedback.sentiment_score) }}>
                        {feedback.sentiment_score > 0 ? 'Positive' : 
                         feedback.sentiment_score < 0 ? 'Negative' : 'Neutral'}
                      </span>
                    </div>
                  )}
                </div>
                <span className="feedback-date">
                  {new Date(feedback.created_at).toLocaleDateString()}
                </span>
              </div>
              
              <div className="feedback-content">
                <h4>{feedback.title}</h4>
                <p>{feedback.description}</p>
              </div>
              
              <div className="feedback-footer">
                <div className="feedback-tags">
                  {feedback.feature_area && (
                    <span className="feature-tag">{feedback.feature_area}</span>
                  )}
                  <span className={`status-badge ${feedback.status}`}>
                    {feedback.status.replace('_', ' ')}
                  </span>
                </div>
                {!feedback.is_anonymous && (
                  <span className="author">
                    {feedback.user_name}
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Analytics Summary */}
      {feedbackList.length > 0 && (
        <div className="feedback-analytics">
          <h3>Feedback Analytics</h3>
          <div className="analytics-grid">
            <div className="analytics-card">
              <h4>Total Feedback</h4>
              <span className="analytics-number">{feedbackList.length}</span>
            </div>
            <div className="analytics-card">
              <h4>Average Rating</h4>
              <span className="analytics-number">
                {(feedbackList.reduce((sum, f) => sum + f.rating, 0) / feedbackList.length).toFixed(1)}
              </span>
            </div>
            <div className="analytics-card">
              <h4>Positive Sentiment</h4>
              <span className="analytics-number">
                {Math.round((feedbackList.filter(f => f.sentiment_score > 0.3).length / feedbackList.length) * 100)}%
              </span>
            </div>
            <div className="analytics-card">
              <h4>Feature Requests</h4>
              <span className="analytics-number">
                {feedbackList.filter(f => f.feedback_type === 'feature_request').length}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Feedback;