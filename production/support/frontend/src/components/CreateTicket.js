import React, { useState } from 'react';
import { FaPlus, FaExclamationTriangle, FaFileAlt } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

const CreateTicket = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    category: '',
    priority: 'medium',
    medical_case_id: '',
    attachments: []
  });
  const [errors, setErrors] = useState({});

  const categories = [
    { value: 'technical', label: 'Technical Issue' },
    { value: 'medical', label: 'Medical/Clinical' },
    { value: 'billing', label: 'Billing & Account' },
    { value: 'training', label: 'Training & Onboarding' },
    { value: 'feature_request', label: 'Feature Request' },
    { value: 'account', label: 'Account Management' },
    { value: 'integration', label: 'API/Integration' },
    { value: 'security', label: 'Security' }
  ];

  const priorities = [
    { value: 'low', label: 'Low', description: 'General inquiry or minor issue' },
    { value: 'medium', label: 'Medium', description: 'Standard support request' },
    { value: 'high', label: 'High', description: 'Issue affecting workflow' },
    { value: 'critical', label: 'Critical', description: 'System down or major functionality broken' },
    { value: 'emergency', label: 'Emergency', description: 'Medical emergency or patient safety concern' }
  ];

  const validateForm = () => {
    const newErrors = {};

    if (!formData.title.trim()) {
      newErrors.title = 'Title is required';
    }

    if (!formData.description.trim()) {
      newErrors.description = 'Description is required';
    }

    if (!formData.category) {
      newErrors.category = 'Category is required';
    }

    if (formData.priority === 'emergency' && !formData.medical_case_id.trim()) {
      newErrors.medical_case_id = 'Medical case ID is required for emergency priority tickets';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    const validFiles = files.filter(file => {
      const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'application/pdf', 'text/plain'];
      const maxSize = 10 * 1024 * 1024; // 10MB
      
      if (!validTypes.includes(file.type)) {
        alert(`File type ${file.type} is not supported`);
        return false;
      }
      
      if (file.size > maxSize) {
        alert(`File ${file.name} is too large (max 10MB)`);
        return false;
      }
      
      return true;
    });

    setFormData(prev => ({
      ...prev,
      attachments: [...prev.attachments, ...validFiles]
    }));
  };

  const removeFile = (index) => {
    setFormData(prev => ({
      ...prev,
      attachments: prev.attachments.filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    
    try {
      const formDataToSend = new FormData();
      formDataToSend.append('title', formData.title);
      formDataToSend.append('description', formData.description);
      formDataToSend.append('category', formData.category);
      formDataToSend.append('priority', formData.priority);
      if (formData.medical_case_id) {
        formDataToSend.append('medical_case_id', formData.medical_case_id);
      }
      
      formData.attachments.forEach((file, index) => {
        formDataToSend.append(`attachment_${index}`, file);
      });

      const response = await fetch('/api/tickets', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formDataToSend
      });

      if (response.ok) {
        const newTicket = await response.json();
        navigate(`/tickets/${newTicket.id}`);
      } else {
        const errorData = await response.json();
        setErrors({ submit: errorData.error || 'Failed to create ticket' });
      }
    } catch (error) {
      setErrors({ submit: 'Network error. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="create-ticket-container">
      <div className="page-header">
        <h1>Create Support Ticket</h1>
      </div>

      {/* Emergency Alert */}
      {(formData.priority === 'emergency' || formData.category === 'medical') && (
        <div className="emergency-alert">
          <FaExclamationTriangle />
          <div>
            <h3>Medical Emergency Detected</h3>
            <p>
              This ticket has been flagged as a medical emergency. Our medical support team 
              will respond within 30 minutes. For immediate assistance, call +1-800-MEDICAL.
            </p>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="ticket-form">
        <div className="form-section">
          <h2>Ticket Information</h2>
          
          <div className="form-group">
            <label htmlFor="title">
              Title <span className="required">*</span>
            </label>
            <input
              type="text"
              id="title"
              name="title"
              value={formData.title}
              onChange={handleInputChange}
              placeholder="Brief description of the issue"
              className={errors.title ? 'error' : ''}
            />
            {errors.title && <span className="error-message">{errors.title}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="description">
              Description <span className="required">*</span>
            </label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              rows={6}
              placeholder="Detailed description of the issue, including steps to reproduce, error messages, and any relevant context"
              className={errors.description ? 'error' : ''}
            />
            {errors.description && <span className="error-message">{errors.description}</span>}
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="category">
                Category <span className="required">*</span>
              </label>
              <select
                id="category"
                name="category"
                value={formData.category}
                onChange={handleInputChange}
                className={errors.category ? 'error' : ''}
              >
                <option value="">Select a category</option>
                {categories.map(cat => (
                  <option key={cat.value} value={cat.value}>
                    {cat.label}
                  </option>
                ))}
              </select>
              {errors.category && <span className="error-message">{errors.category}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="priority">
                Priority <span className="required">*</span>
              </label>
              <select
                id="priority"
                name="priority"
                value={formData.priority}
                onChange={handleInputChange}
              >
                {priorities.map(priority => (
                  <option key={priority.value} value={priority.value}>
                    {priority.label}
                  </option>
                ))}
              </select>
              <small className="help-text">
                {priorities.find(p => p.value === formData.priority)?.description}
              </small>
            </div>
          </div>

          {(formData.category === 'medical' || formData.priority === 'emergency') && (
            <div className="form-group">
              <label htmlFor="medical_case_id">
                Medical Case ID <span className="required">*</span>
              </label>
              <input
                type="text"
                id="medical_case_id"
                name="medical_case_id"
                value={formData.medical_case_id}
                onChange={handleInputChange}
                placeholder="Associated medical case or patient ID"
                className={errors.medical_case_id ? 'error' : ''}
              />
              {errors.medical_case_id && <span className="error-message">{errors.medical_case_id}</span>}
              <small className="help-text">
                Required for medical-related or emergency priority tickets
              </small>
            </div>
          )}
        </div>

        <div className="form-section">
          <h2>Attachments</h2>
          <div className="form-group">
            <label htmlFor="attachments">
              Upload Files
            </label>
            <input
              type="file"
              id="attachments"
              multiple
              accept=".jpg,.jpeg,.png,.gif,.pdf,.txt"
              onChange={handleFileUpload}
            />
            <small className="help-text">
              Supported formats: JPG, PNG, GIF, PDF, TXT. Maximum file size: 10MB each.
            </small>
          </div>

          {formData.attachments.length > 0 && (
            <div className="attachments-list">
              <h4>Selected Files:</h4>
              {formData.attachments.map((file, index) => (
                <div key={index} className="attachment-item">
                  <FaFileAlt className="file-icon" />
                  <span>{file.name}</span>
                  <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                  <button
                    type="button"
                    onClick={() => removeFile(index)}
                    className="remove-file"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {errors.submit && (
          <div className="error-message submit-error">
            {errors.submit}
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            onClick={() => navigate('/tickets')}
            className="btn-secondary"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="btn-primary"
          >
            {loading ? 'Creating...' : 'Create Ticket'}
          </button>
        </div>
      </form>

      <div className="help-info">
        <h3>Need Help?</h3>
        <p>
          Before creating a ticket, check our{' '}
          <a href="/knowledge" target="_blank">Knowledge Base</a> for solutions 
          to common issues.
        </p>
      </div>
    </div>
  );
};

export default CreateTicket;