import React, { useState, useEffect } from 'react';
import { FaUser, FaSave, FaShieldAlt, FaBell, FaEnvelope, FaPhone, FaMapMarkerAlt } from 'react-icons/fa';

const Profile = () => {
  const [user, setUser] = useState(null);
  const [organization, setOrganization] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [activeTab, setActiveTab] = useState('profile');
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    department: '',
    license_number: ''
  });

  useEffect(() => {
    fetchUserProfile();
  }, []);

  const fetchUserProfile = async () => {
    try {
      const response = await fetch('/api/auth/profile', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        setOrganization(data.organization);
        setFormData({
          full_name: data.user.full_name || '',
          email: data.user.email || '',
          phone: data.user.phone || '',
          department: data.user.department || '',
          license_number: data.user.license_number || ''
        });
      }
    } catch (error) {
      console.error('Failed to fetch profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/auth/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        setEditMode(false);
        fetchUserProfile();
      }
    } catch (error) {
      console.error('Failed to update profile:', error);
    }
  };

  const tabs = [
    { id: 'profile', label: 'Profile', icon: FaUser },
    { id: 'security', label: 'Security', icon: FaShieldAlt },
    { id: 'notifications', label: 'Notifications', icon: FaBell }
  ];

  if (loading) {
    return <div className="loading">Loading profile...</div>;
  }

  if (!user) {
    return <div className="error">Failed to load profile</div>;
  }

  return (
    <div className="profile-container">
      <div className="page-header">
        <h1>User Profile</h1>
        {!editMode ? (
          <button onClick={() => setEditMode(true)} className="btn-primary">
            Edit Profile
          </button>
        ) : (
          <div className="edit-actions">
            <button onClick={() => setEditMode(false)} className="btn-secondary">
              Cancel
            </button>
            <button onClick={handleSaveProfile} className="btn-success">
              <FaSave />
              Save Changes
            </button>
          </div>
        )}
      </div>

      <div className="profile-content">
        <div className="profile-sidebar">
          <div className="profile-avatar">
            <FaUser className="avatar-icon" />
          </div>
          <h2>{user.full_name}</h2>
          <p className="user-role">{user.role}</p>
          <p className="user-department">{user.department}</p>
          
          <div className="organization-info">
            <h3>Organization</h3>
            <p>{organization?.name}</p>
            <p className="organization-type">{organization?.type}</p>
            <span className={`subscription-tier ${organization?.subscription_tier}`}>
              {organization?.subscription_tier}
            </span>
          </div>
        </div>

        <div className="profile-main">
          <div className="profile-tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              >
                <tab.icon />
                {tab.label}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {activeTab === 'profile' && (
              <div className="profile-form">
                <h3>Personal Information</h3>
                <form onSubmit={handleSaveProfile}>
                  <div className="form-row">
                    <div className="form-group">
                      <label>Full Name</label>
                      <input
                        type="text"
                        name="full_name"
                        value={formData.full_name}
                        onChange={handleInputChange}
                        disabled={!editMode}
                      />
                    </div>
                    <div className="form-group">
                      <label>Email</label>
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleInputChange}
                        disabled={!editMode}
                      />
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label>Phone</label>
                      <input
                        type="tel"
                        name="phone"
                        value={formData.phone}
                        onChange={handleInputChange}
                        disabled={!editMode}
                      />
                    </div>
                    <div className="form-group">
                      <label>Department</label>
                      <input
                        type="text"
                        name="department"
                        value={formData.department}
                        onChange={handleInputChange}
                        disabled={!editMode}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label>License Number</label>
                    <input
                      type="text"
                      name="license_number"
                      value={formData.license_number}
                      onChange={handleInputChange}
                      disabled={!editMode}
                    />
                  </div>
                </form>

                <div className="profile-stats">
                  <h3>Account Statistics</h3>
                  <div className="stats-grid">
                    <div className="stat-item">
                      <span className="stat-label">Member Since</span>
                      <span className="stat-value">
                        {new Date(user.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Last Login</span>
                      <span className="stat-value">
                        {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Tickets Created</span>
                      <span className="stat-value">0</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Courses Completed</span>
                      <span className="stat-value">0</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="security-section">
                <h3>Security Settings</h3>
                
                <div className="security-card">
                  <h4>Password</h4>
                  <p>Change your password to keep your account secure</p>
                  <button className="btn-secondary">Change Password</button>
                </div>

                <div className="security-card">
                  <h4>Two-Factor Authentication</h4>
                  <p>Add an extra layer of security to your account</p>
                  <button className="btn-secondary">Enable 2FA</button>
                </div>

                <div className="security-card">
                  <h4>Login Sessions</h4>
                  <p>Manage your active login sessions</p>
                  <button className="btn-secondary">View Sessions</button>
                </div>

                <div className="security-card">
                  <h4>Data Export</h4>
                  <p>Download a copy of your personal data</p>
                  <button className="btn-secondary">Request Export</button>
                </div>
              </div>
            )}

            {activeTab === 'notifications' && (
              <div className="notifications-section">
                <h3>Notification Preferences</h3>
                
                <div className="notification-groups">
                  <div className="notification-group">
                    <h4>Email Notifications</h4>
                    <div className="notification-toggles">
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>Ticket updates</span>
                      </label>
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>System maintenance</span>
                      </label>
                      <label className="toggle-item">
                        <input type="checkbox" />
                        <span>Product updates</span>
                      </label>
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>Security alerts</span>
                      </label>
                    </div>
                  </div>

                  <div className="notification-group">
                    <h4>Medical Emergency Alerts</h4>
                    <div className="notification-toggles">
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>Emergency notifications</span>
                      </label>
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>SLA breach alerts</span>
                      </label>
                    </div>
                  </div>

                  <div className="notification-group">
                    <h4>Training & Certification</h4>
                    <div className="notification-toggles">
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>Course completions</span>
                      </label>
                      <label className="toggle-item">
                        <input type="checkbox" defaultChecked />
                        <span>Certificate renewals</span>
                      </label>
                      <label className="toggle-item">
                        <input type="checkbox" />
                        <span>New course available</span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;