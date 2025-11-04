import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import { 
  FaTicketAlt, 
  FaHeartbeat, 
  FaComments, 
  FaBook, 
  FaGraduationCap, 
  FaChartBar,
  FaBell,
  FaUser,
  FaCog
} from 'react-icons/fa';
import Dashboard from './components/Dashboard';
import TicketsList from './components/TicketsList';
import TicketDetail from './components/TicketDetail';
import CreateTicket from './components/CreateTicket';
import Feedback from './components/Feedback';
import HealthMonitor from './components/HealthMonitor';
import KnowledgeBase from './components/KnowledgeBase';
import Training from './components/Training';
import Metrics from './components/Metrics';
import Profile from './components/Profile';
import Notifications from './components/Notifications';
import './App.css';

const App = () => {
  const [user, setUser] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    // Check authentication status
    const token = localStorage.getItem('token');
    if (token) {
      // Verify token and get user info
      fetchUserProfile();
    }
    
    // Set up real-time notifications
    setupNotifications();
  }, []);

  const fetchUserProfile = async () => {
    try {
      const response = await fetch('/api/auth/profile', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      }
    } catch (error) {
      console.error('Failed to fetch user profile:', error);
    }
  };

  const setupNotifications = () => {
    // Real-time notification setup
    // In production, this would use WebSockets
    setInterval(() => {
      checkForNotifications();
    }, 30000); // Check every 30 seconds
  };

  const checkForNotifications = async () => {
    try {
      const response = await fetch('/api/notifications/unread', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setNotifications(data);
        setUnreadCount(data.filter(n => !n.read).length);
      }
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
    window.location.href = '/login';
  };

  if (!user) {
    return <LoginForm onLogin={setUser} />;
  }

  return (
    <Router>
      <div className="app">
        <Header 
          user={user} 
          notifications={notifications} 
          unreadCount={unreadCount}
          onLogout={handleLogout}
        />
        
        <div className="main-container">
          <Sidebar user={user} />
          
          <main className="content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/tickets" element={<TicketsList />} />
              <Route path="/tickets/new" element={<CreateTicket />} />
              <Route path="/tickets/:id" element={<TicketDetail />} />
              <Route path="/feedback" element={<Feedback />} />
              <Route path="/health" element={<HealthMonitor />} />
              <Route path="/knowledge" element={<KnowledgeBase />} />
              <Route path="/training" element={<Training />} />
              <Route path="/metrics" element={<Metrics />} />
              <Route path="/profile" element={<Profile />} />
              <Route path="/notifications" element={<Notifications />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
};

const Header = ({ user, notifications, unreadCount, onLogout }) => {
  const [showNotifications, setShowNotifications] = useState(false);

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <FaHeartbeat className="logo-icon" />
          <span>Healthcare Support</span>
        </div>
        
        <div className="header-actions">
          <button 
            className="notification-btn"
            onClick={() => setShowNotifications(!showNotifications)}
          >
            <FaBell />
            {unreadCount > 0 && (
              <span className="notification-badge">{unreadCount}</span>
            )}
          </button>
          
          <div className="user-menu">
            <FaUser className="user-icon" />
            <span>{user.full_name}</span>
            <button onClick={onLogout} className="logout-btn">Logout</button>
          </div>
        </div>
      </div>
      
      {showNotifications && (
        <div className="notifications-dropdown">
          <h4>Notifications</h4>
          {notifications.length === 0 ? (
            <p>No new notifications</p>
          ) : (
            notifications.map(notification => (
              <div key={notification.id} className="notification-item">
                <p>{notification.message}</p>
                <small>{new Date(notification.created_at).toLocaleString()}</small>
              </div>
            ))
          )}
        </div>
      )}
    </header>
  );
};

const Sidebar = ({ user }) => {
  const menuItems = [
    { path: '/', icon: FaChartBar, label: 'Dashboard' },
    { path: '/tickets', icon: FaTicketAlt, label: 'Support Tickets' },
    { path: '/feedback', icon: FaComments, label: 'Feedback' },
    { path: '/health', icon: FaHeartbeat, label: 'Health Monitor' },
    { path: '/knowledge', icon: FaBook, label: 'Knowledge Base' },
    { path: '/training', icon: FaGraduationCap, label: 'Training' },
    { path: '/metrics', icon: FaChartBar, label: 'Success Metrics' },
    { path: '/profile', icon: FaCog, label: 'Profile' }
  ];

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        {menuItems.map(item => (
          <Link 
            key={item.path} 
            to={item.path} 
            className="nav-item"
            activeclassname="active"
          >
            <item.icon className="nav-icon" />
            <span className="nav-label">{item.label}</span>
          </Link>
        ))}
      </nav>
    </aside>
  );
};

const LoginForm = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      });
      
      if (response.ok) {
        const { token, user } = await response.json();
        localStorage.setItem('token', token);
        onLogin(user);
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Login failed');
      }
    } catch (error) {
      setError('Network error');
    }
  };

  return (
    <div className="login-container">
      <div className="login-form">
        <div className="logo">
          <FaHeartbeat className="logo-icon" />
          <span>Healthcare Support System</span>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={credentials.email}
              onChange={(e) => setCredentials({...credentials, email: e.target.value})}
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={credentials.password}
              onChange={(e) => setCredentials({...credentials, password: e.target.value})}
              required
            />
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <button type="submit" className="btn-primary">
            Login
          </button>
        </form>
        
        <div className="emergency-support">
          <p>Medical Emergency? Call: <strong>+1-800-MEDICAL</strong></p>
        </div>
      </div>
    </div>
  );
};

export default App;