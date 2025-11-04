import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { 
  Activity, 
  BarChart3,
  ClipboardList,
  LogOut,
  Bell,
  Settings,
  Menu,
  X
} from 'lucide-react';
import QueueView from '@/components/nurse/QueueView';
import PARDetailView from '@/components/nurse/PARDetailView';
import AnalyticsDashboard from '@/components/nurse/AnalyticsDashboard';
import { webSocketService } from '@/services/websocket';

const NurseDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState<number>(0);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    connectWebSocket();
    
    return () => {
      webSocketService.disconnect();
    };
  }, []);

  const connectWebSocket = async () => {
    try {
      await webSocketService.connect(undefined, true); // Nurse mode
      setWsConnected(true);

      // Listen for new PAR notifications
      webSocketService.on('new_par', (message) => {
        setNotifications(prev => prev + 1);
        // Show toast notification
        console.log('New PAR received:', message);
      });

      // Listen for queue updates
      webSocketService.on('queue_update', (message) => {
        console.log('Queue updated:', message);
      });
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
    }
  };

  const navigationItems = [
    { path: '/nurse/queue', icon: ClipboardList, label: 'Queue', badge: notifications > 0 ? notifications : undefined },
    { path: '/nurse/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/nurse/settings', icon: Settings, label: 'Settings' }
  ];

  const isActive = (path: string) => {
    return location.pathname.startsWith(path);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    setMobileMenuOpen(false);
    if (path === '/nurse/queue') {
      setNotifications(0); // Clear notifications when viewing queue
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden"
              >
                {mobileMenuOpen ? (
                  <X className="w-6 h-6 text-gray-600" />
                ) : (
                  <Menu className="w-6 h-6 text-gray-600" />
                )}
              </button>
              
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                  <Activity className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">
                    Nurse Dashboard
                  </h1>
                  <p className="text-xs text-gray-600 hidden sm:block">
                    Patient Assessment Review System
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* WebSocket Status */}
              <div className="hidden sm:flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-xs text-gray-600">
                  {wsConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* Notifications */}
              <button className="relative p-2 hover:bg-gray-100 rounded-lg">
                <Bell className="w-5 h-5 text-gray-600" />
                {notifications > 0 && (
                  <span className="absolute top-0 right-0 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                    {notifications > 9 ? '9+' : notifications}
                  </span>
                )}
              </button>

              {/* User Menu */}
              <div className="flex items-center space-x-3">
                <div className="text-right hidden sm:block">
                  <p className="text-sm font-medium text-gray-900">
                    {user?.first_name || user?.email}
                  </p>
                  <p className="text-xs text-gray-500 capitalize">{user?.role}</p>
                </div>
                <Button variant="outline" size="sm" onClick={logout}>
                  <LogOut className="w-4 h-4 sm:mr-2" />
                  <span className="hidden sm:inline">Logout</span>
                </Button>
              </div>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex space-x-1 pb-4">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              
              return (
                <button
                  key={item.path}
                  onClick={() => handleNavigation(item.path)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors relative ${
                    active
                      ? 'bg-blue-100 text-blue-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-white border-b shadow-lg">
          <div className="px-4 py-3 space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              
              return (
                <button
                  key={item.path}
                  onClick={() => handleNavigation(item.path)}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg w-full transition-colors ${
                    active
                      ? 'bg-blue-100 text-blue-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="ml-auto w-6 h-6 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<QueueView />} />
          <Route path="/queue" element={<QueueView />} />
          <Route path="/par/:parId" element={<PARDetailView />} />
          <Route path="/analytics" element={<AnalyticsDashboard />} />
          <Route path="/settings" element={<SettingsView />} />
        </Routes>
      </div>
    </div>
  );
};

// Settings View Component
const SettingsView: React.FC = () => {
  const { user } = useAuth();
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-600">Manage your preferences and account settings</p>
      </div>

      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Information</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <p className="text-gray-900">{user?.email}</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <p className="text-gray-900">
              {user?.first_name && user?.last_name
                ? `${user.first_name} ${user.last_name}`
                : 'Not set'}
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
            <p className="text-gray-900 capitalize">{user?.role}</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
            <p className="text-gray-900">{user?.is_active ? 'Active' : 'Inactive'}</p>
          </div>
        </div>
      </Card>

      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Notification Preferences</h3>
        <div className="space-y-3">
          <label className="flex items-center space-x-3">
            <input type="checkbox" defaultChecked className="rounded" />
            <span className="text-gray-700">New PAR notifications</span>
          </label>
          <label className="flex items-center space-x-3">
            <input type="checkbox" defaultChecked className="rounded" />
            <span className="text-gray-700">Urgent case alerts</span>
          </label>
          <label className="flex items-center space-x-3">
            <input type="checkbox" defaultChecked className="rounded" />
            <span className="text-gray-700">Red flag warnings</span>
          </label>
          <label className="flex items-center space-x-3">
            <input type="checkbox" className="rounded" />
            <span className="text-gray-700">Daily summary emails</span>
          </label>
        </div>
      </Card>

      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Display Preferences</h3>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Queue Refresh Interval
            </label>
            <select className="w-full px-3 py-2 border rounded-lg">
              <option value="30">30 seconds</option>
              <option value="60">1 minute</option>
              <option value="120">2 minutes</option>
              <option value="300">5 minutes</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Items per Page
            </label>
            <select className="w-full px-3 py-2 border rounded-lg">
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>
        </div>
      </Card>

      <div className="flex justify-end">
        <Button>Save Preferences</Button>
      </div>
    </div>
  );
};

export default NurseDashboard;
