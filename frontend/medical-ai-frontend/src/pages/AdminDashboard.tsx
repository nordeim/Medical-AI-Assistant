import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAuth } from '../contexts/AuthContext';
import { 
  Settings, 
  Users, 
  Activity, 
  Shield, 
  Database, 
  AlertCircle,
  LogOut,
  TrendingUp,
  Clock,
  CheckCircle
} from 'lucide-react';

const AdminDashboard: React.FC = () => {
  const { user, logout } = useAuth();

  // Mock data for demonstration
  const systemStats = {
    totalUsers: 1247,
    activeSessions: 23,
    totalSessions: 5692,
    aiAccuracy: 94.2,
    avgResponseTime: '2.3s',
    systemUptime: '99.8%',
  };

  const recentActivity = [
    {
      id: 1,
      type: 'user_registration',
      user: 'john.doe@email.com',
      timestamp: '2024-01-20T12:30:00Z',
      status: 'success',
    },
    {
      id: 2,
      type: 'ai_response',
      session: 'sess_123',
      timestamp: '2024-01-20T12:29:00Z',
      status: 'success',
    },
    {
      id: 3,
      type: 'system_backup',
      timestamp: '2024-01-20T12:00:00Z',
      status: 'success',
    },
    {
      id: 4,
      type: 'security_alert',
      user: 'unknown@ip',
      timestamp: '2024-01-20T11:45:00Z',
      status: 'warning',
    },
  ];

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'user_registration':
        return <Users className="w-4 h-4" />;
      case 'ai_response':
        return <Activity className="w-4 h-4" />;
      case 'system_backup':
        return <Database className="w-4 h-4" />;
      case 'security_alert':
        return <Shield className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'bg-green-100 text-green-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
                <Settings className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Admin Dashboard
                </h1>
                <p className="text-gray-600">
                  System Management & Analytics
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">
                  {user?.first_name || user?.email}
                </p>
                <p className="text-xs text-gray-500">Administrator</p>
              </div>
              <Button variant="outline" onClick={logout}>
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Users</p>
                <p className="text-2xl font-bold text-gray-900">{systemStats.totalUsers}</p>
              </div>
              <Users className="w-8 h-8 text-blue-600" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Sessions</p>
                <p className="text-2xl font-bold text-green-600">{systemStats.activeSessions}</p>
              </div>
              <Activity className="w-8 h-8 text-green-600" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Sessions</p>
                <p className="text-2xl font-bold text-blue-600">{systemStats.totalSessions}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-600" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">AI Accuracy</p>
                <p className="text-2xl font-bold text-purple-600">{systemStats.aiAccuracy}%</p>
              </div>
              <CheckCircle className="w-8 h-8 text-purple-600" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Response</p>
                <p className="text-2xl font-bold text-orange-600">{systemStats.avgResponseTime}</p>
              </div>
              <Clock className="w-8 h-8 text-orange-600" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Uptime</p>
                <p className="text-2xl font-bold text-green-600">{systemStats.systemUptime}</p>
              </div>
              <Activity className="w-8 h-8 text-green-600" />
            </div>
          </Card>
        </div>

        {/* System Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <Card>
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-900">
                System Controls
              </h2>
            </div>
            <div className="p-6 space-y-4">
              <Button className="w-full justify-start" variant="outline">
                <Database className="w-4 h-4 mr-2" />
                Database Backup
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Shield className="w-4 h-4 mr-2" />
                Security Scan
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Activity className="w-4 h-4 mr-2" />
                System Health Check
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Settings className="w-4 h-4 mr-2" />
                AI Model Configuration
              </Button>
            </div>
          </Card>

          <Card>
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-900">
                Recent Activity
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                      {getActivityIcon(activity.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900">
                        {activity.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        {activity.user && (
                          <span className="text-gray-500"> - {activity.user}</span>
                        )}
                        {activity.session && (
                          <span className="text-gray-500"> - {activity.session}</span>
                        )}
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatTime(activity.timestamp)}
                      </p>
                    </div>
                    <Badge className={getStatusColor(activity.status)}>
                      {activity.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>

        {/* System Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-900">
                System Status
              </h2>
            </div>
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">API Services</span>
                <Badge className="bg-green-100 text-green-800">Online</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Database</span>
                <Badge className="bg-green-100 text-green-800">Connected</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">AI Models</span>
                <Badge className="bg-green-100 text-green-800">Running</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">WebSocket</span>
                <Badge className="bg-green-100 text-green-800">Connected</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">File Storage</span>
                <Badge className="bg-green-100 text-green-800">Available</Badge>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-900">
                Quick Actions
              </h2>
            </div>
            <div className="p-6 space-y-4">
              <Button className="w-full justify-start" variant="outline">
                <Users className="w-4 h-4 mr-2" />
                Manage Users
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Activity className="w-4 h-4 mr-2" />
                View Analytics
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Shield className="w-4 h-4 mr-2" />
                Audit Logs
              </Button>
              <Button className="w-full justify-start" variant="outline">
                <Settings className="w-4 h-4 mr-2" />
                System Settings
              </Button>
            </div>
          </Card>
        </div>

        {/* Status Message */}
        <div className="mt-8 bg-purple-50 border border-purple-200 rounded-lg p-6">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-purple-600 mt-0.5" />
            <div>
              <h3 className="font-medium text-purple-900 mb-1">
                Administrative Interface
              </h3>
              <p className="text-sm text-purple-700">
                This admin dashboard is currently in development with mock data. 
                Production features will include user management, system configuration, 
                audit trails, and comprehensive analytics for the Medical AI Assistant system.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;