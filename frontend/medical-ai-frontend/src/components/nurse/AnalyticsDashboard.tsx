// @ts-nocheck - recharts type compatibility issues with React 18
import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  TrendingUp,
  Clock,
  CheckCircle,
  AlertTriangle,
  Activity,
  Calendar,
  Users
} from 'lucide-react';
import { AnalyticsMetrics } from '@/types';
import { apiService } from '@/services/api';

const AnalyticsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'today' | 'week' | 'month'>('today');

  useEffect(() => {
    loadAnalytics();
  }, [timeRange]);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      
      // Fetch real analytics data from backend
      const params: { start_date?: string; end_date?: string } = {};
      
      if (timeRange === 'today') {
        params.start_date = new Date().toISOString().split('T')[0];
        params.end_date = new Date().toISOString().split('T')[0];
      } else if (timeRange === 'week') {
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        params.start_date = weekAgo.toISOString().split('T')[0];
      } else if (timeRange === 'month') {
        const monthAgo = new Date();
        monthAgo.setMonth(monthAgo.getMonth() - 1);
        params.start_date = monthAgo.toISOString().split('T')[0];
      }
      
      const response = await apiService.getNurseAnalytics(params);
      
      if (response.success && response.data) {
        setMetrics(response.data as AnalyticsMetrics);
      }
    } catch (err) {
      console.error('Failed to load analytics:', err);
      // Show error to user
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">{loading ? 'Loading analytics...' : 'Failed to load analytics data'}</p>
          {!loading && !metrics && (
            <Button onClick={loadAnalytics} className="mt-4">
              Retry
            </Button>
          )}
        </div>
      </div>
    );
  }

  const urgencyData = Object.entries(metrics.pars_by_urgency).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value
  }));

  const COLORS = ['#10b981', '#f59e0b', '#ef4444'];

  const reviewData = [
    { name: 'Approved', value: metrics.approval_rate * 100 },
    { name: 'Override', value: metrics.override_rate * 100 }
  ];

  const REVIEW_COLORS = ['#10b981', '#ef4444'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="text-sm text-gray-600">System performance and metrics</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={timeRange === 'today' ? 'default' : 'outline'}
            onClick={() => setTimeRange('today')}
            size="sm"
          >
            Today
          </Button>
          <Button
            variant={timeRange === 'week' ? 'default' : 'outline'}
            onClick={() => setTimeRange('week')}
            size="sm"
          >
            This Week
          </Button>
          <Button
            variant={timeRange === 'month' ? 'default' : 'outline'}
            onClick={() => setTimeRange('month')}
            size="sm"
          >
            This Month
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <Calendar className="w-8 h-8 text-blue-600" />
            <TrendingUp className="w-5 h-5 text-green-600" />
          </div>
          <p className="text-sm text-gray-600">Total PARs Today</p>
          <p className="text-3xl font-bold text-gray-900">{metrics.total_pars_today}</p>
          <p className="text-xs text-gray-500 mt-1">{metrics.total_pars_week} this week</p>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <Clock className="w-8 h-8 text-orange-600" />
          </div>
          <p className="text-sm text-gray-600">Avg Wait Time</p>
          <p className="text-3xl font-bold text-gray-900">{metrics.avg_wait_time_minutes}m</p>
          <p className="text-xs text-gray-500 mt-1">
            {metrics.avg_review_time_minutes}m avg review time
          </p>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <p className="text-sm text-gray-600">Approval Rate</p>
          <p className="text-3xl font-bold text-gray-900">
            {(metrics.approval_rate * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {(metrics.override_rate * 100).toFixed(0)}% override rate
          </p>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
          <p className="text-sm text-gray-600">High Risk Cases</p>
          <p className="text-3xl font-bold text-gray-900">
            {(metrics.high_risk_rate * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {(metrics.red_flag_rate * 100).toFixed(0)}% red flags
          </p>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* PARs by Hour */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">PARs by Hour</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics.pars_by_hour}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="hour" 
                label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="count" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="PARs"
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* PARs by Urgency */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">PARs by Urgency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={urgencyData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {urgencyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Complaints */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Chief Complaints</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metrics.top_complaints} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="complaint" type="category" width={150} />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#3b82f6" name="Count" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Review Decisions */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Review Decisions</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={reviewData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value.toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {reviewData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={REVIEW_COLORS[index % REVIEW_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Additional Insights */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2" />
          Performance Insights
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Average Wait Time</p>
            <p className="text-2xl font-bold text-blue-600">{metrics.avg_wait_time_minutes} min</p>
            <p className="text-xs text-gray-500 mt-1">Target: &lt; 15 min</p>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Review Efficiency</p>
            <p className="text-2xl font-bold text-green-600">{metrics.avg_review_time_minutes} min</p>
            <p className="text-xs text-gray-500 mt-1">Per assessment</p>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Daily Throughput</p>
            <p className="text-2xl font-bold text-purple-600">{metrics.total_pars_today}</p>
            <p className="text-xs text-gray-500 mt-1">Assessments today</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AnalyticsDashboard;
