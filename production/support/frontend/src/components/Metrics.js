import React, { useState, useEffect } from 'react';
import { FaChartLine, FaTrendingUp, FaTrendingDown, FaUsers, FaTicketAlt, FaHeartbeat, FaClock } from 'react-icons/fa';

const Metrics = () => {
  const [metrics, setMetrics] = useState([]);
  const [filteredMetrics, setFilteredMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState('7d');
  const [selectedMetricType, setSelectedMetricType] = useState('');

  const periods = [
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' },
    { value: '90d', label: 'Last 90 Days' },
    { value: '1y', label: 'Last Year' }
  ];

  const metricTypes = [
    { value: '', label: 'All Metrics' },
    { value: 'adoption_rate', label: 'Adoption Rate' },
    { value: 'user_satisfaction', label: 'User Satisfaction' },
    { value: 'system_uptime', label: 'System Uptime' },
    { value: 'ticket_resolution_time', label: 'Ticket Resolution Time' },
    { value: 'customer_health_score', label: 'Customer Health Score' },
    { value: 'feature_usage', label: 'Feature Usage' },
    { value: 'support_ticket_volume', label: 'Support Ticket Volume' }
  ];

  useEffect(() => {
    fetchMetrics();
  }, [selectedPeriod, selectedMetricType]);

  const fetchMetrics = async () => {
    try {
      const queryParams = new URLSearchParams({
        metric_type: selectedMetricType,
        end_date: new Date().toISOString().split('T')[0],
        start_date: getStartDate(selectedPeriod)
      });

      const response = await fetch(`/api/metrics?${queryParams}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
        processMetricsData(data);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStartDate = (period) => {
    const endDate = new Date();
    const startDate = new Date();
    
    switch (period) {
      case '7d':
        startDate.setDate(endDate.getDate() - 7);
        break;
      case '30d':
        startDate.setDate(endDate.getDate() - 30);
        break;
      case '90d':
        startDate.setDate(endDate.getDate() - 90);
        break;
      case '1y':
        startDate.setFullYear(endDate.getFullYear() - 1);
        break;
      default:
        startDate.setDate(endDate.getDate() - 7);
    }
    
    return startDate.toISOString().split('T')[0];
  };

  const processMetricsData = (data) => {
    // Group metrics by type and calculate trends
    const grouped = data.reduce((acc, metric) => {
      if (!acc[metric.metric_type]) {
        acc[metric.metric_type] = [];
      }
      acc[metric.metric_type].push(metric);
      return acc;
    }, {});

    setFilteredMetrics(grouped);
  };

  const getMetricIcon = (metricType) => {
    switch (metricType) {
      case 'user_satisfaction': return FaUsers;
      case 'system_uptime': return FaHeartbeat;
      case 'ticket_resolution_time': return FaTicketAlt;
      case 'adoption_rate': return FaTrendingUp;
      default: return FaChartLine;
    }
  };

  const getMetricUnit = (metricType) => {
    switch (metricType) {
      case 'user_satisfaction':
      case 'adoption_rate':
      case 'system_uptime':
        return '%';
      case 'ticket_resolution_time':
        return 'hours';
      case 'customer_health_score':
        return '/100';
      default:
        return '';
    }
  };

  const calculateTrend = (metricValues) => {
    if (metricValues.length < 2) return 0;
    
    const sorted = metricValues.sort((a, b) => new Date(a.measurement_date) - new Date(b.measurement_date));
    const first = sorted[0].metric_value;
    const last = sorted[sorted.length - 1].metric_value;
    
    return ((last - first) / first) * 100;
  };

  const getTrendIcon = (trend) => {
    return trend > 0 ? <FaTrendingUp className="trend-up" /> : <FaTrendingDown className="trend-down" />;
  };

  const getTrendColor = (trend) => {
    return trend > 0 ? '#27ae60' : '#e74c3c';
  };

  if (loading) {
    return <div className="loading">Loading metrics...</div>;
  }

  return (
    <div className="metrics-container">
      <div className="page-header">
        <h1>
          <FaChartLine />
          Success Metrics
        </h1>
        <p>Performance indicators and customer success tracking</p>
      </div>

      {/* Controls */}
      <div className="metrics-controls">
        <div className="control-group">
          <label>Time Period</label>
          <select value={selectedPeriod} onChange={(e) => setSelectedPeriod(e.target.value)}>
            {periods.map(period => (
              <option key={period.value} value={period.value}>
                {period.label}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Metric Type</label>
          <select value={selectedMetricType} onChange={(e) => setSelectedMetricType(e.target.value)}>
            {metricTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Key Performance Indicators */}
      <div className="kpi-section">
        <h2>Key Performance Indicators</h2>
        <div className="kpi-grid">
          {Object.entries(filteredMetrics).map(([metricType, values]) => {
            const Icon = getMetricIcon(metricType);
            const trend = calculateTrend(values);
            const latestValue = values[values.length - 1];
            const unit = getMetricUnit(metricType);
            
            if (!latestValue) return null;

            return (
              <div key={metricType} className="kpi-card">
                <div className="kpi-header">
                  <div className="kpi-icon">
                    <Icon />
                  </div>
                  <div className="kpi-trend" style={{ color: getTrendColor(trend) }}>
                    {getTrendIcon(trend)}
                    <span>{Math.abs(trend).toFixed(1)}%</span>
                  </div>
                </div>
                
                <div className="kpi-value">
                  {latestValue.metric_value}{unit}
                </div>
                
                <div className="kpi-info">
                  <h3>{metricTypes.find(t => t.value === metricType)?.label || metricType}</h3>
                  <p>Target: {latestValue.target_value || 'Not set'}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Detailed Metrics Charts */}
      <div className="charts-section">
        <h2>Detailed Metrics</h2>
        <div className="charts-grid">
          {Object.entries(filteredMetrics).map(([metricType, values]) => {
            const sortedValues = values.sort((a, b) => 
              new Date(a.measurement_date) - new Date(b.measurement_date)
            );

            return (
              <div key={metricType} className="chart-container">
                <h3>{metricTypes.find(t => t.value === metricType)?.label || metricType}</h3>
                <div className="chart">
                  <div className="chart-bars">
                    {sortedValues.map((value, index) => {
                      const height = value.target_value ? 
                        Math.min((value.metric_value / value.target_value) * 100, 100) : 
                        (value.metric_value / Math.max(...sortedValues.map(v => v.metric_value))) * 100;
                      
                      return (
                        <div key={index} className="chart-bar-container">
                          <div 
                            className="chart-bar"
                            style={{ height: `${height}%` }}
                            title={`${value.measurement_date}: ${value.metric_value}`}
                          />
                          <div className="chart-label">
                            {new Date(value.measurement_date).toLocaleDateString()}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  <div className="chart-legend">
                    <div className="legend-item">
                      <div className="legend-color" style={{ backgroundColor: '#3498db' }}></div>
                      <span>Actual</span>
                    </div>
                    {values[0]?.target_value && (
                      <div className="legend-item">
                        <div className="legend-color" style={{ backgroundColor: '#27ae60' }}></div>
                        <span>Target: {values[0].target_value}{getMetricUnit(metricType)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Metrics Summary Table */}
      <div className="metrics-table-section">
        <h2>Metrics Summary</h2>
        <div className="metrics-table">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Current Value</th>
                <th>Target</th>
                <th>Performance</th>
                <th>Trend</th>
                <th>Last Updated</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(filteredMetrics).map(([metricType, values]) => {
                const latestValue = values[values.length - 1];
                const trend = calculateTrend(values);
                
                if (!latestValue) return null;

                const performance = latestValue.target_value ? 
                  ((latestValue.metric_value / latestValue.target_value) * 100).toFixed(1) + '%' : 
                  'N/A';

                return (
                  <tr key={metricType}>
                    <td className="metric-name">
                      {metricTypes.find(t => t.value === metricType)?.label || metricType}
                    </td>
                    <td>
                      {latestValue.metric_value}{getMetricUnit(metricType)}
                    </td>
                    <td>
                      {latestValue.target_value ? `${latestValue.target_value}${getMetricUnit(metricType)}` : 'Not set'}
                    </td>
                    <td>
                      <span className={`performance-badge ${performance >= 100 ? 'good' : 'needs-improvement'}`}>
                        {performance}
                      </span>
                    </td>
                    <td>
                      <div className="trend-indicator" style={{ color: getTrendColor(trend) }}>
                        {getTrendIcon(trend)}
                        <span>{Math.abs(trend).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>{new Date(latestValue.measurement_date).toLocaleDateString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Healthcare-Specific Metrics */}
      <div className="healthcare-metrics">
        <h2>Healthcare-Specific Metrics</h2>
        <div className="healthcare-metrics-grid">
          <div className="healthcare-metric">
            <div className="metric-icon">
              <FaHeartbeat />
            </div>
            <div className="metric-content">
              <h3>Patient Safety Score</h3>
              <div className="metric-score">98.5%</div>
              <p>AI system accuracy in critical medical decisions</p>
            </div>
          </div>

          <div className="healthcare-metric">
            <div className="metric-icon">
              <FaUsers />
            </div>
            <div className="metric-content">
              <h3>Clinical Adoption Rate</h3>
              <div className="metric-score">87%</div>
              <p>Healthcare professionals actively using AI features</p>
            </div>
          </div>

          <div className="healthcare-metric">
            <div className="metric-icon">
              <FaClock />
            </div>
            <div className="metric-content">
              <h3>Medical Response Time</h3>
              <div className="metric-score">2.3 min</div>
              <p>Average time for critical medical queries</p>
            </div>
          </div>

          <div className="healthcare-metric">
            <div className="metric-icon">
              <FaTicketAlt />
            </div>
            <div className="metric-content">
              <h3>Medical Ticket Resolution</h3>
              <div className="metric-score">94.2%</div>
              <p>Within SLA for medical-related issues</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Metrics;