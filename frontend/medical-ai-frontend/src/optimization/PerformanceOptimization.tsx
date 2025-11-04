/**
 * Performance Optimization for Real-Time Medical Applications
 * Target latency < 2s, offline capabilities, and performance monitoring
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Progress } from '../ui/progress';

// Performance metrics
interface PerformanceMetrics {
  responseTime: number; // milliseconds
  latency: number; // round-trip time
  throughput: number; // requests per second
  errorRate: number; // percentage
  availability: number; // percentage uptime
  dataTransferRate: number; // KB/s
  memoryUsage: number; // MB
  cpuUsage: number; // percentage
  networkQuality: 'excellent' | 'good' | 'fair' | 'poor' | 'offline';
  timestamp: Date;
}

// Cache configuration
interface CacheConfig {
  enabled: boolean;
  maxSize: number; // MB
  ttl: number; // seconds
  strategy: 'lru' | 'fifo' | 'lfu';
  criticalDataOnly: boolean;
}

// Offline capabilities
interface OfflineConfig {
  enabled: boolean;
  syncOnReconnect: boolean;
  queueSize: number;
  backgroundSync: boolean;
  criticalOperations: string[];
}

// Performance monitoring configuration
interface PerformanceMonitoring {
  realTimeMetrics: boolean;
  alertingEnabled: boolean;
  performanceBudget: {
    responseTime: number; // ms
    latency: number; // ms
    memoryUsage: number; // MB
  };
  thresholds: {
    warning: number;
    critical: number;
  };
}

// Network quality detection
interface NetworkInfo {
  effectiveType: '4g' | '3g' | '2g' | 'slow-2g';
  downlink: number; // Mbps
  rtt: number; // milliseconds
  saveData: boolean;
}

// Real-time data streaming configuration
interface StreamingConfig {
  enabled: boolean;
  batchSize: number;
  batchTimeout: number; // ms
  priorityChannels: string[];
  quality: 'high' | 'balanced' | 'low';
  compression: boolean;
}

// Performance Budget Alert
const PerformanceAlert: React.FC<{
  metric: string;
  current: number;
  threshold: number;
  severity: 'warning' | 'critical';
  onDismiss: () => void;
}> = ({ metric, current, threshold, severity, onDismiss }) => (
  <Alert className={`mb-4 ${severity === 'critical' ? 'border-red-500 bg-red-50' : 'border-yellow-500 bg-yellow-50'}`}>
    <AlertDescription className={`${severity === 'critical' ? 'text-red-800' : 'text-yellow-800'}`}>
      <div className="flex items-center justify-between">
        <div>
          <strong>{metric}</strong> is {severity === 'critical' ? 'critically' : 'significantly'} high: {current.toFixed(2)}ms (threshold: {threshold}ms)
        </div>
        <Button variant="outline" size="sm" onClick={onDismiss}>
          Dismiss
        </Button>
      </div>
    </AlertDescription>
  </Alert>
);

// Network Quality Indicator
const NetworkQualityIndicator: React.FC<{
  networkInfo: NetworkInfo | null;
  latency: number;
}> = ({ networkInfo, latency }) => {
  const getQualityColor = () => {
    if (!networkInfo) return 'bg-gray-500';
    if (networkInfo.effectiveType === '4g' && latency < 100) return 'bg-green-500';
    if (networkInfo.effectiveType === '3g' && latency < 200) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getQualityLabel = () => {
    if (!networkInfo) return 'Offline';
    if (networkInfo.effectiveType === '4g' && latency < 100) return 'Excellent';
    if (networkInfo.effectiveType === '3g' && latency < 200) return 'Good';
    if (networkInfo.effectiveType === '3g') return 'Fair';
    return 'Poor';
  };

  return (
    <div className="flex items-center space-x-2">
      <div className={`w-3 h-3 rounded-full ${getQualityColor()}`} />
      <span className="text-sm font-medium">{getQualityLabel()}</span>
      {networkInfo && (
        <span className="text-xs text-gray-500">
          ({networkInfo.effectiveType.toUpperCase()}, {networkInfo.rtt}ms)
        </span>
      )}
    </div>
  );
};

// Offline Data Queue
interface QueueItem {
  id: string;
  type: 'patient_data' | 'vital_signs' | 'medication_update' | 'emergency_alert';
  data: any;
  timestamp: Date;
  priority: 'low' | 'normal' | 'high' | 'critical';
  attempts: number;
  maxAttempts: number;
}

class OfflineQueueManager {
  private queue: QueueItem[] = [];
  private storageKey = 'medical_app_offline_queue';

  constructor() {
    this.loadFromStorage();
  }

  add(item: Omit<QueueItem, 'id' | 'timestamp' | 'attempts'>): string {
    const queueItem: QueueItem = {
      ...item,
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      timestamp: new Date(),
      attempts: 0
    };

    this.queue.push(queueItem);
    this.saveToStorage();
    return queueItem.id;
  }

  getNext(): QueueItem | null {
    // Sort by priority and timestamp
    this.queue.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, normal: 2, low: 1 };
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return a.timestamp.getTime() - b.timestamp.getTime();
    });

    return this.queue[0] || null;
  }

  markProcessed(id: string, success: boolean): void {
    const item = this.queue.find(item => item.id === id);
    if (item) {
      if (success) {
        // Remove successfully processed item
        this.queue = this.queue.filter(item => item.id !== id);
      } else {
        // Increment attempts and remove if max reached
        item.attempts++;
        if (item.attempts >= item.maxAttempts) {
          this.queue = this.queue.filter(item => item.id !== id);
        }
      }
      this.saveToStorage();
    }
  }

  getSize(): number {
    return this.queue.length;
  }

  getCriticalCount(): number {
    return this.queue.filter(item => item.priority === 'critical').length;
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        this.queue = JSON.parse(stored).map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp)
        }));
      }
    } catch (error) {
      console.error('Failed to load offline queue:', error);
    }
  }

  private saveToStorage(): void {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.queue));
    } catch (error) {
      console.error('Failed to save offline queue:', error);
    }
  }
}

// Performance Monitoring Component
export const PerformanceOptimization: React.FC<{
  onPerformanceUpdate?: (metrics: PerformanceMetrics) => void;
  onNetworkChange?: (isOnline: boolean) => void;
}> = ({ onPerformanceUpdate, onNetworkChange }) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    responseTime: 0,
    latency: 0,
    throughput: 0,
    errorRate: 0,
    availability: 100,
    dataTransferRate: 0,
    memoryUsage: 0,
    cpuUsage: 0,
    networkQuality: 'good',
    timestamp: new Date()
  });

  const [alerts, setAlerts] = useState<Array<{
    id: string;
    metric: string;
    current: number;
    threshold: number;
    severity: 'warning' | 'critical';
  }>>([]);

  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [cacheConfig, setCacheConfig] = useState<CacheConfig>({
    enabled: true,
    maxSize: 50,
    ttl: 300,
    strategy: 'lru',
    criticalDataOnly: false
  });

  const [offlineConfig, setOfflineConfig] = useState<OfflineConfig>({
    enabled: true,
    syncOnReconnect: true,
    queueSize: 1000,
    backgroundSync: true,
    criticalOperations: ['emergency_alert', 'patient_critical_vitals', 'medication_critical_update']
  });

  const [monitoringConfig, setMonitoringConfig] = useState<PerformanceMonitoring>({
    realTimeMetrics: true,
    alertingEnabled: true,
    performanceBudget: {
      responseTime: 2000, // 2 seconds
      latency: 500, // 500ms
      memoryUsage: 100 // 100MB
    },
    thresholds: {
      warning: 0.8, // 80% of budget
      critical: 0.95 // 95% of budget
    }
  });

  const [streamingConfig, setStreamingConfig] = useState<StreamingConfig>({
    enabled: true,
    batchSize: 10,
    batchTimeout: 100,
    priorityChannels: ['vital_signs', 'emergency_alerts', 'patient_status'],
    quality: 'balanced',
    compression: true
  });

  const queueManagerRef = useRef(new OfflineQueueManager());
  const performanceMonitorRef = useRef<any>(null);

  // Network status monitoring
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      onNetworkChange?.(true);
      if (offlineConfig.syncOnReconnect) {
        syncOfflineData();
      }
    };

    const handleOffline = () => {
      setIsOnline(false);
      onNetworkChange?.(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Network Information API
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      setNetworkInfo({
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
        saveData: connection.saveData
      });

      const updateNetworkInfo = () => {
        setNetworkInfo({
          effectiveType: connection.effectiveType,
          downlink: connection.downlink,
          rtt: connection.rtt,
          saveData: connection.saveData
        });
      };

      connection.addEventListener('change', updateNetworkInfo);
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [offlineConfig.syncOnReconnect, onNetworkChange]);

  // Performance monitoring
  useEffect(() => {
    if (!monitoringConfig.realTimeMetrics) return;

    const interval = setInterval(() => {
      const startTime = performance.now();
      
      // Simulate API call measurement
      fetch('/api/health', { method: 'HEAD', cache: 'no-store' })
        .then(() => {
          const endTime = performance.now();
          const responseTime = endTime - startTime;
          
          // Update metrics
          setMetrics(prev => ({
            ...prev,
            responseTime,
            latency: Math.max(responseTime * 0.8, 50), // Estimate RTT
            throughput: Math.random() * 100 + 50, // Simulate throughput
            errorRate: Math.random() * 2, // Simulate error rate
            memoryUsage: (performance as any).memory?.usedJSHeapSize / 1024 / 1024 || 0,
            cpuUsage: Math.random() * 100,
            networkQuality: isOnline ? 
              (responseTime < 200 ? 'excellent' : responseTime < 500 ? 'good' : 'fair') : 'offline',
            timestamp: new Date()
          }));

          // Performance alerting
          if (monitoringConfig.alertingEnabled) {
            checkPerformanceAlerts({
              ...metrics,
              responseTime,
              latency: Math.max(responseTime * 0.8, 50),
              memoryUsage: (performance as any).memory?.usedJSHeapSize / 1024 / 1024 || 0
            });
          }

          onPerformanceUpdate?.({
            ...metrics,
            responseTime,
            latency: Math.max(responseTime * 0.8, 50),
            timestamp: new Date()
          });
        })
        .catch(() => {
          setMetrics(prev => ({
            ...prev,
            errorRate: Math.min(prev.errorRate + 0.1, 10),
            networkQuality: 'poor'
          }));
        });
    }, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, [monitoringConfig.realTimeMetrics, monitoringConfig.alertingEnabled, monitoringConfig.performanceBudget, isOnline, onPerformanceUpdate]);

  // Performance alerting
  const checkPerformanceAlerts = useCallback((currentMetrics: PerformanceMetrics) => {
    const newAlerts: typeof alerts = [];

    // Response time alerts
    if (currentMetrics.responseTime > monitoringConfig.performanceBudget.responseTime * monitoringConfig.thresholds.critical) {
      newAlerts.push({
        id: 'response-time-critical',
        metric: 'Response Time',
        current: currentMetrics.responseTime,
        threshold: monitoringConfig.performanceBudget.responseTime,
        severity: 'critical'
      });
    } else if (currentMetrics.responseTime > monitoringConfig.performanceBudget.responseTime * monitoringConfig.thresholds.warning) {
      newAlerts.push({
        id: 'response-time-warning',
        metric: 'Response Time',
        current: currentMetrics.responseTime,
        threshold: monitoringConfig.performanceBudget.responseTime,
        severity: 'warning'
      });
    }

    // Latency alerts
    if (currentMetrics.latency > monitoringConfig.performanceBudget.latency * monitoringConfig.thresholds.critical) {
      newAlerts.push({
        id: 'latency-critical',
        metric: 'Network Latency',
        current: currentMetrics.latency,
        threshold: monitoringConfig.performanceBudget.latency,
        severity: 'critical'
      });
    }

    // Memory usage alerts
    if (currentMetrics.memoryUsage > monitoringConfig.performanceBudget.memoryUsage * monitoringConfig.thresholds.critical) {
      newAlerts.push({
        id: 'memory-critical',
        metric: 'Memory Usage',
        current: currentMetrics.memoryUsage,
        threshold: monitoringConfig.performanceBudget.memoryUsage,
        severity: 'critical'
      });
    }

    setAlerts(newAlerts);
  }, [monitoringConfig]);

  // Offline data synchronization
  const syncOfflineData = async () => {
    const queueManager = queueManagerRef.current;
    const criticalCount = queueManager.getCriticalCount();

    if (criticalCount > 0) {
      console.log(`Syncing ${criticalCount} critical offline items...`);
      // Simulate sync process
      let syncAttempts = 0;
      const maxSyncAttempts = 3;

      while (syncAttempts < maxSyncAttempts) {
        const nextItem = queueManager.getNext();
        if (!nextItem) break;

        try {
          // Simulate API call
          await new Promise(resolve => setTimeout(resolve, 100));
          queueManager.markProcessed(nextItem.id, true);
          console.log(`Synced ${nextItem.type} item:`, nextItem.id);
        } catch (error) {
          console.error(`Failed to sync item ${nextItem.id}:`, error);
          queueManager.markProcessed(nextItem.id, false);
        }
        
        syncAttempts++;
      }
    }
  };

  // Add to offline queue
  const addToOfflineQueue = (type: QueueItem['type'], data: any, priority: QueueItem['priority'] = 'normal') => {
    const queueManager = queueManagerRef.current;
    const id = queueManager.add({ type, data, priority, maxAttempts: 3 });
    
    if (!isOnline) {
      console.log(`Added ${type} to offline queue (ID: ${id})`);
    }
    
    return id;
  };

  // Cache management
  const clearCache = () => {
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => {
          caches.delete(name);
        });
      });
    }
  };

  const getCacheSize = async (): Promise<number> => {
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      let totalSize = 0;
      
      for (const name of cacheNames) {
        const cache = await caches.open(name);
        const requests = await cache.keys();
        for (const request of requests) {
          const response = await cache.match(request);
          if (response) {
            const blob = await response.blob();
            totalSize += blob.size;
          }
        }
      }
      
      return totalSize / 1024 / 1024; // Convert to MB
    }
    return 0;
  };

  return (
    <div className="h-full bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Performance Optimization</h1>
            <p className="text-gray-600">Real-time medical application performance monitoring</p>
          </div>
          <div className="flex items-center space-x-4">
            <NetworkQualityIndicator networkInfo={networkInfo} latency={metrics.latency} />
            <Badge variant={isOnline ? 'default' : 'destructive'}>
              {isOnline ? '🟢 Online' : '🔴 Offline'}
            </Badge>
            <Badge variant="outline">
              {metrics.responseTime.toFixed(0)}ms response
            </Badge>
          </div>
        </div>
      </div>

      {/* Performance Alerts */}
      {alerts.length > 0 && (
        <div className="p-6">
          {alerts.map(alert => (
            <PerformanceAlert
              key={alert.id}
              metric={alert.metric}
              current={alert.current}
              threshold={alert.threshold}
              severity={alert.severity}
              onDismiss={() => setAlerts(prev => prev.filter(a => a.id !== alert.id))}
            />
          ))}
        </div>
      )}

      {/* Main Content */}
      <div className="p-6 space-y-6">
        {/* Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className={`text-2xl font-bold ${metrics.responseTime < 2000 ? 'text-green-600' : metrics.responseTime < 3000 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {metrics.responseTime.toFixed(0)}ms
                </div>
                <div className="text-sm text-gray-600">Response Time</div>
                <Progress 
                  value={Math.min((metrics.responseTime / 3000) * 100, 100)} 
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className={`text-2xl font-bold ${metrics.latency < 500 ? 'text-green-600' : metrics.latency < 800 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {metrics.latency.toFixed(0)}ms
                </div>
                <div className="text-sm text-gray-600">Network Latency</div>
                <Progress 
                  value={Math.min((metrics.latency / 1000) * 100, 100)} 
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(metrics.memoryUsage || 0).toFixed(1)}MB
                </div>
                <div className="text-sm text-gray-600">Memory Usage</div>
                <Progress 
                  value={Math.min(((metrics.memoryUsage || 0) / 200) * 100, 100)} 
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <div className={`text-2xl font-bold ${metrics.errorRate < 1 ? 'text-green-600' : metrics.errorRate < 3 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {metrics.errorRate.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Error Rate</div>
                <Progress 
                  value={Math.min(metrics.errorRate * 10, 100)} 
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Configuration Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Cache Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Cache Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Enable Caching</label>
                <input
                  type="checkbox"
                  checked={cacheConfig.enabled}
                  onChange={(e) => setCacheConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Max Size (MB)</label>
                <input
                  type="number"
                  value={cacheConfig.maxSize}
                  onChange={(e) => setCacheConfig(prev => ({ ...prev, maxSize: parseInt(e.target.value) }))}
                  className="w-full border rounded px-3 py-2 mt-1"
                  min="10"
                  max="500"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">TTL (seconds)</label>
                <input
                  type="number"
                  value={cacheConfig.ttl}
                  onChange={(e) => setCacheConfig(prev => ({ ...prev, ttl: parseInt(e.target.value) }))}
                  className="w-full border rounded px-3 py-2 mt-1"
                  min="60"
                  max="3600"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Critical Data Only</label>
                <input
                  type="checkbox"
                  checked={cacheConfig.criticalDataOnly}
                  onChange={(e) => setCacheConfig(prev => ({ ...prev, criticalDataOnly: e.target.checked }))}
                />
              </div>
              
              <Button variant="outline" onClick={clearCache} className="w-full">
                Clear All Cache
              </Button>
            </CardContent>
          </Card>

          {/* Offline Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Offline Capabilities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Enable Offline Mode</label>
                <input
                  type="checkbox"
                  checked={offlineConfig.enabled}
                  onChange={(e) => setOfflineConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Sync on Reconnect</label>
                <input
                  type="checkbox"
                  checked={offlineConfig.syncOnReconnect}
                  onChange={(e) => setOfflineConfig(prev => ({ ...prev, syncOnReconnect: e.target.checked }))}
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Queue Size</label>
                <input
                  type="number"
                  value={offlineConfig.queueSize}
                  onChange={(e) => setOfflineConfig(prev => ({ ...prev, queueSize: parseInt(e.target.value) }))}
                  className="w-full border rounded px-3 py-2 mt-1"
                  min="100"
                  max="10000"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Background Sync</label>
                <input
                  type="checkbox"
                  checked={offlineConfig.backgroundSync}
                  onChange={(e) => setOfflineConfig(prev => ({ ...prev, backgroundSync: e.target.checked }))}
                />
              </div>
              
              <div className="pt-2 border-t">
                <div className="text-sm font-medium mb-2">Offline Queue Status</div>
                <div className="space-y-1 text-sm">
                  <div>Items in queue: {queueManagerRef.current.getSize()}</div>
                  <div>Critical items: {queueManagerRef.current.getCriticalCount()}</div>
                  <Badge variant={isOnline ? 'default' : 'destructive'}>
                    {isOnline ? 'Ready to sync' : 'Offline mode active'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance Monitoring */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Monitoring</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium">Response Time Budget (ms)</label>
                <input
                  type="number"
                  value={monitoringConfig.performanceBudget.responseTime}
                  onChange={(e) => setMonitoringConfig(prev => ({
                    ...prev,
                    performanceBudget: { ...prev.performanceBudget, responseTime: parseInt(e.target.value) }
                  }))}
                  className="w-full border rounded px-3 py-2 mt-1"
                  min="500"
                  max="5000"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Latency Budget (ms)</label>
                <input
                  type="number"
                  value={monitoringConfig.performanceBudget.latency}
                  onChange={(e) => setMonitoringConfig(prev => ({
                    ...prev,
                    performanceBudget: { ...prev.performanceBudget, latency: parseInt(e.target.value) }
                  }))}
                  className="w-full border rounded px-3 py-2 mt-1"
                  min="100"
                  max="1000"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Memory Budget (MB)</label>
                <input
                  type="number"
                  value={monitoringConfig.performanceBudget.memoryUsage}
                  onChange={(e) => setMonitoringConfig(prev => ({
                    ...prev,
                    performanceBudget: { ...prev.performanceBudget, memoryUsage: parseInt(e.target.value) }
                  }))}
                  className="w-full border rounded px-3 py-2 mt-1"
                  min="50"
                  max="500"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-4 pt-4 border-t">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={monitoringConfig.realTimeMetrics}
                  onChange={(e) => setMonitoringConfig(prev => ({ ...prev, realTimeMetrics: e.target.checked }))}
                />
                <label className="text-sm font-medium">Real-time Monitoring</label>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={monitoringConfig.alertingEnabled}
                  onChange={(e) => setMonitoringConfig(prev => ({ ...prev, alertingEnabled: e.target.checked }))}
                />
                <label className="text-sm font-medium">Performance Alerts</label>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Test Offline Functionality */}
        <Card>
          <CardHeader>
            <CardTitle>Test Offline Functionality</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-600">
              Test the offline capabilities by simulating network disconnection and adding data to the offline queue.
            </p>
            
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                onClick={() => addToOfflineQueue('vital_signs', { patientId: '123', heartRate: 75 }, 'normal')}
                disabled={!offlineConfig.enabled}
              >
                Add Vital Signs
              </Button>
              
              <Button
                variant="outline"
                onClick={() => addToOfflineQueue('emergency_alert', { patientId: '456', alert: 'Patient distress' }, 'critical')}
                disabled={!offlineConfig.enabled}
              >
                Add Emergency Alert
              </Button>
              
              <Button
                variant="outline"
                onClick={() => addToOfflineQueue('medication_update', { patientId: '789', medication: 'Morphine', dose: '5mg' }, 'high')}
                disabled={!offlineConfig.enabled}
              >
                Add Medication Update
              </Button>
              
              <Button
                variant="outline"
                onClick={syncOfflineData}
                disabled={!isOnline || !offlineConfig.syncOnReconnect}
              >
                Force Sync
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PerformanceOptimization;