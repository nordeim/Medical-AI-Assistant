// Real-time Activity Monitor for Healthcare User Management
// Production-grade monitoring with anomaly detection and alerting

const EventEmitter = require('events');
const config = require('../config/user-management-config');

class RealTimeMonitor extends EventEmitter {
  constructor() {
    super();
    this.config = config.monitoring;
    this.activeSessions = new Map();
    this.activityPatterns = new Map();
    this.thresholdAlerts = new Map();
    this.anomalyModels = this.initializeAnomalyModels();
    this.monitoringInterval = null;
  }

  // Initialize Real-time Monitoring
  async initialize() {
    await this.setupActivityTracking();
    await this.initializeAnomalyDetection();
    this.startRealTimeProcessing();
    console.log('Real-time monitoring initialized');
  }

  // Track User Activity in Real-time
  async trackUserActivity(userId, activityData) {
    try {
      const {
        action,
        resource,
        context,
        timestamp = new Date().toISOString(),
        ipAddress,
        userAgent,
        sessionId
      } = activityData;

      // Create activity record
      const activityRecord = {
        userId,
        action,
        resource,
        context,
        timestamp,
        ipAddress: this.hashIP(ipAddress),
        userAgent: this.sanitizeUserAgent(userAgent),
        sessionId,
        riskScore: await this.calculateActivityRiskScore(activityData)
      };

      // Store activity
      await this.storeActivityRecord(activityRecord);

      // Update user session
      await this.updateUserSession(userId, sessionId, activityRecord);

      // Analyze for patterns
      await this.analyzeActivityPatterns(userId, activityRecord);

      // Check thresholds
      await this.checkThresholds(userId, activityRecord);

      // Detect anomalies
      await this.detectAnomalies(userId, activityRecord);

      // Emit events for real-time processing
      this.emit('userActivity', activityRecord);

      return activityRecord;

    } catch (error) {
      console.error('Activity tracking error:', error);
      throw error;
    }
  }

  // Pattern Analysis
  async analyzeActivityPatterns(userId, currentActivity) {
    try {
      const userPatternKey = `user_${userId}`;
      let userPattern = this.activityPatterns.get(userPatternKey);

      if (!userPattern) {
        userPattern = {
          userId,
          commonActions: new Map(),
          accessTimes: [],
          resources: new Map(),
          ipAddresses: new Map(),
          userAgents: new Map(),
          patterns: {
            dailyActivity: {},
            weeklyActivity: {},
            monthlyActivity: {}
          }
        };
      }

      // Update action frequency
      const actionCount = userPattern.commonActions.get(currentActivity.action) || 0;
      userPattern.commonActions.set(currentActivity.action, actionCount + 1);

      // Update access time patterns
      const accessTime = new Date(currentActivity.timestamp);
      const hourKey = accessTime.getHours();
      const dayKey = accessTime.getDay();
      const weekKey = Math.floor(accessTime.getTime() / (7 * 24 * 60 * 60 * 1000));
      
      userPattern.patterns.dailyActivity[hourKey] = (userPattern.patterns.dailyActivity[hourKey] || 0) + 1;
      userPattern.patterns.weeklyActivity[dayKey] = (userPattern.patterns.weeklyActivity[dayKey] || 0) + 1;
      userPattern.patterns.monthlyActivity[weekKey] = (userPattern.patterns.monthlyActivity[weekKey] || 0) + 1;

      // Update resource access patterns
      const resourceCount = userPattern.resources.get(currentActivity.resource) || 0;
      userPattern.resources.set(currentActivity.resource, resourceCount + 1);

      // Update IP address patterns
      const ipCount = userPattern.ipAddresses.get(currentActivity.ipAddress) || 0;
      userPattern.ipAddresses.set(currentActivity.ipAddress, ipCount + 1);

      // Detect new patterns or deviations
      const patternChanges = await this.detectPatternChanges(userPattern, currentActivity);

      if (patternChanges.hasSignificantChange) {
        this.emit('patternChange', {
          userId,
          changes: patternChanges,
          activity: currentActivity
        });
      }

      // Store updated pattern
      this.activityPatterns.set(userPatternKey, userPattern);

      // Cleanup old patterns periodically
      await this.cleanupActivityPatterns();

    } catch (error) {
      console.error('Pattern analysis error:', error);
    }
  }

  // Threshold Monitoring
  async checkThresholds(userId, activity) {
    const thresholds = this.config.monitoring.thresholds;

    // Check rate limits
    await this.checkRateLimits(userId, activity, thresholds);

    // Check access patterns
    await this.checkAccessPatterns(userId, activity, thresholds);

    // Check time-based patterns
    await this.checkTimePatterns(userId, activity, thresholds);

    // Check resource limits
    await this.checkResourceLimits(userId, activity, thresholds);
  }

  // Anomaly Detection
  async detectAnomalies(userId, activity) {
    try {
      const anomalyScore = await this.calculateAnomalyScore(userId, activity);
      
      if (anomalyScore > 0.7) { // High anomaly threshold
        const anomaly = {
          userId,
          activity,
          anomalyScore,
          timestamp: new Date().toISOString(),
          anomalyTypes: await this.identifyAnomalyTypes(userId, activity, anomalyScore),
          severity: this.determineAnomalySeverity(anomalyScore)
        };

        // Store anomaly
        await this.storeAnomaly(anomaly);

        // Emit anomaly event
        this.emit('anomalyDetected', anomaly);

        // Trigger immediate response for critical anomalies
        if (anomaly.severity === 'critical') {
          await this.handleCriticalAnomaly(anomaly);
        }
      }

    } catch (error) {
      console.error('Anomaly detection error:', error);
    }
  }

  // Event Processing
  async processEvent(event) {
    const processingStartTime = Date.now();

    try {
      // Categorize event
      const category = this.categorizeEvent(event);

      // Apply category-specific processing
      switch (category) {
        case 'security':
          await this.processSecurityEvent(event);
          break;
        case 'performance':
          await this.processPerformanceEvent(event);
          break;
        case 'compliance':
          await this.processComplianceEvent(event);
          break;
        case 'user_behavior':
          await this.processUserBehaviorEvent(event);
          break;
        default:
          await this.processGenericEvent(event);
      }

      // Update real-time metrics
      await this.updateRealTimeMetrics(event);

      // Check for immediate threats
      await this.checkImmediateThreats(event);

      const processingTime = Date.now() - processingStartTime;
      
      // Log processing performance
      if (processingTime > 1000) { // Log slow processing
        console.warn(`Slow event processing: ${processingTime}ms for event ${event.eventId}`);
      }

    } catch (error) {
      console.error('Event processing error:', error);
      
      // Emit error event for monitoring
      this.emit('processingError', {
        event,
        error: error.message,
        processingTime: Date.now() - processingStartTime
      });
    }
  }

  // Health Monitoring
  async performHealthCheck() {
    const healthMetrics = {
      timestamp: new Date().toISOString(),
      status: 'healthy',
      metrics: {
        activeSessions: this.activeSessions.size,
        activityPatterns: this.activityPatterns.size,
        thresholdAlerts: this.thresholdAlerts.size,
        processingQueue: this.getProcessingQueueSize(),
        anomalyDetectionRate: await this.getAnomalyDetectionRate(),
        falsePositiveRate: await this.getFalsePositiveRate()
      },
      alerts: await this.getActiveAlerts(),
      recommendations: []
    };

    // Check critical thresholds
    if (healthMetrics.metrics.activeSessions > 10000) {
      healthMetrics.alerts.push({
        type: 'high_load',
        severity: 'medium',
        message: 'High number of active sessions detected'
      });
    }

    if (healthMetrics.metrics.anomalyDetectionRate < 0.01) {
      healthMetrics.recommendations.push({
        type: 'model_retraining',
        message: 'Anomaly detection rate is low - consider model retraining'
      });
    }

    return healthMetrics;
  }

  // Helper Methods
  initializeAnomalyModels() {
    return {
      isolationForest: this.createIsolationForestModel(),
      statisticalModel: this.createStatisticalModel(),
      mlModel: this.createMLAnomalyModel()
    };
  }

  async calculateActivityRiskScore(activity) {
    let riskScore = 0;

    // Time-based risk
    const accessTime = new Date(activity.timestamp);
    const hour = accessTime.getHours();
    
    // Off-hours access increases risk
    if (hour < 6 || hour > 22) {
      riskScore += 20;
    }

    // Weekend access
    if (accessTime.getDay() === 0 || accessTime.getDay() === 6) {
      riskScore += 15;
    }

    // New IP address
    const userPattern = this.activityPatterns.get(`user_${activity.userId}`);
    if (userPattern && !userPattern.ipAddresses.has(activity.ipAddress)) {
      riskScore += 30;
    }

    // New resource access
    if (userPattern && !userPattern.resources.has(activity.resource)) {
      riskScore += 25;
    }

    // High-risk actions
    const highRiskActions = ['delete', 'export', 'admin_change', 'role_change'];
    if (highRiskActions.includes(activity.action)) {
      riskScore += 40;
    }

    return Math.min(riskScore, 100);
  }

  async detectPatternChanges(userPattern, currentActivity) {
    const changes = {
      hasSignificantChange: false,
      changes: [],
      confidence: 0
    };

    // Check unusual access time
    const accessTime = new Date(currentActivity.timestamp);
    const hour = accessTime.getHours();
    const usualHours = userPattern.patterns.dailyActivity;
    
    const usualHourFrequency = usualHours[hour] || 0;
    const averageHourly = Object.values(usualHours).reduce((a, b) => a + b, 0) / Object.keys(usualHours).length;
    
    if (usualHourFrequency < averageHourly * 0.1 && averageHourly > 5) {
      changes.changes.push({
        type: 'unusual_time',
        detail: `Accessing at unusual hour: ${hour}`,
        significance: 'medium'
      });
    }

    // Check new IP location
    const ipFrequency = userPattern.ipAddresses.get(currentActivity.ipAddress) || 0;
    if (ipFrequency === 0) {
      changes.changes.push({
        type: 'new_location',
        detail: 'Accessing from new IP address',
        significance: 'high'
      });
    }

    // Check unusual resource access
    const resourceFrequency = userPattern.resources.get(currentActivity.resource) || 0;
    const avgResourceFrequency = Array.from(userPattern.resources.values()).reduce((a, b) => a + b, 0) / userPattern.resources.size;
    
    if (resourceFrequency === 0 && avgResourceFrequency > 10) {
      changes.changes.push({
        type: 'new_resource',
        detail: `Accessing new resource: ${currentActivity.resource}`,
        significance: 'medium'
      });
    }

    changes.hasSignificantChange = changes.changes.length > 0;
    changes.confidence = changes.changes.reduce((acc, change) => {
      const weight = change.significance === 'high' ? 0.8 : change.significance === 'medium' ? 0.6 : 0.4;
      return acc + weight;
    }, 0) / changes.changes.length;

    return changes;
  }

  async calculateAnomalyScore(userId, activity) {
    // Combine multiple anomaly detection methods
    const isolationScore = await this.isolationForestScore(userId, activity);
    const statisticalScore = await this.statisticalAnomalyScore(userId, activity);
    const behavioralScore = await this.behavioralAnomalyScore(userId, activity);

    // Weighted average
    const weights = { isolation: 0.4, statistical: 0.3, behavioral: 0.3 };
    
    return (
      isolationScore * weights.isolation +
      statisticalScore * weights.statistical +
      behavioralScore * weights.behavioral
    );
  }

  async identifyAnomalyTypes(userId, activity, anomalyScore) {
    const anomalyTypes = [];

    // Time-based anomalies
    const accessTime = new Date(activity.timestamp);
    const hour = accessTime.getHours();
    if (hour < 6 || hour > 22) {
      anomalyTypes.push('unusual_time_access');
    }

    // Location-based anomalies
    const userPattern = this.activityPatterns.get(`user_${userId}`);
    if (userPattern && !userPattern.ipAddresses.has(activity.ipAddress)) {
      anomalyTypes.push('new_location');
    }

    // Resource-based anomalies
    if (userPattern && !userPattern.resources.has(activity.resource)) {
      anomalyTypes.push('unauthorized_resource_access');
    }

    // Frequency anomalies
    const recentActivities = await this.getRecentActivities(userId, 60); // Last 60 minutes
    if (recentActivities.length > 20) {
      anomalyTypes.push('high_frequency_access');
    }

    // Action-based anomalies
    const highRiskActions = ['delete', 'export', 'admin_change'];
    if (highRiskActions.includes(activity.action) && anomalyScore > 0.8) {
      anomalyTypes.push('suspicious_high_risk_action');
    }

    return anomalyTypes;
  }

  determineAnomalySeverity(anomalyScore) {
    if (anomalyScore > 0.9) return 'critical';
    if (anomalyScore > 0.8) return 'high';
    if (anomalyScore > 0.7) return 'medium';
    return 'low';
  }

  // Database Operations
  async storeActivityRecord(record) {
    await require('../database/user-database').storeActivityRecord(record);
  }

  async updateUserSession(userId, sessionId, activity) {
    if (!sessionId) return;

    const sessionKey = `${userId}_${sessionId}`;
    let session = this.activeSessions.get(sessionKey);

    if (!session) {
      session = {
        sessionId,
        userId,
        startTime: new Date().toISOString(),
        activities: [],
        lastActivity: null,
        riskScore: 0
      };
    }

    session.lastActivity = activity.timestamp;
    session.activities.push(activity);
    session.riskScore = activity.riskScore;

    this.activeSessions.set(sessionKey, session);

    // Update database session
    await require('../database/user-database').updateUserSession(session);
  }

  async storeAnomaly(anomaly) {
    await require('../database/user-database').storeAnomaly(anomaly);
  }

  async getRecentActivities(userId, minutes) {
    const cutoff = new Date(Date.now() - (minutes * 60 * 1000)).toISOString();
    return await require('../database/user-database').getRecentActivities(userId, cutoff);
  }

  async cleanupActivityPatterns() {
    const cutoff = new Date(Date.now() - (30 * 24 * 60 * 60 * 1000)); // 30 days
    const cutoffTimestamp = cutoff.toISOString();

    for (const [key, pattern] of this.activityPatterns.entries()) {
      const lastActivity = pattern.lastActivity;
      if (lastActivity && new Date(lastActivity) < cutoff) {
        this.activityPatterns.delete(key);
      }
    }
  }

  // Threshold Checks
  async checkRateLimits(userId, activity, thresholds) {
    const recentActivities = await this.getRecentActivities(userId, 15); // Last 15 minutes
    const rateLimit = thresholds.rateLimits?.default || 100;

    if (recentActivities.length > rateLimit) {
      const alert = {
        userId,
        type: 'rate_limit_breach',
        severity: 'medium',
        message: `User ${userId} exceeded rate limit: ${recentActivities.length}/${rateLimit}`,
        activity
      };

      await this.triggerThresholdAlert(alert);
    }
  }

  async checkAccessPatterns(userId, activity, thresholds) {
    const accessThreshold = thresholds.accessPatterns?.failureThreshold || 5;
    const recentFailures = await this.getRecentFailures(userId, 60);

    if (recentFailures.length > accessThreshold) {
      const alert = {
        userId,
        type: 'access_pattern_violation',
        severity: 'high',
        message: `User ${userId} has ${recentFailures.length} failed access attempts`,
        activity
      };

      await this.triggerThresholdAlert(alert);
    }
  }

  async checkTimePatterns(userId, activity, thresholds) {
    const offHoursThreshold = thresholds.timePatterns?.offHoursThreshold || 3;
    const recentOffHours = await this.getRecentOffHoursAccess(userId, 1440); // Last 24 hours

    if (recentOffHours.length > offHoursThreshold) {
      const alert = {
        userId,
        type: 'off_hours_access_pattern',
        severity: 'medium',
        message: `User ${userId} has ${recentOffHours.length} off-hours access attempts`,
        activity
      };

      await this.triggerThresholdAlert(alert);
    }
  }

  async checkResourceLimits(userId, activity, thresholds) {
    const exportThreshold = thresholds.resourceLimits?.exportThreshold || 1000;
    
    if (activity.action === 'export') {
      const recentExports = await this.getRecentExports(userId, 60); // Last hour
      const totalExported = recentExports.reduce((sum, exp) => sum + (exp.recordCount || 0), 0);

      if (totalExported > exportThreshold) {
        const alert = {
          userId,
          type: 'export_limit_breach',
          severity: 'high',
          message: `User ${userId} exceeded export threshold: ${totalExported}/${exportThreshold}`,
          activity
        };

        await this.triggerThresholdAlert(alert);
      }
    }
  }

  // Event Processing
  categorizeEvent(event) {
    if (event.event?.includes('security') || event.event?.includes('breach')) {
      return 'security';
    }
    if (event.event?.includes('performance') || event.event?.includes('slow')) {
      return 'performance';
    }
    if (event.event?.includes('compliance') || event.event?.includes('audit')) {
      return 'compliance';
    }
    return 'user_behavior';
  }

  async processSecurityEvent(event) {
    this.emit('securityEvent', event);
    
    if (event.severity === 'critical') {
      await this.handleCriticalSecurityEvent(event);
    }
  }

  async processPerformanceEvent(event) {
    this.emit('performanceEvent', event);
    
    if (event.details?.responseTime > 5000) { // 5 seconds
      await this.handlePerformanceIssue(event);
    }
  }

  async processComplianceEvent(event) {
    this.emit('complianceEvent', event);
  }

  async processUserBehaviorEvent(event) {
    this.emit('userBehaviorEvent', event);
  }

  async processGenericEvent(event) {
    this.emit('genericEvent', event);
  }

  // Anomaly Detection Models
  async isolationForestScore(userId, activity) {
    // Simplified isolation forest score calculation
    const userPattern = this.activityPatterns.get(`user_${userId}`);
    if (!userPattern) return 0.5;

    let isolationScore = 0;
    const factors = [
      { check: () => !userPattern.ipAddresses.has(activity.ipAddress), weight: 0.3 },
      { check: () => !userPattern.resources.has(activity.resource), weight: 0.2 },
      { check: () => activity.riskScore > 70, weight: 0.4 },
      { check: () => this.isOffHours(activity.timestamp), weight: 0.1 }
    ];

    for (const factor of factors) {
      if (factor.check()) {
        isolationScore += factor.weight;
      }
    }

    return Math.min(isolationScore, 1.0);
  }

  async statisticalAnomalyScore(userId, activity) {
    const userPattern = this.activityPatterns.get(`user_${userId}`);
    if (!userPattern) return 0.5;

    // Calculate z-score based patterns
    const hourlyPattern = userPattern.patterns.dailyActivity;
    const accessHour = new Date(activity.timestamp).getHours();
    const usualFrequency = hourlyPattern[accessHour] || 0;
    const meanFrequency = Object.values(hourlyPattern).reduce((a, b) => a + b, 0) / Object.keys(hourlyPattern).length;
    const stdDev = this.calculateStandardDeviation(Object.values(hourlyPattern));
    
    if (stdDev === 0) return 0.5;
    
    const zScore = Math.abs((usualFrequency - meanFrequency) / stdDev);
    return Math.min(zScore / 3, 1.0); // Normalized z-score
  }

  async behavioralAnomalyScore(userId, activity) {
    const recentActivities = await this.getRecentActivities(userId, 1440); // Last 24 hours
    if (recentActivities.length < 10) return 0.3;

    // Compare current activity to behavioral baseline
    const baselineActions = this.calculateBaselineActions(recentActivities);
    const currentActionFreq = baselineActions[activity.action] || 0;
    
    // Calculate deviation from baseline
    const totalBaselineActions = Object.values(baselineActions).reduce((a, b) => a + b, 0);
    const currentFreqRatio = currentActionFreq / totalBaselineActions;
    
    if (currentFreqRatio < 0.1 && totalBaselineActions > 50) {
      return 0.7; // High anomaly score for unusual actions
    }
    
    return 0.2; // Low anomaly score
  }

  // Event Handlers
  async handleCriticalAnomaly(anomaly) {
    await this.triggerImmediateAlert({
      type: 'critical_anomaly',
      userId: anomaly.userId,
      severity: 'critical',
      anomaly,
      immediateAction: true
    });
  }

  async handleCriticalSecurityEvent(event) {
    await this.triggerImmediateAlert({
      type: 'critical_security_event',
      severity: 'critical',
      event,
      immediateAction: true
    });
  }

  async handlePerformanceIssue(event) {
    await this.triggerImmediateAlert({
      type: 'performance_issue',
      severity: 'high',
      event,
      immediateAction: false
    });
  }

  // Alert Management
  async triggerThresholdAlert(alert) {
    this.thresholdAlerts.set(`${alert.userId}_${alert.type}`, alert);
    this.emit('thresholdAlert', alert);
  }

  async triggerImmediateAlert(alert) {
    this.emit('immediateAlert', alert);
  }

  // Monitoring Control
  startRealTimeProcessing() {
    this.monitoringInterval = setInterval(async () => {
      await this.performPeriodicChecks();
      await this.cleanupExpiredSessions();
    }, 60000); // Every minute
  }

  stopRealTimeProcessing() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  async performPeriodicChecks() {
    // Check session timeouts
    for (const [sessionKey, session] of this.activeSessions) {
      const lastActivity = new Date(session.lastActivity);
      const timeout = 30 * 60 * 1000; // 30 minutes
      
      if (Date.now() - lastActivity.getTime() > timeout) {
        this.activeSessions.delete(sessionKey);
        await this.expireSession(session.sessionId);
      }
    }

    // Check for system-wide anomalies
    await this.checkSystemWideAnomalies();
  }

  async cleanupExpiredSessions() {
    const expiredSessions = [];
    
    for (const [sessionKey, session] of this.activeSessions) {
      const lastActivity = new Date(session.lastActivity);
      if (Date.now() - lastActivity.getTime() > (60 * 60 * 1000)) { // 1 hour
        expiredSessions.push(sessionKey);
      }
    }

    expiredSessions.forEach(sessionKey => {
      this.activeSessions.delete(sessionKey);
    });
  }

  // Utility Methods
  hashIP(ip) {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(ip).digest('hex');
  }

  sanitizeUserAgent(agent) {
    return agent?.substring(0, 255) || 'unknown';
  }

  isOffHours(timestamp) {
    const hour = new Date(timestamp).getHours();
    return hour < 6 || hour > 22;
  }

  calculateStandardDeviation(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  calculateBaselineActions(activities) {
    const actions = {};
    activities.forEach(activity => {
      actions[activity.action] = (actions[activity.action] || 0) + 1;
    });
    return actions;
  }

  // Placeholder methods for database operations
  async getRecentFailures(userId, minutes) {
    return await require('../database/user-database').getRecentFailures(userId, minutes);
  }

  async getRecentOffHoursAccess(userId, minutes) {
    return await require('../database/user-database').getRecentOffHoursAccess(userId, minutes);
  }

  async getRecentExports(userId, minutes) {
    return await require('../database/user-database').getRecentExports(userId, minutes);
  }

  async expireSession(sessionId) {
    await require('../database/user-database').expireSession(sessionId);
  }

  async checkSystemWideAnomalies() {
    const activeUsers = this.activeSessions.size;
    if (activeUsers > 1000) {
      this.emit('highLoadDetected', { activeUsers });
    }
  }

  async getProcessingQueueSize() {
    return this.eventQueue?.length || 0;
  }

  async getAnomalyDetectionRate() {
    // Calculate recent anomaly detection rate
    return 0.05; // Placeholder
  }

  async getFalsePositiveRate() {
    return 0.02; // Placeholder
  }

  async getActiveAlerts() {
    return Array.from(this.thresholdAlerts.values());
  }

  // Model creation placeholders
  createIsolationForestModel() {
    return { type: 'isolation_forest', accuracy: 0.85 };
  }

  createStatisticalModel() {
    return { type: 'statistical', accuracy: 0.78 };
  }

  createMLAnomalyModel() {
    return { type: 'machine_learning', accuracy: 0.82 };
  }

  setupActivityTracking() {
    // Setup real-time activity tracking infrastructure
    console.log('Setting up activity tracking infrastructure');
  }

  initializeAnomalyDetection() {
    // Initialize anomaly detection models
    console.log('Initializing anomaly detection models');
  }

  updateRealTimeMetrics(event) {
    // Update real-time monitoring metrics
    this.emit('metricsUpdate', {
      timestamp: new Date().toISOString(),
      eventType: event.event,
      processed: true
    });
  }

  checkImmediateThreats(event) {
    // Check for immediate security threats
    const criticalEvents = ['security.breach', 'system.compromise', 'data.exfiltration'];
    if (criticalEvents.some(critical => event.event?.includes(critical))) {
      this.emit('immediateThreat', event);
    }
  }
}

module.exports = RealTimeMonitor;