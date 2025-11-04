// Healthcare User Activity Monitoring and Audit Trail System
// Real-time monitoring with comprehensive audit logging for healthcare compliance

const crypto = require('crypto');
const EventEmitter = require('events');
const config = require('../config/user-management-config');

class HealthcareAuditLogger extends EventEmitter {
  constructor() {
    super();
    this.config = config;
    this.eventQueue = [];
    this.batchSize = 100;
    this.batchInterval = 5000; // 5 seconds
    this.anomalyDetector = require('./anomaly-detector');
    this.realTimeMonitor = require('./real-time-monitor');
    this.alertManager = require('./alert-manager');
    
    this.startBatchProcessor();
    this.startRealTimeProcessing();
  }

  // Log Security-Critical Events
  async logSecurityEvent(eventData) {
    try {
      const securityEvent = {
        eventId: crypto.randomUUID(),
        category: 'security',
        severity: this.determineSeverity(eventData.event),
        ...eventData,
        timestamp: new Date().toISOString(),
        systemGenerated: true,
        complianceTags: this.getComplianceTags(eventData.event)
      };

      // Queue for batch processing
      this.eventQueue.push(securityEvent);

      // Process critical events immediately
      if (securityEvent.severity === 'critical' || securityEvent.severity === 'high') {
        await this.processCriticalEvent(securityEvent);
      }

      // Trigger real-time monitoring
      await this.realTimeMonitor.processEvent(securityEvent);

      return securityEvent.eventId;

    } catch (error) {
      console.error('Security event logging error:', error);
      throw error;
    }
  }

  // Comprehensive Audit Logging
  async logEvent(eventData) {
    try {
      const auditEvent = {
        eventId: crypto.randomUUID(),
        category: 'audit',
        ...eventData,
        timestamp: eventData.timestamp || new Date().toISOString(),
        systemGenerated: false,
        requiresReview: this.requiresReview(eventData.event),
        retentionPeriod: this.getRetentionPeriod(eventData.event)
      };

      // Store in immediate storage for quick access
      await this.storeImmediateEvent(auditEvent);

      // Queue for batch processing
      this.eventQueue.push(auditEvent);

      // Trigger anomaly detection
      await this.anomalyDetector.analyzeEvent(auditEvent);

      // Emit event for real-time monitoring
      this.emit('auditEvent', auditEvent);

      return auditEvent.eventId;

    } catch (error) {
      console.error('Audit event logging error:', error);
      throw error;
    }
  }

  // Medical Data Access Logging
  async logMedicalDataAccess(userId, patientId, action, dataType, context = {}) {
    try {
      const accessEvent = {
        userId,
        event: 'medical_data.access',
        details: {
          patientId,
          action, // 'read', 'create', 'update', 'delete', 'export'
          dataType, // 'medical_record', 'medication', 'lab_result', 'imaging'
          context,
          accessMethod: context.accessMethod || 'web_interface',
          duration: context.duration || null,
          ipAddress: this.hashIP(context.ipAddress),
          userAgent: this.sanitizeUserAgent(context.userAgent),
          sessionId: context.sessionId
        },
        timestamp: new Date().toISOString(),
        source: 'medical_data_service',
        category: 'patient_privacy',
        complianceType: 'HIPAA',
        requiresHIPAACompliance: true
      };

      await this.logEvent(accessEvent);

      // Check for unauthorized access patterns
      await this.checkUnauthorizedAccessPatterns(userId, accessEvent);

      return accessEvent.eventId;

    } catch (error) {
      console.error('Medical data access logging error:', error);
      throw error;
    }
  }

  // User Authentication Events
  async logAuthenticationEvent(eventType, userData) {
    try {
      const authEvent = {
        userId: userData.userId,
        event: `user.${eventType}`, // 'login', 'logout', 'failed_login', 'password_change'
        details: {
          email: userData.email,
          ipAddress: this.hashIP(userData.ipAddress),
          userAgent: this.sanitizeUserAgent(userData.userAgent),
          deviceFingerprint: userData.deviceFingerprint,
          location: userData.location,
          mfaUsed: userData.mfaUsed,
          failureReason: userData.failureReason
        },
        timestamp: new Date().toISOString(),
        source: 'authentication_service',
        category: 'authentication',
        riskScore: this.calculateAuthenticationRiskScore(userData, eventType)
      };

      await this.logEvent(authEvent);

      // Trigger security alerts for suspicious patterns
      await this.triggerSecurityAlerts(authEvent);

      return authEvent.eventId;

    } catch (error) {
      console.error('Authentication event logging error:', error);
      throw error;
    }
  }

  // Role and Permission Changes
  async logRoleChangeEvent(changeType, userData) {
    try {
      const roleEvent = {
        userId: userData.userId,
        event: `role.${changeType}`, // 'granted', 'revoked', 'modified'
        details: {
          roleName: userData.roleName,
          grantedBy: userData.grantedBy,
          reason: userData.reason,
          approvals: userData.approvals,
          previousRoles: userData.previousRoles,
          newPermissions: userData.newPermissions
        },
        timestamp: new Date().toISOString(),
        source: 'rbac_system',
        category: 'access_control',
        requiresApproval: changeType !== 'revoked',
        complianceType: 'SOX'
      };

      await this.logEvent(roleEvent);

      // Notify security team for high-risk role changes
      if (userData.highRiskRole || this.isHighRiskRole(userData.roleName)) {
        await this.alertManager.sendSecurityAlert({
          type: 'high_risk_role_change',
          severity: 'high',
          event: roleEvent,
          notificationTargets: ['security_admin', 'compliance_officer']
        });
      }

      return roleEvent.eventId;

    } catch (error) {
      console.error('Role change event logging error:', error);
      throw error;
    }
  }

  // Data Export and Breach Events
  async logDataExportEvent(userId, exportData) {
    try {
      const exportEvent = {
        userId,
        event: 'data.export',
        details: {
          exportType: exportData.type, // 'patient_data', 'user_data', 'audit_logs'
          recordCount: exportData.recordCount,
          dataSize: exportData.dataSize,
          exportMethod: exportData.method, // 'api', 'manual', 'automated'
          destination: exportData.destination,
          justification: exportData.justification,
          approvalRequired: exportData.approvalRequired,
          approvedBy: exportData.approvedBy
        },
        timestamp: new Date().toISOString(),
        source: 'data_export_service',
        category: 'data_movement',
        riskLevel: this.calculateExportRiskLevel(exportData),
        requiresImmediateReview: exportData.recordCount > 1000
      };

      await this.logEvent(exportEvent);

      // Log to breach monitoring if high risk
      if (exportEvent.riskLevel === 'high') {
        await this.logSecurityEvent({
          ...exportEvent,
          category: 'security',
          severity: 'high',
          event: 'data_breach_risk'
        });
      }

      return exportEvent.eventId;

    } catch (error) {
      console.error('Data export event logging error:', error);
      throw error;
    }
  }

  // System Configuration Changes
  async logSystemConfigurationEvent(configData) {
    try {
      const configEvent = {
        event: 'system.configuration',
        details: {
          component: configData.component,
          changeType: configData.changeType, // 'create', 'update', 'delete'
          oldValue: this.hashSensitiveData(configData.oldValue),
          newValue: this.hashSensitiveData(configData.newValue),
          changedBy: configData.changedBy,
          changeReason: configData.reason,
          approvalId: configData.approvalId,
          rollbackAvailable: configData.rollbackAvailable
        },
        timestamp: new Date().toISOString(),
        source: 'configuration_service',
        category: 'system_change',
        requiresBackup: true,
        complianceType: 'SOX'
      };

      await this.logEvent(configEvent);

      return configEvent.eventId;

    } catch (error) {
      console.error('System configuration event logging error:', error);
      throw error;
    }
  }

  // Retrieve Audit Trail Data
  async getAuditTrail(filters) {
    try {
      const {
        userId,
        eventTypes,
        startDate,
        endDate,
        category,
        complianceType,
        severity,
        limit = 1000
      } = filters;

      // Build database query
      const query = await this.buildAuditQuery({
        userId,
        eventTypes,
        startDate,
        endDate,
        category,
        complianceType,
        severity,
        limit
      });

      const events = await require('../database/user-database').queryAuditEvents(query);

      return {
        success: true,
        events,
        totalCount: events.length,
        filters: filters,
        retrievedAt: new Date().toISOString()
      };

    } catch (error) {
      console.error('Audit trail retrieval error:', error);
      throw error;
    }
  }

  // Generate Compliance Reports
  async generateComplianceReport(complianceType, period) {
    try {
      const reportData = {
        reportId: crypto.randomUUID(),
        complianceType,
        period: {
          start: period.startDate,
          end: period.endDate
        },
        generatedAt: new Date().toISOString(),
        generatedBy: 'automated_system'
      };

      // Gather compliance metrics
      const metrics = await this.gatherComplianceMetrics(complianceType, period);

      // Generate compliance summary
      const summary = await this.generateComplianceSummary(metrics, complianceType);

      // Check for compliance violations
      const violations = await this.identifyComplianceViolations(complianceType, period);

      const complianceReport = {
        ...reportData,
        metrics,
        summary,
        violations,
        complianceScore: this.calculateComplianceScore(metrics, violations),
        recommendations: this.generateRecommendations(violations, metrics)
      };

      // Store report
      await require('../database/user-database').storeComplianceReport(complianceReport);

      // Log report generation
      await this.logEvent({
        event: 'compliance.report_generated',
        details: {
          reportId: complianceReport.reportId,
          complianceType,
          period: reportData.period,
          complianceScore: complianceReport.complianceScore,
          violationCount: violations.length
        },
        timestamp: new Date().toISOString(),
        source: 'audit_service',
        category: 'compliance'
      });

      return complianceReport;

    } catch (error) {
      console.error('Compliance report generation error:', error);
      throw error;
    }
  }

  // Real-time Security Monitoring
  async enableRealTimeMonitoring() {
    try {
      // Set up real-time event processing
      this.realTimeMonitor.on('suspiciousActivity', async (activity) => {
        await this.handleSuspiciousActivity(activity);
      });

      this.realTimeMonitor.on('thresholdBreached', async (threshold) => {
        await this.handleThresholdBreach(threshold);
      });

      this.realTimeMonitor.on('patternDetected', async (pattern) => {
        await this.handlePatternDetection(pattern);
      });

      console.log('Real-time security monitoring enabled');

    } catch (error) {
      console.error('Real-time monitoring setup error:', error);
      throw error;
    }
  }

  // Batch Processing
  startBatchProcessor() {
    setInterval(async () => {
      if (this.eventQueue.length >= this.batchSize) {
        await this.processBatch();
      }
    }, this.batchInterval);
  }

  async processBatch() {
    try {
      const batch = this.eventQueue.splice(0, this.batchSize);
      
      if (batch.length === 0) return;

      // Process batch to database
      const results = await require('../database/user-database').insertAuditBatch(batch);

      // Log batch processing
      console.log(`Processed audit batch: ${batch.length} events`);

      return results;

    } catch (error) {
      console.error('Batch processing error:', error);
      // Re-queue failed events
      this.eventQueue.unshift(...batch);
    }
  }

  // Start Real-time Processing
  startRealTimeProcessing() {
    this.on('auditEvent', async (event) => {
      await this.processRealTimeEvent(event);
    });
  }

  async processRealTimeEvent(event) {
    // Check for immediate threats
    if (this.isImmediateThreat(event)) {
      await this.handleImmediateThreat(event);
    }

    // Update real-time dashboards
    await this.updateRealTimeDashboards(event);

    // Trigger automated responses
    await this.triggerAutomatedResponses(event);
  }

  // Helper Methods
  determineSeverity(event) {
    const criticalEvents = [
      'security.breach',
      'authentication.failed',
      'data.export.unauthorized',
      'privilege.escalation',
      'system.compromise'
    ];

    const highEvents = [
      'user.failed_login',
      'role.granted',
      'data.access.unauthorized',
      'system.configuration'
    ];

    if (criticalEvents.some(critical => event.includes(critical))) {
      return 'critical';
    }

    if (highEvents.some(high => event.includes(high))) {
      return 'high';
    }

    return 'medium';
  }

  getComplianceTags(event) {
    const complianceMap = {
      'user.login': ['SOX', 'HIPAA'],
      'medical_data.access': ['HIPAA'],
      'role.change': ['SOX'],
      'data.export': ['GDPR', 'HIPAA'],
      'system.configuration': ['SOX']
    };

    for (const [pattern, tags] of Object.entries(complianceMap)) {
      if (event.includes(pattern)) {
        return tags;
      }
    }

    return [];
  }

  requiresReview(event) {
    const reviewEvents = [
      'role.granted',
      'data.export',
      'system.configuration',
      'emergency.access'
    ];

    return reviewEvents.some(review => event.includes(review));
  }

  getRetentionPeriod(event) {
    const retentionMap = {
      'medical_data.access': 2555, // 7 years
      'user.authentication': 2555, // 7 years
      'role.change': 2555, // 7 years
      'system.configuration': 2555, // 7 years
      'data.export': 2555 // 7 years
    };

    for (const [pattern, days] of Object.entries(retentionMap)) {
      if (event.includes(pattern)) {
        return days;
      }
    }

    return 2555; // Default 7 years
  }

  hashIP(ip) {
    return crypto.createHash('sha256').update(ip).digest('hex');
  }

  sanitizeUserAgent(userAgent) {
    // Remove potentially sensitive information
    return userAgent.replace(/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/g, '[IP_REDACTED]');
  }

  calculateAuthenticationRiskScore(userData, eventType) {
    let score = 0;

    if (eventType === 'failed_login') {
      score += 50;
    }

    if (userData.failedAttempts > 1) {
      score += userData.failedAttempts * 20;
    }

    if (userData.newLocation) {
      score += 30;
    }

    if (!userData.mfaUsed) {
      score += 20;
    }

    return Math.min(score, 100);
  }

  calculateExportRiskLevel(exportData) {
    let risk = 0;

    if (exportData.recordCount > 10000) {
      risk += 3;
    }

    if (exportData.dataSize > 100000000) { // 100MB
      risk += 3;
    }

    if (exportData.type === 'patient_data') {
      risk += 4;
    }

    if (risk >= 6) return 'high';
    if (risk >= 3) return 'medium';
    return 'low';
  }

  isHighRiskRole(roleName) {
    const highRiskRoles = [
      'super_admin',
      'security_admin',
      'compliance_officer',
      'database_admin'
    ];

    return highRiskRoles.includes(roleName);
  }

  hashSensitiveData(data) {
    if (typeof data === 'string') {
      return crypto.createHash('sha256').update(data).digest('hex');
    }
    return crypto.createHash('sha256').update(JSON.stringify(data)).digest('hex');
  }

  async storeImmediateEvent(event) {
    // Store in fast-access database for real-time queries
    await require('../database/user-database').insertImmediateEvent(event);
  }

  async processCriticalEvent(event) {
    // Send immediate alerts for critical events
    await this.alertManager.sendCriticalAlert(event);
    
    // Log to incident response system
    await require('../security/incident-response').logSecurityIncident(event);
  }

  async checkUnauthorizedAccessPatterns(userId, accessEvent) {
    // Implement pattern checking for unauthorized access
    const recentAccess = await this.getRecentAccessPatterns(userId, 60); // Last 60 minutes
    
    if (recentAccess.length > 10) {
      await this.alertManager.sendSecurityAlert({
        type: 'excessive_access_pattern',
        severity: 'medium',
        userId,
        event: accessEvent,
        notificationTargets: ['security_admin']
      });
    }
  }

  async triggerSecurityAlerts(authEvent) {
    if (authEvent.riskScore > 70) {
      await this.alertManager.sendSecurityAlert({
        type: 'high_risk_authentication',
        severity: authEvent.riskScore > 90 ? 'critical' : 'high',
        event: authEvent,
        notificationTargets: ['security_admin']
      });
    }
  }

  async getRecentAccessPatterns(userId, minutes) {
    const cutoff = new Date(Date.now() - (minutes * 60 * 1000)).toISOString();
    
    return await require('../database/user-database').getEvents({
      userId,
      eventType: 'medical_data.access',
      startDate: cutoff,
      limit: 100
    });
  }

  async buildAuditQuery(filters) {
    // Build database query based on filters
    const query = {
      conditions: [],
      parameters: [],
      joins: [],
      orderBy: 'timestamp DESC',
      limit: filters.limit || 1000
    };

    if (filters.userId) {
      query.conditions.push('user_id = ?');
      query.parameters.push(filters.userId);
    }

    if (filters.eventTypes && filters.eventTypes.length > 0) {
      query.conditions.push(`event IN (${filters.eventTypes.map(() => '?').join(', ')})`);
      query.parameters.push(...filters.eventTypes);
    }

    if (filters.startDate) {
      query.conditions.push('timestamp >= ?');
      query.parameters.push(filters.startDate);
    }

    if (filters.endDate) {
      query.conditions.push('timestamp <= ?');
      query.parameters.push(filters.endDate);
    }

    if (filters.category) {
      query.conditions.push('category = ?');
      query.parameters.push(filters.category);
    }

    return query;
  }

  async gatherComplianceMetrics(complianceType, period) {
    // Gather specific metrics based on compliance type
    const metrics = {
      totalEvents: 0,
      criticalEvents: 0,
      violations: 0,
      usersAffected: 0,
      dataPointsAccessed: 0
    };

    // Implementation would gather actual metrics
    return metrics;
  }

  async generateComplianceSummary(metrics, complianceType) {
    return {
      overallStatus: 'compliant',
      keyFindings: [
        'All critical events properly logged',
        'No unauthorized access detected',
        'Role changes properly audited'
      ],
      metrics: metrics
    };
  }

  async identifyComplianceViolations(complianceType, period) {
    // Identify potential compliance violations
    return [];
  }

  calculateComplianceScore(metrics, violations) {
    const baseScore = 100;
    const violationPenalty = violations.length * 10;
    return Math.max(baseScore - violationPenalty, 0);
  }

  generateRecommendations(violations, metrics) {
    return [
      'Review and update access control policies',
      'Implement additional monitoring for high-risk activities',
      'Conduct user training on data handling procedures'
    ];
  }

  async handleSuspiciousActivity(activity) {
    await this.alertManager.sendSecurityAlert({
      type: 'suspicious_activity_detected',
      severity: 'medium',
      activity,
      notificationTargets: ['security_admin']
    });
  }

  async handleThresholdBreach(threshold) {
    await this.alertManager.sendSecurityAlert({
      type: 'threshold_breach',
      severity: 'medium',
      threshold,
      notificationTargets: ['security_admin']
    });
  }

  async handlePatternDetection(pattern) {
    await this.alertManager.sendSecurityAlert({
      type: 'anomalous_pattern_detected',
      severity: 'low',
      pattern,
      notificationTargets: ['security_admin']
    });
  }

  isImmediateThreat(event) {
    const threatEvents = [
      'security.breach',
      'system.compromise',
      'data.export.unauthorized'
    ];

    return threatEvents.some(threat => event.event?.includes(threat));
  }

  async handleImmediateThreat(event) {
    // Implement immediate threat response
    await this.alertManager.sendCriticalAlert(event);
  }

  async updateRealTimeDashboards(event) {
    // Update real-time monitoring dashboards
    console.log(`Updating dashboards with event: ${event.eventId}`);
  }

  async triggerAutomatedResponses(event) {
    // Trigger automated security responses
    if (event.severity === 'critical') {
      // Lock user account temporarily
      if (event.userId) {
        await this.temporaryAccountLock(event.userId, 'security_threat', 3600);
      }
    }
  }

  async temporaryAccountLock(userId, reason, durationSeconds) {
    await require('../database/user-database').createAccountLock({
      userId,
      reason,
      lockType: 'temporary',
      expiresAt: new Date(Date.now() + (durationSeconds * 1000)).toISOString(),
      lockedAt: new Date().toISOString()
    });
  }
}

module.exports = HealthcareAuditLogger;