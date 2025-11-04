// Healthcare Security Alert Management System
// Production-grade alerting system for user management security events

const crypto = require('crypto');
const EventEmitter = require('events');
const config = require('../config/user-management-config');

class SecurityAlertManager extends EventEmitter {
  constructor() {
    super();
    this.config = config;
    this.activeAlerts = new Map();
    this.alertRules = this.initializeAlertRules();
    this.alertEscalation = this.initializeEscalationRules();
    this.notificationChannels = this.initializeNotificationChannels();
    this.alertHistory = new Map();
    this.suppressionRules = new Map();
  }

  // Alert Rules Configuration
  initializeAlertRules() {
    return {
      authentication_failures: {
        condition: 'failed_login_count > 5',
        threshold: 5,
        timeWindow: 900, // 15 minutes
        severity: 'high',
        autoBlock: true,
        escalation: true
      },
      privilege_escalation: {
        condition: 'role_change && !approved',
        threshold: 1,
        timeWindow: 60,
        severity: 'critical',
        autoBlock: false,
        escalation: true
      },
      data_exfiltration: {
        condition: 'export_size > threshold',
        threshold: 10000000, // 10MB
        timeWindow: 300,
        severity: 'critical',
        autoBlock: true,
        escalation: true
      },
      suspicious_ip: {
        condition: 'ip_reputation == "malicious"',
        threshold: 1,
        timeWindow: 60,
        severity: 'high',
        autoBlock: true,
        escalation: false
      },
      off_hours_mass_access: {
        condition: 'access_count > 50 && off_hours',
        threshold: 50,
        timeWindow: 1800,
        severity: 'medium',
        autoBlock: false,
        escalation: false
      },
      compliance_violation: {
        condition: 'violation_type == "hipaa" || violation_type == "gdpr"',
        threshold: 1,
        timeWindow: 60,
        severity: 'high',
        autoBlock: false,
        escalation: true
      }
    };
  }

  // Escalation Rules
  initializeEscalationRules() {
    return {
      level_1: {
        name: 'Technical Support',
        responseTime: 300, // 5 minutes
        notify: ['support_team'],
        conditions: ['medium', 'high']
      },
      level_2: {
        name: 'Security Team',
        responseTime: 180, // 3 minutes
        notify: ['security_admin', 'security_team'],
        conditions: ['high', 'critical']
      },
      level_3: {
        name: 'Executive Team',
        responseTime: 60, // 1 minute
        notify: ['security_admin', 'compliance_officer', 'hospital_admin'],
        conditions: ['critical']
      },
      emergency: {
        name: 'Emergency Response',
        responseTime: 30, // 30 seconds
        notify: ['medical_emergency_team', 'security_admin', 'compliance_officer'],
        conditions: ['critical'],
        always: ['medical_emergency', 'data_breach']
      }
    };
  }

  // Notification Channels
  initializeNotificationChannels() {
    return {
      email: {
        enabled: true,
        templates: 'healthcare_security',
        priority: true
      },
      sms: {
        enabled: true,
        templates: 'security_urgent',
        priority: true
      },
      push: {
        enabled: true,
        templates: 'security_alert',
        priority: true
      },
      slack: {
        enabled: true,
        webhook: process.env.SLACK_SECURITY_WEBHOOK,
        priority: true
      },
      teams: {
        enabled: true,
        webhook: process.env.TEAMS_SECURITY_WEBHOOK,
        priority: true
      }
    };
  }

  // Process Security Event
  async processSecurityEvent(event) {
    try {
      const alertId = crypto.randomUUID();
      const matchingRules = this.findMatchingRules(event);
      
      if (matchingRules.length === 0) {
        return null; // No matching rules
      }

      const alerts = [];
      
      for (const rule of matchingRules) {
        const alert = await this.createAlert(alertId, event, rule);
        alerts.push(alert);
        
        // Process auto-blocking if configured
        if (rule.autoBlock) {
          await this.processAutoBlock(alert);
        }
        
        // Process escalation if configured
        if (rule.escalation) {
          await this.processEscalation(alert);
        }
      }

      // Store alerts
      for (const alert of alerts) {
        await this.storeAlert(alert);
        this.activeAlerts.set(alert.alertId, alert);
      }

      // Send notifications
      await this.sendAlertNotifications(alerts);

      return alerts;

    } catch (error) {
      console.error('Security event processing error:', error);
      throw error;
    }
  }

  // Create Alert
  async createAlert(alertId, event, rule) {
    const severity = rule.severity || 'medium';
    
    const alert = {
      alertId,
      event,
      rule: rule.name,
      severity,
      title: this.generateAlertTitle(event, rule),
      message: this.generateAlertMessage(event, rule),
      timestamp: new Date().toISOString(),
      status: 'active',
      assignedTo: null,
      resolution: null,
      escalationLevel: this.determineInitialEscalationLevel(severity),
      metadata: {
        eventType: event.event,
        userId: event.userId,
        sourceIP: event.details?.ipAddress,
        riskScore: event.details?.riskScore,
        complianceType: event.complianceType || []
      },
      actions: this.generateAlertActions(rule, severity),
      context: await this.gatherAlertContext(event)
    };

    return alert;
  }

  // Find Matching Alert Rules
  findMatchingRules(event) {
    const matchingRules = [];
    
    for (const [ruleName, rule] of Object.entries(this.alertRules)) {
      if (this.evaluateRuleCondition(event, rule)) {
        rule.name = ruleName;
        matchingRules.push(rule);
      }
    }
    
    return matchingRules;
  }

  // Evaluate Rule Conditions
  evaluateRuleCondition(event, rule) {
    const condition = rule.condition;
    
    try {
      // Simple condition evaluation (in production, use a proper rule engine)
      if (condition.includes('failed_login_count')) {
        const threshold = rule.threshold;
        // Check recent failed logins
        return event.details?.failedLoginCount >= threshold;
      }
      
      if (condition.includes('role_change')) {
        return event.event?.includes('role') && event.event?.includes('change');
      }
      
      if (condition.includes('export_size')) {
        const threshold = rule.threshold;
        return event.details?.dataSize >= threshold;
      }
      
      if (condition.includes('ip_reputation')) {
        return event.details?.ipReputation === 'malicious';
      }
      
      if (condition.includes('off_hours')) {
        const hour = new Date(event.timestamp).getHours();
        return hour < 6 || hour > 22;
      }
      
      if (condition.includes('access_count')) {
        const threshold = rule.threshold;
        return event.details?.accessCount >= threshold;
      }
      
      if (condition.includes('violation_type')) {
        const violationTypes = ['hipaa', 'gdpr'];
        return violationTypes.some(type => 
          event.event?.includes(type) || event.details?.violationType === type
        );
      }
      
      return false;
      
    } catch (error) {
      console.error('Rule condition evaluation error:', error);
      return false;
    }
  }

  // Auto-blocking Logic
  async processAutoBlock(alert) {
    try {
      const { userId, sourceIP } = alert.metadata;
      
      if (userId && alert.rule.includes('authentication')) {
        await this.blockUser(userId, 'security_alert', 3600); // 1 hour block
        alert.actions.push('user_blocked');
      }
      
      if (sourceIP && alert.rule.includes('suspicious_ip')) {
        await this.blockIP(sourceIP, 'security_alert', 3600);
        alert.actions.push('ip_blocked');
      }
      
      if (userId && alert.rule.includes('data_exfiltration')) {
        await this.suspendUserAccount(userId, 'potential_data_breach');
        alert.actions.push('account_suspended');
      }
      
      alert.autoBlockExecuted = true;
      alert.autoBlockTimestamp = new Date().toISOString();
      
    } catch (error) {
      console.error('Auto-blocking error:', error);
      alert.autoBlockError = error.message;
    }
  }

  // Escalation Processing
  async processEscalation(alert) {
    try {
      const escalationLevel = this.determineEscalationLevel(alert);
      const escalationRule = this.alertEscalation[escalationLevel];
      
      if (escalationRule) {
        alert.escalationLevel = escalationLevel;
        alert.escalatedAt = new Date().toISOString();
        alert.escalationDeadline = new Date(
          Date.now() + (escalationRule.responseTime * 1000)
        ).toISOString();
        
        // Notify escalation recipients
        for (const recipient of escalationRule.notify) {
          await this.notifyEscalationRecipient(recipient, alert, escalationRule);
        }
        
        // Set escalation timeout
        this.setEscalationTimeout(alert, escalationRule.responseTime * 1000);
      }
      
    } catch (error) {
      console.error('Escalation processing error:', error);
    }
  }

  // Alert Notifications
  async sendAlertNotifications(alerts) {
    for (const alert of alerts) {
      try {
        // Send immediate notifications for critical alerts
        if (alert.severity === 'critical') {
          await this.sendCriticalAlertNotifications(alert);
        } else {
          await this.sendStandardAlertNotifications(alert);
        }
        
        // Check for alert suppression
        if (await this.shouldSuppressAlert(alert)) {
          alert.suppressed = true;
          alert.suppressionReason = await this.getSuppressionReason(alert);
        }
        
      } catch (error) {
        console.error('Alert notification error:', error);
        alert.notificationError = error.message;
      }
    }
  }

  // Critical Alert Notifications
  async sendCriticalAlertNotifications(alert) {
    const channels = ['email', 'sms', 'push', 'slack', 'teams'];
    
    for (const channel of channels) {
      if (this.notificationChannels[channel]?.enabled) {
        try {
          await this.sendToChannel(channel, alert, 'critical');
        } catch (error) {
          console.error(`Failed to send ${channel} notification:`, error);
        }
      }
    }
  }

  // Standard Alert Notifications
  async sendStandardAlertNotifications(alert) {
    const channels = ['email', 'push', 'slack'];
    
    for (const channel of channels) {
      if (this.notificationChannels[channel]?.enabled) {
        try {
          await this.sendToChannel(channel, alert, alert.severity);
        } catch (error) {
          console.error(`Failed to send ${channel} notification:`, error);
        }
      }
    }
  }

  // Send to Specific Channel
  async sendToChannel(channel, alert, priority) {
    const channelConfig = this.notificationChannels[channel];
    
    switch (channel) {
      case 'email':
        await this.sendEmailAlert(alert, channelConfig);
        break;
      case 'sms':
        await this.sendSMSAlert(alert, channelConfig);
        break;
      case 'push':
        await this.sendPushAlert(alert, channelConfig);
        break;
      case 'slack':
        await this.sendSlackAlert(alert, channelConfig);
        break;
      case 'teams':
        await this.sendTeamsAlert(alert, channelConfig);
        break;
    }
  }

  // Alert Management
  async acknowledgeAlert(alertId, acknowledgedBy, notes = '') {
    const alert = this.activeAlerts.get(alertId);
    
    if (!alert) {
      throw new Error('Alert not found or already resolved');
    }
    
    alert.status = 'acknowledged';
    alert.acknowledgedBy = acknowledgedBy;
    alert.acknowledgedAt = new Date().toISOString();
    alert.notes = notes;
    
    // Update escalation if acknowledged
    if (alert.escalationDeadline) {
      clearTimeout(alert.escalationTimeout);
    }
    
    await this.storeAlert(alert);
    
    return alert;
  }

  async resolveAlert(alertId, resolvedBy, resolution = '') {
    const alert = this.activeAlerts.get(alertId);
    
    if (!alert) {
      throw new Error('Alert not found');
    }
    
    alert.status = 'resolved';
    alert.resolvedBy = resolvedBy;
    alert.resolvedAt = new Date().toISOString();
    alert.resolution = resolution;
    
    // Remove from active alerts
    this.activeAlerts.delete(alertId);
    
    // Move to history
    this.alertHistory.set(alertId, alert);
    
    await this.storeAlert(alert);
    
    return alert;
  }

  async escalateAlert(alertId, escalatedBy, escalationLevel, reason = '') {
    const alert = this.activeAlerts.get(alertId);
    
    if (!alert) {
      throw new Error('Alert not found');
    }
    
    const previousLevel = alert.escalationLevel;
    alert.escalationLevel = escalationLevel;
    alert.escalatedBy = escalatedBy;
    alert.escalatedAt = new Date().toISOString();
    alert.escalationReason = reason;
    
    // Clear previous timeout
    if (alert.escalationTimeout) {
      clearTimeout(alert.escalationTimeout);
    }
    
    // Set new escalation
    const escalationRule = this.alertEscalation[escalationLevel];
    if (escalationRule) {
      alert.escalationDeadline = new Date(
        Date.now() + (escalationRule.responseTime * 1000)
      ).toISOString();
      
      this.setEscalationTimeout(alert, escalationRule.responseTime * 1000);
      
      // Notify new escalation recipients
      for (const recipient of escalationRule.notify) {
        await this.notifyEscalationRecipient(recipient, alert, escalationRule);
      }
    }
    
    await this.storeAlert(alert);
    
    return alert;
  }

  // Alert Analytics
  async getAlertMetrics(timeframe) {
    const metrics = {
      timeframe,
      generatedAt: new Date().toISOString(),
      summary: {
        totalAlerts: 0,
        criticalAlerts: 0,
        highAlerts: 0,
        mediumAlerts: 0,
        lowAlerts: 0,
        resolvedAlerts: 0,
        escalationRate: 0,
        autoBlockRate: 0
      },
      topAlertTypes: {},
      topUsers: {},
      resolutionTimes: {},
      falsePositiveRate: 0
    };
    
    const alerts = await this.getAlertsInTimeframe(timeframe);
    
    // Calculate summary metrics
    metrics.summary.totalAlerts = alerts.length;
    alerts.forEach(alert => {
      metrics.summary[`${alert.severity}Alerts`]++;
      
      if (alert.status === 'resolved') {
        metrics.summary.resolvedAlerts++;
      }
    });
    
    metrics.summary.escalationRate = alerts.filter(a => a.escalatedAt).length / alerts.length;
    metrics.summary.autoBlockRate = alerts.filter(a => a.autoBlockExecuted).length / alerts.length;
    
    // Top alert types
    alerts.forEach(alert => {
      metrics.topAlertTypes[alert.rule] = (metrics.topAlertTypes[alert.rule] || 0) + 1;
    });
    
    // Resolution times
    const resolvedAlerts = alerts.filter(a => a.resolvedAt);
    const avgResolutionTime = resolvedAlerts.reduce((sum, alert) => {
      const resolutionTime = new Date(alert.resolvedAt).getTime() - new Date(alert.timestamp).getTime();
      return sum + resolutionTime;
    }, 0) / resolvedAlerts.length;
    
    metrics.resolutionTimes.average = Math.round(avgResolutionTime / 1000 / 60); // minutes
    
    return metrics;
  }

  // Helper Methods
  generateAlertTitle(event, rule) {
    const eventType = event.event || 'Unknown Event';
    const severity = rule.severity || 'medium';
    return `[${severity.toUpperCase()}] Security Alert: ${eventType}`;
  }

  generateAlertMessage(event, rule) {
    const userId = event.userId || 'Unknown';
    const sourceIP = event.details?.ipAddress || 'Unknown';
    const timestamp = new Date(event.timestamp).toISOString();
    
    return `
Security Alert: ${rule.name}
Severity: ${rule.severity}
User: ${userId}
Source IP: ${sourceIP}
Time: ${timestamp}
Event: ${event.event}
Rule: ${rule.condition}
    `.trim();
  }

  determineInitialEscalationLevel(severity) {
    switch (severity) {
      case 'critical': return 'level_3';
      case 'high': return 'level_2';
      case 'medium': return 'level_1';
      default: return 'level_1';
    }
  }

  generateAlertActions(rule, severity) {
    const actions = ['investigate', 'monitor'];
    
    if (rule.autoBlock) {
      actions.push('auto_block');
    }
    
    if (rule.escalation) {
      actions.push('escalate');
    }
    
    if (severity === 'critical') {
      actions.push('immediate_response', 'incident_response');
    }
    
    return actions;
  }

  determineEscalationLevel(alert) {
    // Check if it's an always-emergency situation
    const emergencyTriggers = ['medical_emergency', 'data_breach'];
    const eventType = alert.metadata.eventType || '';
    
    if (emergencyTriggers.some(trigger => eventType.includes(trigger))) {
      return 'emergency';
    }
    
    return alert.escalationLevel || this.determineInitialEscalationLevel(alert.severity);
  }

  async gatherAlertContext(event) {
    return {
      systemState: await this.getSystemState(),
      relatedAlerts: await this.getRelatedAlerts(event.userId, event.timestamp),
      userProfile: await this.getUserProfile(event.userId),
      networkContext: await this.getNetworkContext(event.details?.ipAddress)
    };
  }

  // Auto-blocking methods
  async blockUser(userId, reason, duration) {
    await require('../database/user-database').createAccountLock({
      userId,
      reason,
      lockType: 'temporary',
      duration,
      lockedAt: new Date().toISOString()
    });
  }

  async blockIP(ipAddress, reason, duration) {
    await require('../database/user-database').createIPBlock({
      ipAddress,
      reason,
      duration,
      blockedAt: new Date().toISOString()
    });
  }

  async suspendUserAccount(userId, reason) {
    await require('../database/user-database').updateUserAccountStatus(userId, 'suspended', {
      suspendedAt: new Date().toISOString(),
      suspensionReason: reason
    });
  }

  // Notification methods
  async sendEmailAlert(alert, config) {
    const notificationService = require('../support/notification-service');
    
    const recipients = await this.getAlertRecipients('email', alert.severity);
    
    for (const recipient of recipients) {
      await notificationService.sendEmail({
        recipient,
        subject: alert.title,
        template: 'security_alert',
        templateData: alert,
        priority: alert.severity,
        medicalContext: true
      });
    }
  }

  async sendSMSAlert(alert, config) {
    const notificationService = require('../support/notification-service');
    
    const recipients = await this.getAlertRecipients('sms', alert.severity);
    
    for (const recipient of recipients) {
      await notificationService.sendSMS({
        recipient,
        message: alert.message,
        template: 'security_urgent',
        templateData: alert,
        priority: 'critical',
        medicalContext: true
      });
    }
  }

  async sendPushAlert(alert, config) {
    // Implementation for push notifications
    console.log(`Sending push alert: ${alert.alertId}`);
  }

  async sendSlackAlert(alert, config) {
    // Implementation for Slack notifications
    console.log(`Sending Slack alert: ${alert.alertId}`);
  }

  async sendTeamsAlert(alert, config) {
    // Implementation for Teams notifications
    console.log(`Sending Teams alert: ${alert.alertId}`);
  }

  async notifyEscalationRecipient(recipient, alert, escalationRule) {
    const notificationService = require('../support/notification-service');
    
    await notificationService.sendToRole(recipient, {
      type: 'escalation',
      title: 'Security Alert Escalation',
      message: `Alert ${alert.alertId} escalated to ${escalationRule.name}`,
      data: { alertId: alert.alertId },
      priority: 'high'
    });
  }

  // Timeout management
  setEscalationTimeout(alert, timeoutMs) {
    alert.escalationTimeout = setTimeout(async () => {
      await this.handleEscalationTimeout(alert);
    }, timeoutMs);
  }

  async handleEscalationTimeout(alert) {
    console.log(`Escalation timeout reached for alert: ${alert.alertId}`);
    
    // Auto-escalate to next level
    const nextLevel = this.getNextEscalationLevel(alert.escalationLevel);
    if (nextLevel) {
      await this.escalateAlert(alert.alertId, 'system', nextLevel, 'auto_escalation_timeout');
    }
  }

  getNextEscalationLevel(currentLevel) {
    const levels = ['level_1', 'level_2', 'level_3', 'emergency'];
    const currentIndex = levels.indexOf(currentLevel);
    
    if (currentIndex < levels.length - 1) {
      return levels[currentIndex + 1];
    }
    
    return null;
  }

  // Suppression logic
  async shouldSuppressAlert(alert) {
    const suppressionKey = `${alert.rule}_${alert.metadata.sourceIP}`;
    const suppressionRule = this.suppressionRules.get(suppressionKey);
    
    if (suppressionRule) {
      const suppressionEnd = new Date(suppressionRule.suppressedUntil);
      return new Date() < suppressionEnd;
    }
    
    return false;
  }

  async getSuppressionReason(alert) {
    const suppressionKey = `${alert.rule}_${alert.metadata.sourceIP}`;
    const suppressionRule = this.suppressionRules.get(suppressionKey);
    return suppressionRule?.reason || 'Suppressed by rule';
  }

  // Database operations (placeholders)
  async storeAlert(alert) {
    await require('../database/user-database').storeSecurityAlert(alert);
  }

  async getAlertsInTimeframe(timeframe) {
    return await require('../database/user-database').getSecurityAlertsInTimeframe(timeframe);
  }

  async getRelatedAlerts(userId, timestamp) {
    const window = 60 * 60 * 1000; // 1 hour
    const start = new Date(new Date(timestamp).getTime() - window).toISOString();
    const end = new Date(new Date(timestamp).getTime() + window).toISOString();
    
    return await require('../database/user-database').getSecurityAlertsInRange(userId, start, end);
  }

  async getSystemState() {
    return await require('../database/user-database').getSystemState();
  }

  async getUserProfile(userId) {
    return await require('../database/user-database').getUserProfile(userId);
  }

  async getNetworkContext(ipAddress) {
    return await require('../database/user-database').getNetworkContext(ipAddress);
  }

  async getAlertRecipients(channel, severity) {
    return await require('../database/user-database').getAlertRecipients(channel, severity);
  }
}

module.exports = SecurityAlertManager;