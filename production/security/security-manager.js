/**
 * Production-Grade Security Manager Orchestrator
 * Central orchestration of all security and compliance components
 */

const path = require('path');
const EventEmitter = require('events');

class SecurityManager extends EventEmitter {
  constructor(configPath = null) {
    super();
    
    this.configPath = configPath;
    this.config = null;
    this.components = new Map();
    this.isInitialized = false;
    this.healthStatus = new Map();
    this.performanceMetrics = new Map();
    
    this.initializeComponents();
  }

  /**
   * Initialize all security components
   */
  async initialize() {
    try {
      console.log('[SECURITY-MANAGER] Initializing comprehensive security system...');
      
      // Load configuration
      await this.loadConfiguration();
      
      // Initialize all security components
      await this.initializeRBACManager();
      await this.initializeAuditLogger();
      await this.initializeEncryption();
      await this.initializePenetrationTesting();
      await this.initializeIncidentResponse();
      await this.initializeComplianceDocs();
      await this.initializeSecurityMonitoring();
      
      // Setup cross-component integrations
      await this.setupIntegrations();
      
      // Start monitoring and health checks
      this.startHealthMonitoring();
      this.startPerformanceMonitoring();
      
      this.isInitialized = true;
      this.emit('security_system_initialized', {
        timestamp: new Date().toISOString(),
        components_initialized: this.components.size,
        status: 'operational'
      });
      
      console.log('[SECURITY-MANAGER] All security components initialized successfully');
      return true;
      
    } catch (error) {
      console.error('[SECURITY-MANAGER] Failed to initialize security system:', error);
      this.emit('initialization_failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Load security configuration
   */
  async loadConfiguration() {
    const SecurityConfig = require('./config/security-config');
    this.config = new SecurityConfig();
    
    if (this.configPath) {
      await this.config.loadFromFile(this.configPath);
    }
    
    console.log('[SECURITY-MANAGER] Configuration loaded and validated');
  }

  /**
   * Initialize RBAC Manager component
   */
  async initializeRBACManager() {
    console.log('[SECURITY-MANAGER] Initializing RBAC Manager...');
    
    const RBACManager = require('./access-control/rbac-manager');
    const rbacConfig = this.config.getComponentConfig('rbac-manager');
    
    this.components.set('rbac-manager', new RBACManager(rbacConfig));
    
    // Setup event listeners
    this.components.get('rbac-manager').on('audit_event', (event) => {
      this.handleRBACAuditEvent(event);
    });
    
    this.healthStatus.set('rbac-manager', 'healthy');
    console.log('[SECURITY-MANAGER] RBAC Manager initialized');
  }

  /**
   * Initialize Audit Logger component
   */
  async initializeAuditLogger() {
    console.log('[SECURITY-MANAGER] Initializing Audit Logger...');
    
    const AuditLogger = require('./audit-logging/audit-logger');
    const auditConfig = this.config.getComponentConfig('audit-logger');
    
    this.components.set('audit-logger', new AuditLogger(auditConfig));
    
    this.healthStatus.set('audit-logger', 'healthy');
    console.log('[SECURITY-MANAGER] Audit Logger initialized');
  }

  /**
   * Initialize PHI Encryption component
   */
  async initializeEncryption() {
    console.log('[SECURITY-MANAGER] Initializing PHI Encryption...');
    
    const PHIEncryption = require('./encryption/phi-encryption');
    const encryptionConfig = this.config.getComponentConfig('phi-encryption');
    
    this.components.set('phi-encryption', new PHIEncryption(encryptionConfig));
    
    this.healthStatus.set('phi-encryption', 'healthy');
    console.log('[SECURITY-MANAGER] PHI Encryption initialized');
  }

  /**
   * Initialize Penetration Testing component
   */
  async initializePenetrationTesting() {
    console.log('[SECURITY-MANAGER] Initializing Penetration Testing Framework...');
    
    const PenetrationTesting = require('./penetration-testing/pen-test-framework');
    const penTestConfig = this.config.getComponentConfig('pen-test-framework');
    
    this.components.set('pen-test-framework', new PenetrationTesting(penTestConfig));
    
    // Start automated scanning
    this.components.get('pen-test-framework').startAutomatedScanning();
    
    this.healthStatus.set('pen-test-framework', 'healthy');
    console.log('[SECURITY-MANAGER] Penetration Testing Framework initialized');
  }

  /**
   * Initialize Incident Response component
   */
  async initializeIncidentResponse() {
    console.log('[SECURITY-MANAGER] Initializing Incident Response System...');
    
    const SecurityIncidentResponse = require('./incident-response/incident-response-system');
    const incidentConfig = this.config.getComponentConfig('incident-response');
    
    this.components.set('incident-response', new SecurityIncidentResponse(incidentConfig));
    
    this.healthStatus.set('incident-response', 'healthy');
    console.log('[SECURITY-MANAGER] Incident Response System initialized');
  }

  /**
   * Initialize Compliance Documentation component
   */
  async initializeComplianceDocs() {
    console.log('[SECURITY-MANAGER] Initializing Compliance Documentation...');
    
    const HIPAAComplianceDocs = require('./compliance-docs/compliance-docs-system');
    const complianceConfig = this.config.getComponentConfig('compliance-docs');
    
    this.components.set('compliance-docs', new HIPAAComplianceDocs(complianceConfig));
    
    this.healthStatus.set('compliance-docs', 'healthy');
    console.log('[SECURITY-MANAGER] Compliance Documentation initialized');
  }

  /**
   * Initialize Security Monitoring component
   */
  async initializeSecurityMonitoring() {
    console.log('[SECURITY-MANAGER] Initializing Security Monitoring...');
    
    const SecurityMonitor = require('./monitoring/security-monitor');
    const monitoringConfig = this.config.getComponentConfig('security-monitor');
    
    this.components.set('security-monitor', new SecurityMonitor(monitoringConfig));
    
    // Setup event listeners
    this.components.get('security-monitor').on('security_alert', (alert) => {
      this.handleSecurityAlert(alert);
    });
    
    this.components.get('security-monitor').on('security_event', (event) => {
      this.handleSecurityEvent(event);
    });
    
    this.healthStatus.set('security-monitor', 'healthy');
    console.log('[SECURITY-MANAGER] Security Monitoring initialized');
  }

  /**
   * Setup cross-component integrations
   */
  async setupIntegrations() {
    console.log('[SECURITY-MANAGER] Setting up component integrations...');
    
    const auditLogger = this.components.get('audit-logger');
    const securityMonitor = this.components.get('security-monitor');
    const incidentResponse = this.components.get('incident-response');
    
    // Monitor -> Audit Logger integration
    securityMonitor.on('security_event', async (event) => {
      await auditLogger.logEvent({
        userId: event.userId || 'system',
        action: `security_event_${event.type}`,
        resource: event.source,
        severity: event.severity,
        metadata: event
      });
    });
    
    // Security Monitor -> Incident Response integration
    securityMonitor.on('security_alert', async (alert) => {
      if (alert.severity === 'critical' || alert.severity === 'high') {
        await incidentResponse.reportIncident({
          title: `Security Alert: ${alert.rule_name}`,
          description: `Automated incident from security monitoring: ${alert.description}`,
          severity: alert.severity,
          category: 'security_incident',
          indicators: [{
            type: 'automated_detection',
            source: 'security_monitor',
            alert_id: alert.id
          }],
          autoGenerated: true
        });
      }
    });
    
    // Audit Logger -> Compliance integration
    auditLogger.on('audit_event', async (event) => {
      if (event.severity === 'HIGH' || event.category === 'PHI_ACCESS') {
        // Could trigger compliance checks or notifications
        this.emit('compliance_event', event);
      }
    });
    
    console.log('[SECURITY-MANAGER] Component integrations configured');
  }

  /**
   * Handle RBAC audit events
   */
  async handleRBACAuditEvent(event) {
    const auditLogger = this.components.get('audit-logger');
    await auditLogger.logEvent({
      userId: event.userId,
      action: `rbac_${event.action}`,
      resource: 'access_control_system',
      severity: event.severity,
      metadata: event
    });
  }

  /**
   * Handle security alerts from monitoring
   */
  async handleSecurityAlert(alert) {
    console.log(`[SECURITY-MANAGER] Processing security alert: ${alert.rule_name}`);
    
    // Log to audit trail
    const auditLogger = this.components.get('audit-logger');
    await auditLogger.logEvent({
      userId: 'system',
      action: `security_alert_${alert.category}`,
      resource: 'security_monitoring_system',
      severity: alert.severity,
      metadata: alert
    });
    
    // Check if incident should be created
    if (alert.response_actions?.action === 'incident') {
      const incidentResponse = this.components.get('incident-response');
      await incidentResponse.reportIncident({
        title: `Security Alert: ${alert.rule_name}`,
        description: alert.description,
        severity: alert.severity,
        category: alert.category,
        indicators: [{
          type: 'automated_detection',
          source: 'security_monitor',
          alert_id: alert.id
        }],
        autoGenerated: true
      });
    }
    
    this.emit('security_alert_processed', alert);
  }

  /**
   * Handle security events from monitoring
   */
  async handleSecurityEvent(event) {
    // Enhanced processing of security events
    if (event.severity === 'high' || event.severity === 'critical') {
      this.emit('high_severity_event', event);
    }
  }

  /**
   * User authentication with security checks
   */
  async authenticateUser(username, password, ipAddress = '', mfaToken = '') {
    const rbacManager = this.components.get('rbac-manager');
    const auditLogger = this.components.get('audit-logger');
    
    const result = await rbacManager.authenticate(username, password, ipAddress, mfaToken);
    
    // Log authentication attempt
    await auditLogger.logEvent({
      userId: result.success ? username : 'unknown',
      action: result.success ? 'authentication_success' : 'authentication_failed',
      resource: 'authentication_system',
      severity: result.success ? 'LOW' : 'MEDIUM',
      metadata: {
        ipAddress,
        userAgent: '', // Would be populated from request
        attempt_result: result.success ? 'success' : 'failed',
        failure_reason: result.reason
      }
    });
    
    return result;
  }

  /**
   * Check user permissions
   */
  hasPermission(sessionId, permission) {
    const rbacManager = this.components.get('rbac-manager');
    return rbacManager.hasPermission(sessionId, permission);
  }

  /**
   * Encrypt PHI data
   */
  async encryptPHIData(data, options = {}) {
    const encryption = this.components.get('phi-encryption');
    return await encryption.encryptPHI(data, options);
  }

  /**
   * Decrypt PHI data
   */
  async decryptPHIData(encryptedPackage, options = {}) {
    const encryption = this.components.get('phi-encryption');
    return await encryption.decryptPHI(encryptedPackage, options);
  }

  /**
   * Log PHI access for compliance
   */
  async logPHIAccess(userId, patientId, action, resource = 'patient_record') {
    const auditLogger = this.components.get('audit-logger');
    
    return await auditLogger.logEvent({
      userId,
      action: `phi_${action}`,
      resource,
      severity: 'HIGH',
      metadata: {
        patient_id: patientId,
        access_type: action,
        phi_accessed: true,
        compliance_required: true
      }
    });
  }

  /**
   * Initiate security scan
   */
  async initiateSecurityScan(target, options = {}) {
    const penTest = this.components.get('pen-test-framework');
    return await penTest.scheduleScan(target, options);
  }

  /**
   * Report security incident
   */
  async reportSecurityIncident(incidentData) {
    const incidentResponse = this.components.get('incident-response');
    return await incidentResponse.reportIncident(incidentData);
  }

  /**
   * Get compliance report
   */
  async generateComplianceReport(period) {
    const complianceDocs = this.components.get('compliance-docs');
    return await complianceDocs.generateComplianceReport(period);
  }

  /**
   * Get security dashboard data
   */
  getSecurityDashboard() {
    const securityMonitor = this.components.get('security-monitor');
    const dashboard = securityMonitor.getSecurityDashboard();
    
    // Add component health status
    dashboard.component_health = Object.fromEntries(this.healthStatus);
    
    // Add performance metrics
    dashboard.performance_metrics = Object.fromEntries(this.performanceMetrics);
    
    return dashboard;
  }

  /**
   * Get security metrics
   */
  getSecurityMetrics() {
    const metrics = {
      timestamp: new Date().toISOString(),
      components: {},
      overall_health: 'healthy',
      performance: {}
    };
    
    // Collect metrics from all components
    for (const [componentName, component] of this.components.entries()) {
      try {
        if (component.getSecurityStatistics) {
          metrics.components[componentName] = component.getSecurityStatistics();
        }
      } catch (error) {
        console.warn(`[SECURITY-MANAGER] Failed to get metrics from ${componentName}:`, error);
      }
    }
    
    // Calculate overall health
    const unhealthyComponents = Array.from(this.healthStatus.values())
      .filter(status => status !== 'healthy').length;
    
    if (unhealthyComponents > 0) {
      metrics.overall_health = unhealthyComponents > 2 ? 'critical' : 'degraded';
    }
    
    return metrics;
  }

  /**
   * Start health monitoring
   */
  startHealthMonitoring() {
    setInterval(async () => {
      await this.checkComponentHealth();
    }, 60000); // Check every minute
    
    console.log('[SECURITY-MANAGER] Health monitoring started');
  }

  /**
   * Check health of all components
   */
  async checkComponentHealth() {
    for (const [componentName, component] of this.components.entries()) {
      try {
        // Basic health check for each component
        if (component.getSecurityStatistics) {
          const stats = component.getSecurityStatistics();
          if (stats) {
            this.healthStatus.set(componentName, 'healthy');
          } else {
            this.healthStatus.set(componentName, 'warning');
          }
        }
      } catch (error) {
        console.error(`[SECURITY-MANAGER] Health check failed for ${componentName}:`, error);
        this.healthStatus.set(componentName, 'critical');
        
        this.emit('component_health_failure', {
          component: componentName,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Start performance monitoring
   */
  startPerformanceMonitoring() {
    setInterval(() => {
      this.collectPerformanceMetrics();
    }, 300000); // Collect every 5 minutes
    
    console.log('[SECURITY-MANAGER] Performance monitoring started');
  }

  /**
   * Collect performance metrics from components
   */
  collectPerformanceMetrics() {
    for (const [componentName, component] of this.components.entries()) {
      try {
        if (component.getMonitoringStatistics) {
          const stats = component.getMonitoringStatistics();
          this.performanceMetrics.set(componentName, {
            timestamp: new Date().toISOString(),
            ...stats
          });
        }
      } catch (error) {
        console.warn(`[SECURITY-MANAGER] Performance monitoring failed for ${componentName}:`, error);
      }
    }
  }

  /**
   * Emergency security procedures
   */
  async emergencyResponse(reason, affectedSystems = []) {
    console.log(`[SECURITY-MANAGER] EMERGENCY RESPONSE: ${reason}`);
    
    // Enable emergency monitoring mode
    const securityMonitor = this.components.get('security-monitor');
    securityMonitor.enableEmergencyMode(reason);
    
    // Create emergency incident
    const incidentId = await this.reportSecurityIncident({
      title: `EMERGENCY: ${reason}`,
      description: `Emergency response initiated: ${reason}`,
      severity: 'CRITICAL',
      category: 'emergency',
      affectedSystems,
      involvesPHI: true, // Assume PHI is involved in emergencies
      autoGenerated: true,
      source: 'emergency_system'
    });
    
    // Notify all stakeholders
    this.emit('emergency_response', {
      reason,
      incidentId,
      affectedSystems,
      timestamp: new Date().toISOString()
    });
    
    return incidentId;
  }

  /**
   * Perform security system backup
   */
  async backupSecurityData() {
    console.log('[SECURITY-MANAGER] Starting security data backup...');
    
    const backupResults = [];
    
    // Backup each component's critical data
    for (const [componentName, component] of this.components.entries()) {
      try {
        if (component.backupKeys || component.archiveOldLogs || component.exportConfiguration) {
          let backupPath;
          
          if (component.backupKeys) {
            backupPath = await component.backupKeys();
          } else if (component.archiveOldLogs) {
            await component.archiveOldLogs();
            backupPath = './backup/logs';
          } else if (component.exportConfiguration) {
            backupPath = await component.exportConfiguration();
          }
          
          backupResults.push({
            component: componentName,
            status: 'success',
            path: backupPath
          });
        }
      } catch (error) {
        backupResults.push({
          component: componentName,
          status: 'failed',
          error: error.message
        });
      }
    }
    
    console.log(`[SECURITY-MANAGER] Backup completed: ${backupResults.length} components processed`);
    return backupResults;
  }

  /**
   * Shutdown security system gracefully
   */
  async shutdown() {
    console.log('[SECURITY-MANAGER] Shutting down security system...');
    
    // Stop all monitoring
    this.removeAllListeners();
    
    // Gracefully shutdown each component
    for (const [componentName, component] of this.components.entries()) {
      try {
        if (component.secureWipe) {
          component.secureWipe();
        }
        console.log(`[SECURITY-MANAGER] ${componentName} shutdown completed`);
      } catch (error) {
        console.error(`[SECURITY-MANAGER] Error shutting down ${componentName}:`, error);
      }
    }
    
    this.components.clear();
    this.healthStatus.clear();
    this.performanceMetrics.clear();
    
    console.log('[SECURITY-MANAGER] Security system shutdown complete');
  }

  /**
   * Get system configuration
   */
  getConfiguration() {
    return this.config ? this.config.getConfigurationSummary() : null;
  }

  /**
   * Update system configuration
   */
  updateConfiguration(path, value) {
    if (this.config) {
      this.config.set(path, value);
      console.log(`[SECURITY-MANAGER] Configuration updated: ${path}`);
    }
  }

  /**
   * Export system configuration
   */
  exportConfiguration(format = 'json') {
    if (this.config) {
      return this.config.exportConfiguration(format);
    }
    return null;
  }

  /**
   * Validate security configuration
   */
  validateConfiguration() {
    if (this.config) {
      return this.config.getValidationReport();
    }
    return { status: 'no_config', timestamp: new Date().toISOString() };
  }

  /**
   * Get component instance
   */
  getComponent(componentName) {
    return this.components.get(componentName);
  }

  /**
   * Check if system is initialized
   */
  isSystemInitialized() {
    return this.isInitialized;
  }

  /**
   * Get system status
   */
  getSystemStatus() {
    return {
      initialized: this.isInitialized,
      components: this.components.size,
      health: Object.fromEntries(this.healthStatus),
      last_health_check: new Date().toISOString(),
      uptime: process.uptime(),
      version: '1.0.0'
    };
  }

  /**
   * Run comprehensive security test
   */
  async runSecurityTest() {
    console.log('[SECURITY-MANAGER] Running comprehensive security test...');
    
    const results = {
      test_id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      components_tested: [],
      overall_status: 'passing',
      issues: [],
      recommendations: []
    };

    // Test each component
    for (const [componentName, component] of this.components.entries()) {
      try {
        const testResult = await this.testComponent(componentName, component);
        results.components_tested.push({
          name: componentName,
          status: testResult.status,
          details: testResult.details
        });
        
        if (testResult.status !== 'pass') {
          results.overall_status = 'failing';
          results.issues.push({
            component: componentName,
            issue: testResult.issue,
            severity: testResult.severity || 'medium'
          });
        }
      } catch (error) {
        results.components_tested.push({
          name: componentName,
          status: 'error',
          error: error.message
        });
        
        results.overall_status = 'error';
        results.issues.push({
          component: componentName,
          issue: error.message,
          severity: 'high'
        });
      }
    }

    console.log(`[SECURITY-MANAGER] Security test completed: ${results.overall_status}`);
    return results;
  }

  /**
   * Test individual component
   */
  async testComponent(componentName, component) {
    const testResults = {
      'rbac-manager': async () => {
        // Test authentication
        const testUser = {
          username: 'test_user',
          email: 'test@example.com',
          password: 'TestPassword123!',
          roleId: 'doctor',
          ipAddress: '127.0.0.1'
        };
        
        try {
          await component.registerUser(testUser);
          return { status: 'pass', details: 'User registration works' };
        } catch (error) {
          return { 
            status: 'fail', 
            issue: `User registration failed: ${error.message}`,
            severity: 'medium'
          };
        }
      },
      'audit-logger': async () => {
        // Test logging
        try {
          const eventId = await component.logEvent({
            userId: 'test_user',
            action: 'test_action',
            resource: 'test_resource',
            severity: 'LOW'
          });
          return { status: 'pass', details: `Test log entry created: ${eventId}` };
        } catch (error) {
          return { 
            status: 'fail', 
            issue: `Audit logging failed: ${error.message}`,
            severity: 'high'
          };
        }
      },
      'phi-encryption': async () => {
        // Test encryption
        try {
          const testData = { test: 'data', phi: 'test_phi' };
          const encrypted = await component.encryptPHI(testData);
          const decrypted = await component.decryptPHI(encrypted);
          
          if (JSON.stringify(decrypted.data) === JSON.stringify(testData)) {
            return { status: 'pass', details: 'Encryption/decryption works' };
          } else {
            return { 
              status: 'fail', 
              issue: 'Encryption/decryption data mismatch',
              severity: 'critical'
            };
          }
        } catch (error) {
          return { 
            status: 'fail', 
            issue: `Encryption test failed: ${error.message}`,
            severity: 'critical'
          };
        }
      },
      'security-monitor': async () => {
        // Test monitoring
        try {
          const dashboard = component.getSecurityDashboard();
          return { status: 'pass', details: 'Security monitoring accessible' };
        } catch (error) {
          return { 
            status: 'fail', 
            issue: `Security monitoring failed: ${error.message}`,
            severity: 'high'
          };
        }
      }
    };

    if (testResults[componentName]) {
      return await testResults[componentName]();
    } else {
      return { status: 'skip', details: 'No test defined for this component' };
    }
  }
}

module.exports = SecurityManager;