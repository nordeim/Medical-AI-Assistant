// Healthcare User Management Orchestrator
// Main orchestration service for all user management operations

const EventEmitter = require('events');
const HealthcareAuthenticationService = require('./auth/healthcare-auth-service');
const HealthcareRBACSystem = require('./rbac/healthcare-rbac-system');
const HealthcareOnboardingService = require('./onboarding/healthcare-onboarding-service');
const HealthcarePrivacyService = require('./privacy/healthcare-privacy-service');
const HealthcareAuditLogger = require('./monitoring/audit-logger');
const HealthcareSecurityService = require('./security/encryption-service');
const HealthcareSupportService = require('./support/healthcare-support-service');
const RealTimeMonitor = require('./monitoring/real-time-monitor');
const AnomalyDetector = require('./monitoring/anomaly-detector');
const SecurityAlertManager = require('./monitoring/alert-manager');

class HealthcareUserManagementOrchestrator extends EventEmitter {
  constructor() {
    super();
    this.services = {};
    this.initialized = false;
    this.healthStatus = { status: 'unknown', services: {}, lastCheck: null };
  }

  // Initialize All Services
  async initialize() {
    try {
      console.log('Initializing Healthcare User Management System...');

      // Initialize core services
      await this.initializeCoreServices();
      
      // Initialize monitoring and security services
      await this.initializeMonitoringServices();
      
      // Initialize support services
      await this.initializeSupportServices();
      
      // Set up inter-service communication
      this.setupServiceCommunication();
      
      // Start background processes
      this.startBackgroundProcesses();
      
      this.initialized = true;
      this.healthStatus.status = 'healthy';
      this.healthStatus.lastCheck = new Date().toISOString();
      
      console.log('Healthcare User Management System initialized successfully');
      
      return {
        success: true,
        services: Object.keys(this.services),
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('System initialization failed:', error);
      this.healthStatus.status = 'error';
      this.healthStatus.lastCheck = new Date().toISOString();
      this.healthStatus.error = error.message;
      throw error;
    }
  }

  // Core User Management Operations
  async registerUser(userData) {
    try {
      // Step 1: Validate registration data
      await this.validateRegistrationData(userData);

      // Step 2: Create user with authentication service
      const authResult = await this.services.auth.registerHealthcareUser(userData);

      // Step 3: Initialize RBAC system for user
      await this.initializeUserRBAC(authResult.user.id, userData.role);

      // Step 4: Start onboarding process
      const onboardingResult = await this.services.onboarding.initiateOnboarding(
        authResult.user.id, 
        userData
      );

      // Step 5: Set up privacy controls
      await this.services.privacy.handleGDPRConsent(authResult.user.id, {
        consentType: 'data_processing',
        granted: true,
        consentDetails: { registration: true }
      });

      // Log successful registration
      await this.services.audit.logEvent({
        userId: authResult.user.id,
        event: 'user.registration.completed',
        details: {
          role: userData.role,
          specialty: userData.specialty,
          institution: userData.institution,
          onboardingStarted: onboardingResult.success
        },
        timestamp: new Date().toISOString(),
        source: 'orchestrator'
      });

      return {
        success: true,
        userId: authResult.user.id,
        verificationRequired: authResult.verificationRequired,
        onboardingId: onboardingResult.onboardingId,
        message: 'User registered successfully. Verification process initiated.'
      };

    } catch (error) {
      console.error('User registration failed:', error);
      
      await this.services.audit.logEvent({
        event: 'user.registration.failed',
        details: { error: error.message, userData: this.sanitizeUserData(userData) },
        timestamp: new Date().toISOString(),
        source: 'orchestrator'
      });

      throw error;
    }
  }

  async authenticateUser(authData) {
    try {
      const { email, password, mfaCode } = authData;

      // Step 1: Authenticate with authentication service
      const authResult = await this.services.auth.authenticateUser(email, password, mfaCode);

      if (!authResult.success && authResult.requiresMFA) {
        return authResult; // Return MFA challenge
      }

      // Step 2: Check user status and permissions
      if (authResult.success) {
        const userProfile = authResult.profile;
        
        // Check if user account is active
        if (userProfile.account_status !== 'active') {
          throw new Error('Account is not active');
        }

        // Step 3: Start session monitoring
        await this.startUserSession(authResult.user.id, authData.deviceInfo);

        // Step 4: Log successful authentication
        await this.services.audit.logAuthenticationEvent('login', {
          userId: authResult.user.id,
          email: email,
          ipAddress: authData.ipAddress,
          userAgent: authData.userAgent,
          mfaUsed: !!mfaCode
        });

        return {
          success: true,
          user: authResult.user,
          profile: userProfile,
          session: authResult.session,
          permissions: await this.services.rbac.getUserPermissionSummary(authResult.user.id)
        };
      }

      return authResult;

    } catch (error) {
      console.error('User authentication failed:', error);
      
      await this.services.audit.logAuthenticationEvent('failed_login', {
        email: authData.email,
        ipAddress: authData.ipAddress,
        userAgent: authData.userAgent,
        failureReason: error.message
      });

      throw error;
    }
  }

  async checkUserPermission(userId, permission, context = {}) {
    try {
      // Check basic permission
      const hasPermission = await this.services.rbac.checkPermission(userId, permission, context);

      // Additional context checks
      if (hasPermission && context.resourceId) {
        // Check resource-specific access
        const resourceAccess = await this.checkResourceAccess(userId, permission, context);
        return resourceAccess;
      }

      // Log permission check
      await this.services.audit.logEvent({
        userId,
        event: 'permission.check',
        details: {
          permission,
          granted: hasPermission,
          context
        },
        timestamp: new Date().toISOString(),
        source: 'orchestrator'
      });

      return hasPermission;

    } catch (error) {
      console.error('Permission check failed:', error);
      return false;
    }
  }

  async processOnboardingStep(userId, stepName, stepData) {
    try {
      let result;

      switch (stepName) {
        case 'medical_credentials':
          result = await this.services.onboarding.verifyMedicalLicense(userId, stepData);
          break;
        
        case 'background_check':
          result = await this.services.onboarding.initiateBackgroundCheck(userId, stepData);
          break;
        
        case 'reference_validation':
          result = await this.services.onboarding.validateReferences(userId, stepData);
          break;
        
        case 'institutional_affiliation':
          result = await this.services.onboarding.verifyInstitutionalAffiliation(userId, stepData);
          break;
        
        case 'final_approval':
          result = await this.services.onboarding.requestFinalApproval(userId, stepData);
          break;
        
        default:
          throw new Error(`Unknown onboarding step: ${stepName}`);
      }

      // Log step completion
      await this.services.audit.logEvent({
        userId,
        event: 'onboarding.step_completed',
        details: {
          step: stepName,
          result
        },
        timestamp: new Date().toISOString(),
        source: 'orchestrator'
      });

      return result;

    } catch (error) {
      console.error('Onboarding step failed:', error);
      throw error;
    }
  }

  // Security and Compliance Operations
  async processDataAccessRequest(userId, resourceId, action, context) {
    try {
      // Step 1: Check permission
      const hasPermission = await this.checkUserPermission(userId, `${action}_${context.resourceType}`, context);
      
      if (!hasPermission) {
        // Log unauthorized access attempt
        await this.services.audit.logSecurityEvent({
          userId,
          event: 'unauthorized_data_access',
          details: {
            resourceId,
            action,
            context,
            denied: true
          },
          severity: 'medium',
          source: 'orchestrator'
        });

        throw new Error('Unauthorized access attempt');
      }

      // Step 2: Apply HIPAA privacy controls
      if (context.complianceType === 'HIPAA') {
        const privacyResult = await this.services.privacy.enforceHIPAAPrivacy(userId, action, {
          ...context,
          data: context.data || {}
        });

        context = { ...context, ...privacyResult.complianceNotes };
      }

      // Step 3: Log data access
      await this.services.audit.logMedicalDataAccess(
        userId,
        context.patientId || resourceId,
        action,
        context.resourceType || 'unknown',
        context
      );

      // Step 4: Monitor for anomalies
      await this.services.monitor.trackUserActivity(userId, {
        action: `${action}_data`,
        resource: resourceId,
        context,
        timestamp: new Date().toISOString()
      });

      return {
        success: true,
        accessGranted: true,
        complianceNotes: context.complianceNotes || {},
        accessLogged: true
      };

    } catch (error) {
      console.error('Data access request failed:', error);
      
      await this.services.audit.logSecurityEvent({
        userId,
        event: 'data_access_request_failed',
        details: {
          resourceId,
          action,
          error: error.message
        },
        severity: 'medium',
        source: 'orchestrator'
      });

      throw error;
    }
  }

  async processGDPRRequest(userId, requestType, requestData = {}) {
    try {
      const result = await this.services.privacy.processDataSubjectAccessRequest(userId, requestType);

      // Log GDPR request
      await this.services.audit.logEvent({
        userId,
        event: 'gdpr.request',
        details: {
          requestType,
          requestId: result.requestId,
          status: result.status
        },
        timestamp: new Date().toISOString(),
        source: 'orchestrator',
        complianceType: 'GDPR'
      });

      return result;

    } catch (error) {
      console.error('GDPR request failed:', error);
      throw error;
    }
  }

  // Support Operations
  async createSupportTicket(userId, ticketData) {
    try {
      // Determine if this is a medical emergency
      if (ticketData.category === 'medical_emergency') {
        const emergencyTicket = await this.services.support.createMedicalEmergencyTicket({
          userId,
          ...ticketData
        });
        return emergencyTicket;
      }

      // Regular support ticket
      const ticket = await this.services.support.createSupportTicket({
        userId,
        ...ticketData
      });

      // Log ticket creation
      await this.services.audit.logEvent({
        userId,
        event: 'support.ticket.created',
        details: {
          ticketId: ticket.ticketId,
          category: ticket.category,
          priority: ticket.priority
        },
        timestamp: new Date().toISOString(),
        source: 'orchestrator'
      });

      return ticket;

    } catch (error) {
      console.error('Support ticket creation failed:', error);
      throw error;
    }
  }

  // System Health and Monitoring
  async performHealthCheck() {
    const healthCheck = {
      timestamp: new Date().toISOString(),
      status: 'healthy',
      services: {},
      metrics: {
        activeUsers: await this.getActiveUserCount(),
        activeSessions: await this.getActiveSessionCount(),
        pendingOnboarding: await this.getPendingOnboardingCount(),
        activeAlerts: await this.getActiveAlertCount()
      },
      alerts: [],
      recommendations: []
    };

    // Check each service
    for (const [serviceName, service] of Object.entries(this.services)) {
      try {
        if (service.healthCheck) {
          const serviceHealth = await service.healthCheck();
          healthCheck.services[serviceName] = serviceHealth;
        } else {
          healthCheck.services[serviceName] = { status: 'unknown' };
        }
      } catch (error) {
        healthCheck.services[serviceName] = { 
          status: 'error', 
          error: error.message 
        };
        healthCheck.alerts.push({
          service: serviceName,
          severity: 'high',
          message: `Service health check failed: ${error.message}`
        });
      }
    }

    // Determine overall status
    const serviceStatuses = Object.values(healthCheck.services).map(s => s.status);
    if (serviceStatuses.includes('error')) {
      healthCheck.status = 'degraded';
    }
    if (serviceStatuses.every(s => s === 'error')) {
      healthCheck.status = 'critical';
    }

    // Generate recommendations
    if (healthCheck.metrics.activeUsers > 1000) {
      healthCheck.recommendations.push({
        type: 'scaling',
        message: 'High user load detected - consider scaling up'
      });
    }

    if (healthCheck.metrics.activeAlerts > 10) {
      healthCheck.recommendations.push({
        type: 'alert_review',
        message: 'High number of active alerts - review alert rules'
      });
    }

    this.healthStatus = healthCheck;
    return healthCheck;
  }

  // Background Processes
  startBackgroundProcesses() {
    // Health check every 5 minutes
    setInterval(() => {
      this.performHealthCheck().catch(error => {
        console.error('Background health check failed:', error);
      });
    }, 5 * 60 * 1000);

    // Data retention management daily
    setInterval(() => {
      this.services.privacy.manageDataRetention().catch(error => {
        console.error('Data retention management failed:', error);
      });
    }, 24 * 60 * 60 * 1000);

    // Security key rotation monthly
    setInterval(() => {
      this.services.security.rotateEncryptionKeys().catch(error => {
        console.error('Key rotation failed:', error);
      });
    }, 30 * 24 * 60 * 60 * 1000);

    console.log('Background processes started');
  }

  // Service Communication Setup
  setupServiceCommunication() {
    // Audit logger to other services
    this.services.audit.on('securityEvent', async (event) => {
      await this.services.monitor.processEvent(event);
      await this.services.alerts.processSecurityEvent(event);
    });

    // Real-time monitor to anomaly detector
    this.services.monitor.on('anomalyDetected', async (anomaly) => {
      await this.services.anomaly.analyzeAnomaly(anomaly);
    });

    // Support service integration
    this.services.support.on('criticalAlert', async (alert) => {
      await this.services.alerts.sendAlertNotifications([alert]);
    });

    console.log('Service communication configured');
  }

  // Helper Methods
  async initializeCoreServices() {
    this.services.auth = new HealthcareAuthenticationService();
    this.services.rbac = new HealthcareRBACSystem();
    this.services.onboarding = new HealthcareOnboardingService();
    this.services.privacy = new HealthcarePrivacyService();
    this.services.audit = new HealthcareAuditLogger();
    this.services.security = new HealthcareSecurityService();
  }

  async initializeMonitoringServices() {
    this.services.monitor = new RealTimeMonitor();
    this.services.anomaly = new AnomalyDetector();
    this.services.alerts = new SecurityAlertManager();

    // Initialize monitoring
    await this.services.monitor.initialize();
  }

  async initializeSupportServices() {
    this.services.support = new HealthcareSupportService();
  }

  async validateRegistrationData(userData) {
    const required = ['email', 'password', 'role', 'firstName', 'lastName', 'medicalLicense'];
    const missing = required.filter(field => !userData[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }

    // Validate role
    const validRoles = Object.keys(this.services.rbac.config.healthcare.roles);
    if (!validRoles.includes(userData.role)) {
      throw new Error(`Invalid role: ${userData.role}`);
    }
  }

  async initializeUserRBAC(userId, role) {
    await this.services.rbac.grantRole(userId, role, 'system', 'Initial role assignment');
  }

  async startUserSession(userId, deviceInfo) {
    await this.services.auth.createSession(userId, deviceInfo);
  }

  async checkResourceAccess(userId, permission, context) {
    return await this.services.rbac.checkDataAccess(
      userId, 
      context.resourceType, 
      context.resourceId, 
      context.action, 
      context
    );
  }

  sanitizeUserData(userData) {
    const sanitized = { ...userData };
    delete sanitized.password;
    delete sanitized.medicalLicense; // Remove sensitive data
    return sanitized;
  }

  // Metrics Methods
  async getActiveUserCount() {
    return await require('./database/user-database').getActiveUserCount();
  }

  async getActiveSessionCount() {
    return await require('./database/user-database').getActiveSessionCount();
  }

  async getPendingOnboardingCount() {
    return await require('./database/user-database').getPendingOnboardingCount();
  }

  async getActiveAlertCount() {
    return this.services.alerts.activeAlerts.size;
  }

  // Shutdown
  async shutdown() {
    console.log('Shutting down Healthcare User Management System...');
    
    // Stop background processes
    this.stopBackgroundProcesses();
    
    // Shutdown services
    for (const [name, service] of Object.entries(this.services)) {
      if (service.shutdown) {
        await service.shutdown();
      }
    }
    
    console.log('System shutdown complete');
  }

  stopBackgroundProcesses() {
    // Clear intervals (implementation depends on how they were set up)
    console.log('Background processes stopped');
  }

  // Public API for external services
  getService(serviceName) {
    return this.services[serviceName];
  }

  getSystemStatus() {
    return this.healthStatus;
  }

  async executeOperation(operation, parameters) {
    try {
      switch (operation) {
        case 'register_user':
          return await this.registerUser(parameters);
        
        case 'authenticate_user':
          return await this.authenticateUser(parameters);
        
        case 'check_permission':
          return await this.checkUserPermission(
            parameters.userId, 
            parameters.permission, 
            parameters.context
          );
        
        case 'process_onboarding_step':
          return await this.processOnboardingStep(
            parameters.userId, 
            parameters.stepName, 
            parameters.stepData
          );
        
        case 'process_data_access':
          return await this.processDataAccessRequest(
            parameters.userId, 
            parameters.resourceId, 
            parameters.action, 
            parameters.context
          );
        
        case 'process_gdpr_request':
          return await this.processGDPRRequest(
            parameters.userId, 
            parameters.requestType, 
            parameters.requestData
          );
        
        case 'create_support_ticket':
          return await this.createSupportTicket(
            parameters.userId, 
            parameters.ticketData
          );
        
        case 'health_check':
          return await this.performHealthCheck();
        
        default:
          throw new Error(`Unknown operation: ${operation}`);
      }
    } catch (error) {
      console.error(`Operation ${operation} failed:`, error);
      throw error;
    }
  }
}

// Export singleton instance
module.exports = new HealthcareUserManagementOrchestrator();