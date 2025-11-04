// Healthcare User Privacy and Compliance Management System
// GDPR and HIPAA compliant data handling for healthcare professionals

const crypto = require('crypto');
const config = require('../config/user-management-config');

class HealthcarePrivacyService {
  constructor() {
    this.config = config;
    this.auditLogger = require('../monitoring/audit-logger');
    this.encryptionService = require('../security/encryption-service');
    this.dataRetentionManager = require('./data-retention-manager');
  }

  // GDPR Compliance Management
  async handleGDPRConsent(userId, consentData) {
    try {
      const {
        consentType,
        granted,
        consentDetails,
        ipAddress,
        userAgent,
        timestamp
      } = consentData;

      // Record consent in privacy ledger
      const consentRecord = {
        userId,
        consentId: crypto.randomUUID(),
        consentType, // 'data_processing', 'marketing', 'analytics', etc.
        granted,
        consentDetails,
        ipAddress,
        userAgent,
        timestamp: timestamp || new Date().toISOString(),
        version: '1.0',
        expiresAt: this.calculateConsentExpiry(consentType)
      };

      const { error } = await require('../database/user-database').recordGDPRConsent(consentRecord);
      
      if (error) {
        throw new Error(`Failed to record GDPR consent: ${error.message}`);
      }

      // Update user privacy preferences
      await this.updateUserPrivacyPreferences(userId, {
        consentGiven: granted,
        consentType,
        lastUpdated: new Date().toISOString()
      });

      // If consent withdrawn, initiate data deletion process
      if (!granted) {
        await this.handleConsentWithdrawal(userId, consentType);
      }

      // Log consent event
      await this.auditLogger.logEvent({
        userId,
        event: 'gdpr.consent',
        details: {
          consentType,
          granted,
          consentId: consentRecord.consentId,
          ipAddress: this.hashIP(ipAddress)
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'GDPR'
      });

      return {
        success: true,
        consentId: consentRecord.consentId,
        expiresAt: consentRecord.expiresAt,
        dataProcessingAffected: this.getAffectedProcessing(consentType)
      };

    } catch (error) {
      console.error('GDPR consent handling error:', error);
      throw error;
    }
  }

  // Data Subject Access Request (DSAR)
  async processDataSubjectAccessRequest(userId, requestType) {
    try {
      const requestId = crypto.randomUUID();
      
      // Create access request record
      const accessRequest = {
        requestId,
        userId,
        requestType, // 'export', 'portability', 'rectification', 'erasure'
        status: 'processing',
        createdAt: new Date().toISOString(),
        estimatedCompletion: this.calculateDSARCompletion(),
        priority: 'normal'
      };

      await require('../database/user-database').createDataAccessRequest(accessRequest);

      // Compile user data
      const userData = await this.compileUserData(userId);

      // Process based on request type
      switch (requestType) {
        case 'export':
          return await this.processDataExport(userId, requestId, userData);
        
        case 'portability':
          return await this.processDataPortability(userId, requestId, userData);
        
        case 'rectification':
          return await this.processDataRectification(userId, requestId);
        
        case 'erasure':
          return await this.processDataErasure(userId, requestId);
        
        default:
          throw new Error(`Unsupported DSAR type: ${requestType}`);
      }

    } catch (error) {
      console.error('DSAR processing error:', error);
      throw error;
    }
  }

  // Data Minimization Implementation
  async implementDataMinimization(userId, purpose) {
    try {
      const userProfile = await require('../database/user-database').getUserProfile(userId);
      const requiredFields = this.getRequiredFieldsForPurpose(purpose);
      
      // Identify excessive data
      const currentFields = Object.keys(userProfile);
      const excessiveFields = currentFields.filter(field => 
        !requiredFields.includes(field) && 
        !this.isRequiredForCompliance(field)
      );

      // Anonymize or remove excessive data
      const minimizationActions = [];
      
      for (const field of excessiveFields) {
        if (this.canAnonymize(field)) {
          // Anonymize field
          await this.anonymizeUserField(userId, field);
          minimizationActions.push({ field, action: 'anonymized' });
        } else {
          // Remove field entirely
          await this.removeUserField(userId, field);
          minimizationActions.push({ field, action: 'removed' });
        }
      }

      // Log data minimization
      await this.auditLogger.logEvent({
        userId,
        event: 'data.minimization',
        details: {
          purpose,
          minimizationActions,
          timestamp: new Date().toISOString()
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'GDPR'
      });

      return {
        success: true,
        minimizedFields: minimizationActions.length,
        actions: minimizationActions
      };

    } catch (error) {
      console.error('Data minimization error:', error);
      throw error;
    }
  }

  // HIPAA Privacy Controls
  async enforceHIPAAPrivacy(userId, action, dataContext) {
    try {
      const userRoles = await require('../rbac/healthcare-rbac-system').getUserRoles(userId);
      const userProfile = await require('../database/user-database').getUserProfile(userId);

      // Minimum necessary standard
      const minimumNecessaryData = this.filterMinimumNecessaryData(
        dataContext.data,
        action,
        userRoles,
        userProfile.specialty
      );

      // Log HIPAA access
      await this.auditLogger.logEvent({
        userId,
        event: 'hipaa.access',
        details: {
          action,
          dataType: dataContext.type,
          patientId: dataContext.patientId,
          minimumNecessaryApplied: true,
          timestamp: new Date().toISOString()
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'HIPAA'
      });

      // Encrypt sensitive data
      const encryptedData = await this.encryptionService.encrypt(minimumNecessaryData);

      // Apply access restrictions
      await this.applyHIPAAccessRestrictions(userId, dataContext);

      return {
        success: true,
        data: encryptedData,
        complianceNotes: {
          minimumNecessary: true,
          encrypted: true,
          accessLogged: true,
          restrictionsApplied: true
        }
      };

    } catch (error) {
      console.error('HIPAA privacy enforcement error:', error);
      throw error;
    }
  }

  // Right to Erasure (Right to be Forgotten)
  async processDataErasure(userId, requestId, erasureScope = 'full') {
    try {
      // Validate erasure request
      await this.validateErasureRequest(userId, erasureScope);

      const erasurePlan = await this.createErasurePlan(userId, erasureScope);
      
      // Execute erasure
      const erasureResults = [];
      
      for (const dataSet of erasurePlan.dataSets) {
        const result = await this.eraseDataSet(userId, dataSet);
        erasureResults.push(result);
      }

      // Update request status
      await require('../database/user-database').updateDataAccessRequestStatus(requestId, 'completed', {
        completedAt: new Date().toISOString(),
        erasureResults
      });

      // Log erasure completion
      await this.auditLogger.logEvent({
        userId,
        event: 'data.erasure.completed',
        details: {
          requestId,
          erasureScope,
          dataSetsProcessed: erasureResults.length,
          completionTime: new Date().toISOString()
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'GDPR'
      });

      return {
        success: true,
        requestId,
        erasureScope,
        dataSetsProcessed: erasureResults.length,
        completionTime: new Date().toISOString()
      };

    } catch (error) {
      console.error('Data erasure error:', error);
      throw error;
    }
  }

  // Data Portability
  async processDataPortability(userId, requestId, userData) {
    try {
      // Export in machine-readable format (JSON)
      const portableData = {
        userId,
        exportDate: new Date().toISOString(),
        version: '1.0',
        data: {
          profile: userData.profile,
          preferences: userData.preferences,
          activity: userData.activity,
          compliance: userData.compliance
        }
      };

      // Create secure download
      const exportFile = await this.createSecureExport(portableData);
      
      // Update request status
      await require('../database/user-database').updateDataAccessRequestStatus(requestId, 'completed', {
        completedAt: new Date().toISOString(),
        exportFile: exportFile.filePath,
        expiresAt: exportFile.expiresAt
      });

      // Log data export
      await this.auditLogger.logEvent({
        userId,
        event: 'data.export.completed',
        details: {
          requestId,
          exportFormat: 'JSON',
          exportSize: exportFile.size,
          expiresAt: exportFile.expiresAt
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'GDPR'
      });

      return {
        success: true,
        requestId,
        downloadUrl: exportFile.downloadUrl,
        expiresAt: exportFile.expiresAt,
        format: 'JSON',
        size: exportFile.size
      };

    } catch (error) {
      console.error('Data portability error:', error);
      throw error;
    }
  }

  // Privacy Impact Assessment
  async conductPrivacyImpactAssessment(userId, dataProcessing) {
    try {
      const assessment = {
        assessmentId: crypto.randomUUID(),
        userId,
        dataProcessing,
        assessmentDate: new Date().toISOString(),
        riskLevel: 'medium', // Will be calculated
        riskFactors: [],
        mitigationMeasures: [],
        complianceStatus: 'compliant'
      };

      // Assess risk factors
      assessment.riskFactors = await this.assessRiskFactors(dataProcessing);
      assessment.riskLevel = this.calculateRiskLevel(assessment.riskFactors);

      // Identify mitigation measures
      assessment.mitigationMeasures = await this.identifyMitigationMeasures(dataProcessing);

      // Check compliance
      assessment.complianceStatus = await this.checkComplianceStatus(dataProcessing);

      // Store assessment
      await require('../database/user-database').storePrivacyImpactAssessment(assessment);

      // Log assessment
      await this.auditLogger.logEvent({
        userId,
        event: 'privacy.impact.assessment',
        details: {
          assessmentId: assessment.assessmentId,
          riskLevel: assessment.riskLevel,
          complianceStatus: assessment.complianceStatus,
          riskFactors: assessment.riskFactors.length
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'GDPR'
      });

      return assessment;

    } catch (error) {
      console.error('Privacy impact assessment error:', error);
      throw error;
    }
  }

  // Data Retention Management
  async manageDataRetention() {
    try {
      const retentionRules = this.config.privacy.dataRetention;
      
      for (const [dataType, retentionPeriod] of Object.entries(retentionRules)) {
        if (retentionPeriod) {
          const expiredData = await this.dataRetentionManager.getExpiredData(dataType, retentionPeriod);
          
          for (const dataItem of expiredData) {
            await this.dataRetentionManager.archiveOrDeleteData(dataItem);
            
            // Log retention action
            await this.auditLogger.logEvent({
              event: 'data.retention.action',
              details: {
                dataType,
                dataItemId: dataItem.id,
                action: dataItem.action,
                retentionPeriod
              },
              timestamp: new Date().toISOString(),
              source: 'privacy_service',
              complianceType: 'GDPR'
            });
          }
        }
      }

      return { success: true, processed: Object.keys(retentionRules).length };

    } catch (error) {
      console.error('Data retention management error:', error);
      throw error;
    }
  }

  // Cross-Border Data Transfer
  async manageCrossBorderTransfer(userId, destinationCountry, dataCategories) {
    try {
      // Check adequacy decisions
      const adequacyStatus = this.checkAdequacyDecision(destinationCountry);
      
      // Assess transfer risks
      const transferRisks = await this.assessTransferRisks(destinationCountry, dataCategories);
      
      // Apply safeguards if needed
      const safeguards = await this.applyTransferSafeguards(
        destinationCountry, 
        dataCategories, 
        transferRisks
      );

      // Log transfer
      await this.auditLogger.logEvent({
        userId,
        event: 'data.transfer.cross_border',
        details: {
          destinationCountry,
          dataCategories,
          adequacyStatus,
          transferRisks,
          safeguards,
          timestamp: new Date().toISOString()
        },
        timestamp: new Date().toISOString(),
        source: 'privacy_service',
        complianceType: 'GDPR'
      });

      return {
        success: true,
        adequacyStatus,
        transferAllowed: adequacyStatus === 'adequate' || safeguards.length > 0,
        safeguards,
        transferDate: new Date().toISOString()
      };

    } catch (error) {
      console.error('Cross-border transfer error:', error);
      throw error;
    }
  }

  // Helper Methods
  calculateConsentExpiry(consentType) {
    const expiryMap = {
      data_processing: 365, // 1 year
      marketing: 180,      // 6 months
      analytics: 90,       // 3 months
      cookies: 30          // 1 month
    };

    const days = expiryMap[consentType] || 365;
    return new Date(Date.now() + (days * 24 * 60 * 60 * 1000)).toISOString();
  }

  hashIP(ipAddress) {
    return crypto.createHash('sha256').update(ipAddress).digest('hex');
  }

  getAffectedProcessing(consentType) {
    const processingMap = {
      data_processing: ['user_profiling', 'personalization', 'analytics'],
      marketing: ['email_marketing', 'targeted_ads', 'communication'],
      analytics: ['usage_analytics', 'performance_monitoring', 'user_behavior']
    };
    
    return processingMap[consentType] || [];
  }

  async compileUserData(userId) {
    // Compile all user data across systems
    const profile = await require('../database/user-database').getUserProfile(userId);
    const preferences = await require('../database/user-database').getUserPreferences(userId);
    const activity = await require('../database/user-database').getUserActivity(userId);
    const compliance = await require('../database/user-database').getUserCompliance(userId);

    return { profile, preferences, activity, compliance };
  }

  calculateDSARCompletion() {
    return new Date(Date.now() + (30 * 24 * 60 * 60 * 1000)).toISOString(); // 30 days
  }

  getRequiredFieldsForPurpose(purpose) {
    const fieldMap = {
      authentication: ['email', 'user_id', 'role'],
      medical_care: ['user_id', 'role', 'specialty', 'medical_license'],
      administrative: ['user_id', 'role', 'name', 'institution']
    };
    
    return fieldMap[purpose] || [];
  }

  isRequiredForCompliance(field) {
    const complianceFields = ['user_id', 'email', 'role', 'audit_logs', 'consent_records'];
    return complianceFields.includes(field);
  }

  canAnonymize(field) {
    const anonymizableFields = ['phone_number', 'address', 'emergency_contact'];
    return anonymizableFields.includes(field);
  }

  async anonymizeUserField(userId, field) {
    const anonymizedValue = `ANONYMIZED_${crypto.randomBytes(8).toString('hex')}`;
    await require('../database/user-database').updateUserField(userId, field, anonymizedValue);
  }

  async removeUserField(userId, field) {
    await require('../database/user-database').removeUserField(userId, field);
  }

  filterMinimumNecessaryData(data, action, userRoles, specialty) {
    // Implement minimum necessary filtering logic
    const filtered = { ...data };
    
    // Remove unnecessary sensitive data based on role and action
    if (!userRoles.some(role => ['doctor', 'senior_doctor'].includes(role.role_name))) {
      // Remove detailed medical history for non-doctors
      delete filtered.medicalHistory;
      delete filtered.medications;
    }
    
    return filtered;
  }

  async applyHIPAAccessRestrictions(userId, dataContext) {
    // Apply time-based and purpose-based restrictions
    await require('../database/user-database').createAccessRestriction({
      userId,
      patientId: dataContext.patientId,
      restrictionType: 'time_based',
      expiresAt: new Date(Date.now() + (4 * 60 * 60 * 1000)).toISOString(), // 4 hours
      createdAt: new Date().toISOString()
    });
  }

  async validateErasureRequest(userId, scope) {
    // Check if erasure is legally required or permitted
    // Some data must be retained for legal/medical reasons
    const protectedData = await this.getProtectedDataCategories();
    
    if (scope === 'full' && protectedData.some(cat => cat.mandatory)) {
      throw new Error('Full erasure not permitted due to legal/medical requirements');
    }
  }

  async createErasurePlan(userId, scope) {
    // Create detailed plan for data erasure
    const dataSets = [
      { name: 'profile_data', scope: scope === 'full' ? 'partial' : 'full' },
      { name: 'activity_logs', scope: 'full' },
      { name: 'consent_records', scope: scope === 'full' ? 'partial' : 'full' },
      { name: 'audit_trails', scope: 'legal_only' }
    ];
    
    return { dataSets };
  }

  async eraseDataSet(userId, dataSet) {
    switch (dataSet.scope) {
      case 'full':
        await require('../database/user-database').deleteUserDataSet(userId, dataSet.name);
        break;
      case 'partial':
        await require('../database/user-database').anonymizeUserDataSet(userId, dataSet.name);
        break;
      case 'legal_only':
        await require('../database/user-database').restrictUserDataSet(userId, dataSet.name);
        break;
    }
    
    return { name: dataSet.name, action: dataSet.scope };
  }

  async createSecureExport(data) {
    const fileName = `user_export_${crypto.randomBytes(16).toString('hex')}.json`;
    const filePath = `/secure/exports/${fileName}`;
    
    // In production, this would write to secure file storage
    console.log(`Creating secure export: ${filePath}`);
    
    return {
      filePath,
      downloadUrl: `/api/export/download/${fileName}`,
      expiresAt: new Date(Date.now() + (7 * 24 * 60 * 60 * 1000)).toISOString(), // 7 days
      size: JSON.stringify(data).length
    };
  }

  async assessRiskFactors(dataProcessing) {
    // Analyze risk factors for privacy impact
    return [
      { factor: 'data_volume', level: 'medium' },
      { factor: 'data_sensitivity', level: 'high' },
      { factor: 'processing_purpose', level: 'low' }
    ];
  }

  calculateRiskLevel(riskFactors) {
    const highRisks = riskFactors.filter(f => f.level === 'high').length;
    const mediumRisks = riskFactors.filter(f => f.level === 'medium').length;
    
    if (highRisks > 0) return 'high';
    if (mediumRisks > 2) return 'high';
    if (mediumRisks > 0) return 'medium';
    return 'low';
  }

  async identifyMitigationMeasures(dataProcessing) {
    return [
      { measure: 'data_encryption', applicable: true },
      { measure: 'access_controls', applicable: true },
      { measure: 'data_minimization', applicable: true }
    ];
  }

  async checkComplianceStatus(dataProcessing) {
    return 'compliant'; // Simplified - implement full compliance check
  }

  checkAdequacyDecision(country) {
    // Check EU adequacy decisions
    const adequateCountries = ['US', 'CA', 'CH', 'UK', 'JP', 'KR'];
    return adequateCountries.includes(country) ? 'adequate' : 'inadequate';
  }

  async assessTransferRisks(country, dataCategories) {
    return [{ risk: 'regulatory_gap', level: 'medium' }];
  }

  async applyTransferSafeguards(country, dataCategories, risks) {
    return [{ safeguard: 'standard_contractual_clauses', applied: true }];
  }

  async getProtectedDataCategories() {
    return [
      { category: 'audit_trails', mandatory: true },
      { category: 'medical_records', mandatory: true },
      { category: 'compliance_data', mandatory: true }
    ];
  }

  async updateUserPrivacyPreferences(userId, preferences) {
    await require('../database/user-database').updatePrivacyPreferences(userId, preferences);
  }

  async handleConsentWithdrawal(userId, consentType) {
    await this.implementDataMinimization(userId, `withdrawal_${consentType}`);
  }
}

module.exports = HealthcarePrivacyService;