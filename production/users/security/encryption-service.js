// Healthcare Security Framework and Encryption Service
// Production-grade security implementation for healthcare data protection

const crypto = require('crypto');
const bcrypt = require('bcrypt');
const config = require('../config/user-management-config');

class HealthcareSecurityService {
  constructor() {
    this.config = config.security;
    this.auditLogger = require('../monitoring/audit-logger');
    this.keyRotationSchedule = null;
    this.initializeSecurity();
  }

  // Initialize Security Framework
  initializeSecurity() {
    this.setupKeyRotation();
    this.validateEncryptionSettings();
    console.log('Healthcare Security Service initialized');
  }

  // Data Encryption/Decryption
  async encryptData(data, options = {}) {
    try {
      const algorithm = options.algorithm || this.config.encryption.algorithm;
      const key = await this.getEncryptionKey(options.keyId);
      const iv = crypto.randomBytes(16);
      const cipher = crypto.createCipher(algorithm, key, { iv });

      let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
      encrypted += cipher.final('hex');

      const encryptedPackage = {
        encryptedData: encrypted,
        iv: iv.toString('hex'),
        algorithm,
        keyId: options.keyId || 'current',
        timestamp: new Date().toISOString()
      };

      // Log encryption event
      await this.auditLogger.logSecurityEvent({
        event: 'data.encrypted',
        details: {
          dataType: options.dataType || 'unknown',
          algorithm,
          keyId: encryptedPackage.keyId
        },
        severity: 'low',
        source: 'security_service'
      });

      return encryptedPackage;

    } catch (error) {
      console.error('Encryption error:', error);
      throw new Error(`Encryption failed: ${error.message}`);
    }
  }

  async decryptData(encryptedPackage) {
    try {
      const { encryptedData, iv, algorithm, keyId } = encryptedPackage;
      const key = await this.getEncryptionKey(keyId);

      const decipher = crypto.createDecipher(algorithm, key, Buffer.from(iv, 'hex'));
      let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
      decrypted += decipher.final('utf8');

      const data = JSON.parse(decrypted);

      // Log decryption event
      await this.auditLogger.logSecurityEvent({
        event: 'data.decrypted',
        details: {
          dataType: encryptedPackage.dataType || 'unknown',
          algorithm,
          keyId
        },
        severity: 'low',
        source: 'security_service'
      });

      return data;

    } catch (error) {
      console.error('Decryption error:', error);
      throw new Error(`Decryption failed: ${error.message}`);
    }
  }

  // PHI (Protected Health Information) Encryption
  async encryptPHI(phiData, patientId, userId) {
    try {
      // Add patient-specific salt for additional protection
      const patientSalt = await this.generatePatientSalt(patientId);
      
      const encryptedPHI = await this.encryptData(phiData, {
        algorithm: 'aes-256-gcm',
        keyId: 'phi_encryption',
        additionalContext: {
          patientId,
          encryptedBy: userId,
          phiType: this.determinePHIType(phiData)
        }
      });

      // Create integrity hash for PHI verification
      const integrityHash = this.createIntegrityHash(phiData, patientSalt);

      const phiPackage = {
        ...encryptedPHI,
        patientId,
        integrityHash,
        phiType: encryptedPHI.additionalContext.phiType,
        encryptedBy: userId,
        encryptedAt: new Date().toISOString()
      };

      // Log PHI encryption
      await this.auditLogger.logSecurityEvent({
        userId,
        event: 'phi.encrypted',
        details: {
          patientId,
          phiType: phiPackage.phiType,
          dataSize: JSON.stringify(phiData).length
        },
        severity: 'medium',
        source: 'security_service',
        complianceType: 'HIPAA'
      });

      return phiPackage;

    } catch (error) {
      console.error('PHI encryption error:', error);
      throw error;
    }
  }

  async decryptPHI(phiPackage, userId, accessContext) {
    try {
      // Verify user has access to this PHI
      await this.verifyPHIAccess(userId, phiPackage.patientId, accessContext);

      // Decrypt the data
      const decryptedPHI = await this.decryptData(phiPackage);

      // Verify integrity
      const patientSalt = await this.generatePatientSalt(phiPackage.patientId);
      const expectedHash = this.createIntegrityHash(decryptedPHI, patientSalt);
      
      if (expectedHash !== phiPackage.integrityHash) {
        throw new Error('PHI integrity check failed - data may have been tampered');
      }

      // Log PHI access
      await this.auditLogger.logSecurityEvent({
        userId,
        event: 'phi.accessed',
        details: {
          patientId: phiPackage.patientId,
          phiType: phiPackage.phiType,
          accessContext,
          accessReason: accessContext.reason
        },
        severity: 'medium',
        source: 'security_service',
        complianceType: 'HIPAA'
      });

      return {
        data: decryptedPHI,
        metadata: {
          patientId: phiPackage.patientId,
          phiType: phiPackage.phiType,
          encryptedAt: phiPackage.encryptedAt,
          encryptedBy: phiPackage.encryptedBy,
          accessGranted: true
        }
      };

    } catch (error) {
      console.error('PHI decryption error:', error);
      
      // Log unauthorized access attempt
      await this.auditLogger.logSecurityEvent({
        userId,
        event: 'phi.access.denied',
        details: {
          patientId: phiPackage.patientId,
          reason: error.message,
          accessContext
        },
        severity: 'high',
        source: 'security_service',
        complianceType: 'HIPAA'
      });

      throw error;
    }
  }

  // Password Security
  async hashPassword(password, options = {}) {
    try {
      const saltRounds = options.saltRounds || 12;
      const salt = await bcrypt.genSalt(saltRounds);
      const hash = await bcrypt.hash(password, salt);

      // Log password hashing
      await this.auditLogger.logSecurityEvent({
        event: 'password.hashed',
        details: {
          saltRounds,
          algorithm: 'bcrypt'
        },
        severity: 'low',
        source: 'security_service'
      });

      return {
        hash,
        salt,
        algorithm: 'bcrypt',
        createdAt: new Date().toISOString()
      };

    } catch (error) {
      console.error('Password hashing error:', error);
      throw error;
    }
  }

  async verifyPassword(password, passwordHash) {
    try {
      const isValid = await bcrypt.compare(password, passwordHash.hash);

      // Log password verification attempt
      await this.auditLogger.logSecurityEvent({
        event: 'password.verification',
        details: {
          success: isValid,
          algorithm: passwordHash.algorithm
        },
        severity: isValid ? 'low' : 'medium',
        source: 'security_service'
      });

      return isValid;

    } catch (error) {
      console.error('Password verification error:', error);
      return false;
    }
  }

  // API Security
  async generateAPISignature(payload, secretKey, options = {}) {
    try {
      const timestamp = options.timestamp || Date.now();
      const nonce = options.nonce || crypto.randomBytes(16).toString('hex');
      const method = options.method || 'POST';
      const url = options.url;

      // Create signature base string
      const baseString = `${method}\n${url}\n${timestamp}\n${nonce}\n${JSON.stringify(payload)}`;
      
      const signature = crypto
        .createHmac('sha256', secretKey)
        .update(baseString)
        .digest('hex');

      const apiSignature = {
        signature,
        timestamp,
        nonce,
        method,
        url,
        expiresAt: new Date(timestamp + (options.expiry || 300000)).toISOString() // 5 minutes default
      };

      // Log API signature generation
      await this.auditLogger.logSecurityEvent({
        event: 'api.signature.generated',
        details: {
          method,
          url,
          expiresAt: apiSignature.expiresAt
        },
        severity: 'low',
        source: 'security_service'
      });

      return apiSignature;

    } catch (error) {
      console.error('API signature generation error:', error);
      throw error;
    }
  }

  async verifyAPISignature(signature, payload, secretKey) {
    try {
      const { timestamp, nonce, method, url, expiresAt } = signature;

      // Check expiry
      if (new Date(expiresAt) < new Date()) {
        throw new Error('API signature has expired');
      }

      // Check timestamp validity (prevent replay attacks)
      const now = Date.now();
      if (Math.abs(now - timestamp) > 300000) { // 5 minutes tolerance
        throw new Error('API signature timestamp is invalid');
      }

      // Verify signature
      const baseString = `${method}\n${url}\n${timestamp}\n${nonce}\n${JSON.stringify(payload)}`;
      const expectedSignature = crypto
        .createHmac('sha256', secretKey)
        .update(baseString)
        .digest('hex');

      const isValid = signature.signature === expectedSignature;

      // Log verification result
      await this.auditLogger.logSecurityEvent({
        event: 'api.signature.verification',
        details: {
          success: isValid,
          method,
          url,
          timestamp
        },
        severity: isValid ? 'low' : 'high',
        source: 'security_service'
      });

      return isValid;

    } catch (error) {
      console.error('API signature verification error:', error);
      
      await this.auditLogger.logSecurityEvent({
        event: 'api.signature.verification.failed',
        details: {
          error: error.message,
          method,
          url
        },
        severity: 'high',
        source: 'security_service'
      });

      return false;
    }
  }

  // Security Monitoring and Intrusion Detection
  async detectIntrusionAttempt(ipAddress, userAgent, requestData) {
    try {
      const intrusionIndicators = {
        failedAttempts: await this.getFailedAttempts(ipAddress),
        suspiciousPatterns: await this.analyzeSuspiciousPatterns(ipAddress, requestData),
        velocityCheck: await this.performVelocityCheck(ipAddress),
        reputationCheck: await this.checkIPReputation(ipAddress),
        anomalyScore: 0
      };

      // Calculate intrusion risk score
      intrusionIndicators.anomalyScore = this.calculateIntrusionRiskScore(intrusionIndicators);

      // Determine if intrusion attempt
      const isIntrusionAttempt = intrusionIndicators.anomalyScore > 70;
      
      if (isIntrusionAttempt) {
        await this.handleIntrusionAttempt(ipAddress, intrusionIndicators, requestData);
      }

      return {
        isIntrusionAttempt,
        intrusionIndicators,
        riskScore: intrusionIndicators.anomalyScore,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('Intrusion detection error:', error);
      return {
        isIntrusionAttempt: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  async blockSuspiciousIP(ipAddress, reason, duration = 3600) {
    try {
      const blockRecord = {
        ipAddress,
        reason,
        blockedAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + (duration * 1000)).toISOString(),
        blockType: 'automatic',
        status: 'active'
      };

      // Store block record
      await this.storeIPBlock(blockRecord);

      // Log IP block
      await this.auditLogger.logSecurityEvent({
        event: 'ip.blocked',
        details: {
          ipAddress,
          reason,
          duration,
          blockType: 'automatic'
        },
        severity: 'high',
        source: 'security_service'
      });

      return blockRecord;

    } catch (error) {
      console.error('IP blocking error:', error);
      throw error;
    }
  }

  // Security Incident Response
  async createSecurityIncident(incidentData) {
    try {
      const incident = {
        incidentId: crypto.randomUUID(),
        severity: this.determineIncidentSeverity(incidentData),
        status: 'open',
        createdAt: new Date().toISOString(),
        ...incidentData
      };

      // Store incident
      await this.storeSecurityIncident(incident);

      // Log incident creation
      await this.auditLogger.logSecurityEvent({
        event: 'security.incident.created',
        details: {
          incidentId: incident.incidentId,
          severity: incident.severity,
          type: incident.type,
          description: incident.description
        },
        severity: incident.severity,
        source: 'security_service'
      });

      // Trigger incident response workflow
      await this.triggerIncidentResponse(incident);

      return incident;

    } catch (error) {
      console.error('Security incident creation error:', error);
      throw error;
    }
  }

  // Key Management
  async rotateEncryptionKeys() {
    try {
      const newKeyId = crypto.randomUUID();
      const newKey = crypto.randomBytes(32);

      // Store new key
      await this.storeEncryptionKey(newKeyId, newKey);

      // Update current key reference
      await this.updateCurrentKeyId(newKeyId);

      // Schedule re-encryption of sensitive data with new key
      await this.scheduleDataReencryption(newKeyId);

      // Log key rotation
      await this.auditLogger.logSecurityEvent({
        event: 'encryption.key.rotated',
        details: {
          newKeyId,
          previousKeyId: 'previous_current_key',
          rotationDate: new Date().toISOString()
        },
        severity: 'medium',
        source: 'security_service'
      });

      return {
        success: true,
        newKeyId,
        rotationDate: new Date().toISOString()
      };

    } catch (error) {
      console.error('Key rotation error:', error);
      throw error;
    }
  }

  // Helper Methods
  async getEncryptionKey(keyId = 'current') {
    // In production, this would retrieve from secure key management system
    // For now, using environment variable or generating a key
    let key;
    
    if (keyId === 'current') {
      key = process.env.ENCRYPTION_KEY || crypto.randomBytes(32);
    } else {
      key = await this.retrieveKeyById(keyId);
    }

    return key;
  }

  async generatePatientSalt(patientId) {
    const patientKey = process.env.PATIENT_SALT_KEY || 'default_patient_salt';
    return crypto.createHash('sha256')
      .update(`${patientKey}:${patientId}`)
      .digest('hex');
  }

  determinePHIType(phiData) {
    if (phiData.medicalHistory) return 'medical_history';
    if (phiData.medications) return 'medications';
    if (phiData.labResults) return 'lab_results';
    if (phiData.imaging) return 'imaging';
    return 'general_phi';
  }

  createIntegrityHash(data, salt) {
    const dataString = typeof data === 'string' ? data : JSON.stringify(data);
    return crypto.createHash('sha256')
      .update(`${salt}:${dataString}`)
      .digest('hex');
  }

  async verifyPHIAccess(userId, patientId, context) {
    const rbac = require('../rbac/healthcare-rbac-system');
    
    // Check if user has permission to access this patient's data
    const hasAccess = await rbac.checkPermission(userId, 'patients.read');
    
    if (!hasAccess) {
      throw new Error('User does not have permission to access patient data');
    }

    // Additional context-specific checks
    if (context.emergency) {
      // Emergency access - log special access
      return true;
    }

    return true;
  }

  setupKeyRotation() {
    const rotationInterval = this.config.encryption.keyRotation * 1000; // Convert to milliseconds
    
    this.keyRotationSchedule = setInterval(async () => {
      try {
        await this.rotateEncryptionKeys();
      } catch (error) {
        console.error('Scheduled key rotation failed:', error);
      }
    }, rotationInterval);
  }

  validateEncryptionSettings() {
    const requiredSettings = ['algorithm', 'keyRotation'];
    const missingSettings = requiredSettings.filter(setting => 
      !this.config.encryption[setting]
    );

    if (missingSettings.length > 0) {
      throw new Error(`Missing encryption settings: ${missingSettings.join(', ')}`);
    }
  }

  async getFailedAttempts(ipAddress) {
    // Query database for failed attempts from this IP
    const recentAttempts = await require('../database/user-database').getEvents({
      eventType: 'user.failed_login',
      startDate: new Date(Date.now() - (60 * 60 * 1000)).toISOString(), // Last hour
      limit: 100
    });

    return recentAttempts.filter(attempt => 
      attempt.details.ipAddress === ipAddress
    ).length;
  }

  async analyzeSuspiciousPatterns(ipAddress, requestData) {
    const patterns = [];
    
    // Check for SQL injection patterns
    if (this.containsSQLInjection(requestData)) {
      patterns.push('sql_injection');
    }

    // Check for XSS patterns
    if (this.containsXSS(requestData)) {
      patterns.push('xss');
    }

    // Check for path traversal
    if (this.containsPathTraversal(requestData)) {
      patterns.push('path_traversal');
    }

    return patterns;
  }

  containsSQLInjection(data) {
    const sqlPatterns = [
      /(\%27)|(\')|(\-\-)|(\%23)|(#)/i,
      /((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))/i,
      /\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))/i
    ];

    const dataString = JSON.stringify(data);
    return sqlPatterns.some(pattern => pattern.test(dataString));
  }

  containsXSS(data) {
    const xssPatterns = [
      /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
      /javascript:/i,
      /on\w+\s*=/i
    ];

    const dataString = JSON.stringify(data);
    return xssPatterns.some(pattern => pattern.test(dataString));
  }

  containsPathTraversal(data) {
    const traversalPatterns = [
      /\.\.\//,
      /%2e%2e%2f/gi,
      /etc\/passwd/gi
    ];

    const dataString = JSON.stringify(data);
    return traversalPatterns.some(pattern => pattern.test(dataString));
  }

  async performVelocityCheck(ipAddress) {
    // Check request velocity from this IP
    const recentRequests = await require('../database/user-database').getEvents({
      eventType: 'api.request',
      startDate: new Date(Date.now() - (60 * 1000)).toISOString(), // Last minute
      limit: 1000
    });

    const ipRequests = recentRequests.filter(req => 
      req.details.ipAddress === ipAddress
    ).length;

    return {
      requestsPerMinute: ipRequests,
      threshold: 100,
      isExcessive: ipRequests > 100
    };
  }

  async checkIPReputation(ipAddress) {
    // In production, integrate with IP reputation services
    // For now, basic checks
    const isPrivate = ipAddress.startsWith('192.168.') || 
                     ipAddress.startsWith('10.') || 
                     ipAddress.startsWith('172.');
    
    return {
      reputation: isPrivate ? 'private' : 'unknown',
      isWhitelisted: isPrivate,
      isBlacklisted: false
    };
  }

  calculateIntrusionRiskScore(indicators) {
    let score = 0;

    // Failed attempts factor
    if (indicators.failedAttempts > 5) score += 30;
    else if (indicators.failedAttempts > 2) score += 15;

    // Suspicious patterns factor
    score += indicators.suspiciousPatterns.length * 20;

    // Velocity factor
    if (indicators.velocityCheck.isExcessive) score += 25;

    // Reputation factor
    if (indicators.reputationCheck.isBlacklisted) score += 50;

    return Math.min(score, 100);
  }

  async handleIntrusionAttempt(ipAddress, indicators, requestData) {
    // Block the IP
    await this.blockSuspiciousIP(
      ipAddress,
      'intrusion_attempt_detected',
      3600 // 1 hour block
    );

    // Create security incident
    await this.createSecurityIncident({
      type: 'intrusion_attempt',
      severity: 'high',
      description: `Intrusion attempt detected from IP ${ipAddress}`,
      sourceIP: ipAddress,
      requestData,
      indicators
    });
  }

  async storeIPBlock(blockRecord) {
    // Store in database
    await require('../database/user-database').createIPBlock(blockRecord);
  }

  determineIncidentSeverity(incidentData) {
    const criticalIncidents = ['data_breach', 'system_compromise', 'unauthorized_access'];
    const highSeverityIncidents = ['malware_detection', 'ddos_attempt', 'privilege_escalation'];

    if (criticalIncidents.includes(incidentData.type)) {
      return 'critical';
    }

    if (highSeverityIncidents.includes(incidentData.type)) {
      return 'high';
    }

    return 'medium';
  }

  async storeSecurityIncident(incident) {
    await require('../database/user-database').createSecurityIncident(incident);
  }

  async triggerIncidentResponse(incident) {
    // Notify security team
    await this.notifySecurityTeam(incident);

    // Auto-remediation if possible
    if (incident.type === 'ddos_attempt') {
      await this.enableDDoSProtection();
    }
  }

  async notifySecurityTeam(incident) {
    // Implementation for security team notification
    console.log(`Security incident ${incident.incidentId} created: ${incident.description}`);
  }

  async enableDDoSProtection() {
    // Enable DDoS protection measures
    console.log('DDoS protection enabled');
  }

  async storeEncryptionKey(keyId, key) {
    // Store key securely (in production, use HSM or cloud KMS)
    await require('../database/user-database').storeEncryptionKey(keyId, key);
  }

  async retrieveKeyById(keyId) {
    return await require('../database/user-database').retrieveEncryptionKey(keyId);
  }

  async updateCurrentKeyId(keyId) {
    await require('../database/user-database').updateCurrentKeyId(keyId);
  }

  async scheduleDataReencryption(newKeyId) {
    // Schedule re-encryption of sensitive data with new key
    await require('../database/user-database').scheduleDataReencryption(newKeyId);
  }

  // Cleanup
  shutdown() {
    if (this.keyRotationSchedule) {
      clearInterval(this.keyRotationSchedule);
    }
  }
}

module.exports = new HealthcareSecurityService();