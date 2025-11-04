/**
 * HIPAA-Compliant Comprehensive Audit Logging System
 * Implements tamper-proof audit trails for all PHI access and system events
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class AuditLogger {
  constructor(config = {}) {
    this.config = {
      logDirectory: config.logDirectory || './logs',
      retentionPeriod: config.retentionPeriod || 2555 * 24 * 60 * 60 * 1000, // 7 years for HIPAA
      maxLogSize: config.maxLogSize || 100 * 1024 * 1024, // 100MB
      encryptionKey: config.encryptionKey || process.env.AUDIT_LOG_ENCRYPTION_KEY,
      hashAlgorithm: 'sha256',
      compressionEnabled: true,
      ...config
    };

    this.auditLogs = new Map();
    this.currentLogFiles = new Map();
    this.buffer = [];
    this.bufferSize = 0;
    this.maxBufferSize = 1000;
    this.isWriting = false;
    this.blockedFields = [
      'ssn', 'social_security_number', 'password', 'token', 'secret',
      'credit_card', 'bank_account', 'medical_record_number', 'biometric_data'
    ];

    this.initializeLogSystem();
  }

  /**
   * Initialize the audit logging system
   */
  async initializeLogSystem() {
    try {
      await fs.mkdir(this.config.logDirectory, { recursive: true });
      console.log(`[AUDIT] Audit logging system initialized at ${this.config.logDirectory}`);
    } catch (error) {
      console.error('[AUDIT] Failed to initialize log system:', error);
      throw error;
    }
  }

  /**
   * Log audit event with HIPAA compliance
   */
  async logEvent(event) {
    const timestamp = new Date().toISOString();
    const eventId = crypto.randomUUID();
    
    // Sanitize event data
    const sanitizedEvent = this.sanitizeEvent(event);
    
    // Create audit record
    const auditRecord = {
      id: eventId,
      timestamp,
      sequence: await this.getNextSequenceNumber(),
      hash: '',
      previousHash: await this.getLastRecordHash(),
      ...sanitizedEvent
    };

    // Generate cryptographic hash for integrity
    auditRecord.hash = this.generateHash(auditRecord);

    // Add to buffer for batch writing
    this.buffer.push(auditRecord);
    this.bufferSize++;

    // Write to appropriate log file
    await this.writeToLogFile(auditRecord);

    // Auto-flush buffer if full
    if (this.bufferSize >= this.maxBufferSize) {
      await this.flushBuffer();
    }

    return eventId;
  }

  /**
   * Sanitize event data to remove sensitive information
   */
  sanitizeEvent(event) {
    const sanitized = JSON.parse(JSON.stringify(event)); // Deep clone

    // Remove or mask sensitive fields
    this.removeSensitiveData(sanitized, sanitized.userData);
    this.removeSensitiveData(sanitized, sanitized.patientData);
    this.removeSensitiveData(sanitized, sanitized.phiData);

    // Ensure required fields are present
    if (!sanitized.userId && sanitized.userId !== 0) {
      sanitized.userId = 'system';
    }
    if (!sanitized.action) {
      sanitized.action = 'unknown';
    }
    if (!sanitized.resource) {
      sanitized.resource = 'unknown';
    }

    // Set default severity if not specified
    if (!sanitized.severity) {
      sanitized.severity = this.calculateSeverity(sanitized);
    }

    // Set category if not specified
    if (!sanitized.category) {
      sanitized.category = this.categorizeEvent(sanitized);
    }

    // Add location information
    sanitized.location = {
      ipAddress: sanitized.ipAddress || '',
      userAgent: sanitized.userAgent || '',
      endpoint: sanitized.endpoint || '',
      sessionId: sanitized.sessionId || ''
    };

    // Add compliance metadata
    sanitized.compliance = {
      hipaa_relevant: this.isHIPAARelevant(sanitized),
      retention_period: this.config.retentionPeriod,
      encryption_applied: !!this.config.encryptionKey,
      legal_hold: false
    };

    return sanitized;
  }

  /**
   * Remove or mask sensitive data
   */
  removeSensitiveData(obj, data) {
    if (!data || typeof data !== 'object') return;

    for (const key in data) {
      const lowerKey = key.toLowerCase();
      
      // Check if field is in blocked list
      if (this.blockedFields.some(field => lowerKey.includes(field))) {
        data[key] = '[REDACTED]';
        continue;
      }

      // Check for sensitive patterns
      if (this.isSensitiveField(lowerKey, data[key])) {
        data[key] = this.maskSensitiveData(lowerKey, data[key]);
      }

      // Recursively process nested objects
      if (typeof data[key] === 'object' && data[key] !== null) {
        this.removeSensitiveData(data, data[key]);
      }
    }
  }

  /**
   * Check if field contains sensitive data
   */
  isSensitiveField(fieldName, value) {
    if (typeof value !== 'string') return false;

    const sensitivePatterns = [
      /^\d{3}-\d{2}-\d{4}$/, // SSN pattern
      /^\d{16}$/, // Credit card pattern
      /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/, // Email pattern
      /\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/, // Credit card with spaces
      /\b\d{3}-\d{3}-\d{4}\b/, // Phone number
      /^Bearer\s+/, // JWT tokens
      /^Basic\s+/ // Basic auth tokens
    ];

    return sensitivePatterns.some(pattern => pattern.test(value));
  }

  /**
   * Mask sensitive data based on field type
   */
  maskSensitiveData(fieldName, value) {
    if (typeof value !== 'string') return '[REDACTED]';

    if (fieldName.includes('ssn') || fieldName.includes('social_security')) {
      return 'XXX-XX-' + value.slice(-4);
    }
    
    if (fieldName.includes('email')) {
      const [local, domain] = value.split('@');
      return local.charAt(0) + '***@' + domain;
    }
    
    if (fieldName.includes('phone') || fieldName.includes('telephone')) {
      return '(XXX) XXX-' + value.slice(-4);
    }
    
    if (fieldName.includes('credit_card') || fieldName.includes('card_number')) {
      return '**** **** **** ' + value.slice(-4);
    }
    
    return '[REDACTED]';
  }

  /**
   * Calculate event severity based on event type
   */
  calculateSeverity(event) {
    const highSeverityActions = [
      'data_breach', 'unauthorized_access', 'phi_exported', 'system_compromised',
      'admin_login', 'configuration_changed', 'backup_restored', 'account_locked',
      'multiple_failed_logins', 'after_hours_access'
    ];

    const mediumSeverityActions = [
      'phi_accessed', 'data_modified', 'user_created', 'role_assigned',
      'password_changed', 'session_expired', 'permission_denied'
    ];

    if (highSeverityActions.some(action => event.action.includes(action))) {
      return 'HIGH';
    }
    
    if (mediumSeverityActions.some(action => event.action.includes(action))) {
      return 'MEDIUM';
    }

    return 'LOW';
  }

  /**
   * Categorize event for compliance reporting
   */
  categorizeEvent(event) {
    if (event.action.includes('auth') || event.action.includes('login')) {
      return 'AUTHENTICATION';
    }
    
    if (event.action.includes('phi') || event.action.includes('patient')) {
      return 'PHI_ACCESS';
    }
    
    if (event.action.includes('data') || event.action.includes('record')) {
      return 'DATA_ACCESS';
    }
    
    if (event.action.includes('system') || event.action.includes('config')) {
      return 'SYSTEM';
    }
    
    if (event.action.includes('backup') || event.action.includes('restore')) {
      return 'BACKUP';
    }
    
    return 'GENERAL';
  }

  /**
   * Check if event is HIPAA-relevant
   */
  isHIPAARelevant(event) {
    const hipaaRelevantActions = [
      'phi_accessed', 'phi_read', 'phi_write', 'phi_export',
      'patient_viewed', 'patient_updated', 'medical_record',
      'appointment_accessed', 'prescription_accessed',
      'lab_result_accessed', 'diagnosis_accessed'
    ];

    return hipaaRelevantActions.some(action => event.action.includes(action));
  }

  /**
   * Generate cryptographic hash for audit record
   */
  generateHash(record) {
    const hashData = {
      id: record.id,
      timestamp: record.timestamp,
      sequence: record.sequence,
      userId: record.userId,
      action: record.action,
      resource: record.resource,
      previousHash: record.previousHash
    };

    const dataString = JSON.stringify(hashData);
    return crypto.createHash(this.config.hashAlgorithm).update(dataString).digest('hex');
  }

  /**
   * Get next sequence number for audit log
   */
  async getNextSequenceNumber() {
    const sequenceFile = path.join(this.config.logDirectory, 'sequence.txt');
    try {
      const data = await fs.readFile(sequenceFile, 'utf8');
      const lastSequence = parseInt(data) || 0;
      const nextSequence = lastSequence + 1;
      
      await fs.writeFile(sequenceFile, nextSequence.toString());
      return nextSequence;
    } catch (error) {
      // File doesn't exist or can't be read, start with 1
      await fs.writeFile(sequenceFile, '1');
      return 1;
    }
  }

  /**
   * Get hash of last audit record
   */
  async getLastRecordHash() {
    const indexFile = path.join(this.config.logDirectory, 'index.json');
    try {
      const data = await fs.readFile(indexFile, 'utf8');
      const index = JSON.parse(data);
      return index.lastHash || '';
    } catch (error) {
      return '';
    }
  }

  /**
   * Write audit record to log file
   */
  async writeToLogFile(record) {
    const logCategory = record.category || 'GENERAL';
    const date = new Date().toISOString().split('T')[0];
    const logFileName = `${logCategory.toLowerCase()}_${date}.log`;
    
    const logFilePath = path.join(this.config.logDirectory, logFileName);
    
    try {
      // Check if file rotation is needed
      const stats = await fs.stat(logFilePath).catch(() => null);
      if (stats && stats.size > this.config.maxLogSize) {
        await this.rotateLogFile(logFilePath);
      }

      // Encrypt if encryption key is provided
      let logData = JSON.stringify(record);
      if (this.config.encryptionKey) {
        logData = this.encryptLogData(logData);
      }

      // Append to log file
      await fs.appendFile(logFilePath, logData + '\n');
      
      // Update index
      await this.updateLogIndex(record);
      
    } catch (error) {
      console.error('[AUDIT] Failed to write to log file:', error);
      // In production, this should trigger alerts
    }
  }

  /**
   * Encrypt log data
   */
  encryptLogData(data) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher('aes-256-cbc', this.config.encryptionKey);
    
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    return iv.toString('hex') + ':' + encrypted;
  }

  /**
   * Rotate log file when size limit is reached
   */
  async rotateLogFile(logFilePath) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const rotatedPath = logFilePath.replace('.log', `_${timestamp}.log`);
    
    await fs.rename(logFilePath, rotatedPath);
    
    // In production, this should also trigger compression and archival
    console.log(`[AUDIT] Log file rotated: ${rotatedPath}`);
  }

  /**
   * Update audit log index
   */
  async updateLogIndex(record) {
    const indexFile = path.join(this.config.logDirectory, 'index.json');
    let index = {};
    
    try {
      const data = await fs.readFile(indexFile, 'utf8');
      index = JSON.parse(data);
    } catch (error) {
      // Index file doesn't exist, initialize empty index
    }
    
    index.lastHash = record.hash;
    index.lastUpdate = new Date().toISOString();
    index.totalRecords = (index.totalRecords || 0) + 1;
    
    await fs.writeFile(indexFile, JSON.stringify(index, null, 2));
  }

  /**
   * Flush audit log buffer
   */
  async flushBuffer() {
    if (this.isWriting || this.buffer.length === 0) return;
    
    this.isWriting = true;
    
    try {
      // Write buffer to persistent storage
      for (const record of this.buffer) {
        await this.writeToLogFile(record);
      }
      
      this.buffer = [];
      this.bufferSize = 0;
      
    } catch (error) {
      console.error('[AUDIT] Failed to flush buffer:', error);
    } finally {
      this.isWriting = false;
    }
  }

  /**
   * Query audit logs with filtering
   */
  async queryLogs(filters = {}) {
    const query = {
      userId: filters.userId,
      action: filters.action,
      category: filters.category,
      severity: filters.severity,
      startDate: filters.startDate ? new Date(filters.startDate) : null,
      endDate: filters.endDate ? new Date(filters.endDate) : null,
      ipAddress: filters.ipAddress,
      sessionId: filters.sessionId,
      resource: filters.resource
    };

    const results = [];
    const logFiles = await this.getLogFiles();

    for (const logFile of logFiles) {
      if (this.shouldIncludeLogFile(logFile, query)) {
        const records = await this.readLogFile(logFile);
        const filteredRecords = this.filterRecords(records, query);
        results.push(...filteredRecords);
      }
    }

    return results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }

  /**
   * Get list of audit log files
   */
  async getLogFiles() {
    try {
      const files = await fs.readdir(this.config.logDirectory);
      return files.filter(file => file.endsWith('.log')).map(file => ({
        name: file,
        path: path.join(this.config.logDirectory, file),
        category: file.split('_')[0].toUpperCase()
      }));
    } catch (error) {
      console.error('[AUDIT] Failed to get log files:', error);
      return [];
    }
  }

  /**
   * Check if log file should be included in query
   */
  shouldIncludeLogFile(logFile, query) {
    if (query.category && logFile.category !== query.category) {
      return false;
    }
    
    const fileDate = this.extractDateFromFileName(logFile.name);
    if (fileDate) {
      if (query.startDate && fileDate < query.startDate) return false;
      if (query.endDate && fileDate > query.endDate) return false;
    }
    
    return true;
  }

  /**
   * Extract date from log file name
   */
  extractDateFromFileName(fileName) {
    const match = fileName.match(/(\d{4}-\d{2}-\d{2})/);
    return match ? new Date(match[1]) : null;
  }

  /**
   * Read and decrypt log file
   */
  async readLogFile(logFile) {
    try {
      const content = await fs.readFile(logFile.path, 'utf8');
      const lines = content.trim().split('\n').filter(line => line.trim());
      
      return lines.map(line => {
        try {
          let data = line;
          if (this.config.encryptionKey) {
            data = this.decryptLogData(data);
          }
          return JSON.parse(data);
        } catch (error) {
          console.warn('[AUDIT] Failed to parse log line:', error);
          return null;
        }
      }).filter(record => record !== null);
      
    } catch (error) {
      console.error('[AUDIT] Failed to read log file:', error);
      return [];
    }
  }

  /**
   * Decrypt log data
   */
  decryptLogData(encryptedData) {
    if (!encryptedData.includes(':')) return encryptedData;
    
    try {
      const [ivHex, encrypted] = encryptedData.split(':');
      const iv = Buffer.from(ivHex, 'hex');
      const decipher = crypto.createDecipher('aes-256-cbc', this.config.encryptionKey);
      
      let decrypted = decipher.update(encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      
      return decrypted;
    } catch (error) {
      console.error('[AUDIT] Failed to decrypt log data:', error);
      return encryptedData;
    }
  }

  /**
   * Filter audit records based on query criteria
   */
  filterRecords(records, query) {
    return records.filter(record => {
      if (query.userId && record.userId !== query.userId) return false;
      if (query.action && !record.action.includes(query.action)) return false;
      if (query.severity && record.severity !== query.severity) return false;
      if (query.category && record.category !== query.category) return false;
      if (query.ipAddress && record.location?.ipAddress !== query.ipAddress) return false;
      if (query.sessionId && record.location?.sessionId !== query.sessionId) return false;
      if (query.resource && record.resource !== query.resource) return false;
      
      if (query.startDate && new Date(record.timestamp) < query.startDate) return false;
      if (query.endDate && new Date(record.timestamp) > query.endDate) return false;
      
      return true;
    });
  }

  /**
   * Generate compliance report
   */
  async generateComplianceReport(startDate, endDate) {
    const logs = await this.queryLogs({ startDate, endDate });
    
    const report = {
      reportId: crypto.randomUUID(),
      generatedAt: new Date().toISOString(),
      period: {
        start: startDate,
        end: endDate
      },
      summary: {
        totalEvents: logs.length,
        highSeverityEvents: logs.filter(log => log.severity === 'HIGH').length,
        mediumSeverityEvents: logs.filter(log => log.severity === 'MEDIUM').length,
        phiAccessEvents: logs.filter(log => log.compliance?.hipaa_relevant).length,
        authenticationEvents: logs.filter(log => log.category === 'AUTHENTICATION').length
      },
      categories: this.aggregateByCategory(logs),
      topUsers: this.aggregateByUser(logs),
      topResources: this.aggregateByResource(logs),
      complianceMetrics: this.calculateComplianceMetrics(logs),
      recommendations: this.generateComplianceRecommendations(logs)
    };

    return report;
  }

  /**
   * Aggregate logs by category
   */
  aggregateByCategory(logs) {
    const categories = {};
    logs.forEach(log => {
      const category = log.category || 'UNKNOWN';
      categories[category] = (categories[category] || 0) + 1;
    });
    return categories;
  }

  /**
   * Aggregate logs by user
   */
  aggregateByUser(logs) {
    const users = {};
    logs.forEach(log => {
      const userId = log.userId || 'unknown';
      if (!users[userId]) {
        users[userId] = { total: 0, highSeverity: 0, phiAccess: 0 };
      }
      users[userId].total++;
      if (log.severity === 'HIGH') users[userId].highSeverity++;
      if (log.compliance?.hipaa_relevant) users[userId].phiAccess++;
    });
    return users;
  }

  /**
   * Aggregate logs by resource
   */
  aggregateByResource(logs) {
    const resources = {};
    logs.forEach(log => {
      const resource = log.resource || 'unknown';
      resources[resource] = (resources[resource] || 0) + 1;
    });
    return resources;
  }

  /**
   * Calculate compliance metrics
   */
  calculateComplianceMetrics(logs) {
    const phiEvents = logs.filter(log => log.compliance?.hipaa_relevant);
    const authenticatedEvents = logs.filter(log => log.category === 'AUTHENTICATION');
    
    return {
      totalPHIAccessEvents: phiEvents.length,
      authenticatedAccessPercentage: logs.length > 0 ? 
        (authenticatedEvents.length / logs.length * 100).toFixed(2) : 0,
      averageEventsPerDay: this.calculateAverageEventsPerDay(logs),
      suspiciousActivityDetected: this.detectSuspiciousActivity(logs).length
    };
  }

  /**
   * Calculate average events per day
   */
  calculateAverageEventsPerDay(logs) {
    if (logs.length === 0) return 0;
    
    const dates = logs.map(log => log.timestamp.split('T')[0]);
    const uniqueDates = [...new Set(dates)];
    
    return (logs.length / uniqueDates.length).toFixed(2);
  }

  /**
   * Detect suspicious activity patterns
   */
  detectSuspiciousActivity(logs) {
    const suspicious = [];
    
    // Detect multiple failed login attempts
    const failedLogins = logs.filter(log => log.action.includes('authentication_failed'));
    const userFailures = {};
    
    failedLogins.forEach(log => {
      userFailures[log.userId] = (userFailures[log.userId] || 0) + 1;
    });
    
    Object.entries(userFailures).forEach(([userId, count]) => {
      if (count >= 5) {
        suspicious.push({
          type: 'multiple_failed_logins',
          userId,
          count,
          severity: 'HIGH'
        });
      }
    });
    
    return suspicious;
  }

  /**
   * Generate compliance recommendations
   */
  generateComplianceRecommendations(logs) {
    const recommendations = [];
    
    const highSeverityCount = logs.filter(log => log.severity === 'HIGH').length;
    if (highSeverityCount > 10) {
      recommendations.push({
        priority: 'HIGH',
        category: 'SECURITY',
        recommendation: 'Review and investigate high-severity security events',
        count: highSeverityCount
      });
    }
    
    const phiEvents = logs.filter(log => log.compliance?.hipaa_relevant);
    const unauthenticatedPHI = phiEvents.filter(log => !log.location?.sessionId);
    if (unauthenticatedPHI.length > 0) {
      recommendations.push({
        priority: 'HIGH',
        category: 'HIPAA_COMPLIANCE',
        recommendation: 'Investigate PHI access without proper authentication',
        count: unauthenticatedPHI.length
      });
    }
    
    return recommendations;
  }

  /**
   * Archive old logs based on retention policy
   */
  async archiveOldLogs() {
    const cutoffDate = new Date(Date.now() - this.config.retentionPeriod);
    const logFiles = await this.getLogFiles();
    
    for (const logFile of logFiles) {
      const fileDate = this.extractDateFromFileName(logFile.name);
      if (fileDate && fileDate < cutoffDate) {
        await this.archiveLogFile(logFile);
      }
    }
  }

  /**
   * Archive log file
   */
  async archiveLogFile(logFile) {
    const archiveDir = path.join(this.config.logDirectory, 'archive');
    await fs.mkdir(archiveDir, { recursive: true });
    
    const archivePath = path.join(archiveDir, logFile.name);
    await fs.rename(logFile.path, archivePath);
    
    console.log(`[AUDIT] Log file archived: ${archivePath}`);
  }

  /**
   * Get audit log statistics
   */
  async getLogStatistics() {
    const logFiles = await this.getLogFiles();
    let totalSize = 0;
    let totalRecords = 0;
    
    for (const logFile of logFiles) {
      try {
        const stats = await fs.stat(logFile.path);
        totalSize += stats.size;
        
        const records = await this.readLogFile(logFile);
        totalRecords += records.length;
      } catch (error) {
        console.warn(`[AUDIT] Failed to get stats for ${logFile.name}:`, error);
      }
    }
    
    return {
      totalFiles: logFiles.length,
      totalSize,
      totalRecords,
      averageRecordSize: totalRecords > 0 ? totalSize / totalRecords : 0,
      oldestRecord: await this.getOldestRecord(),
      newestRecord: await this.getNewestRecord()
    };
  }

  /**
   * Get oldest record timestamp
   */
  async getOldestRecord() {
    const logFiles = await this.getLogFiles();
    let oldest = null;
    
    for (const logFile of logFiles) {
      const records = await this.readLogFile(logFile);
      if (records.length > 0) {
        const firstRecord = records[records.length - 1];
        if (!oldest || new Date(firstRecord.timestamp) < new Date(oldest)) {
          oldest = firstRecord.timestamp;
        }
      }
    }
    
    return oldest;
  }

  /**
   * Get newest record timestamp
   */
  async getNewestRecord() {
    const logFiles = await this.getLogFiles();
    let newest = null;
    
    for (const logFile of logFiles) {
      const records = await this.readLogFile(logFile);
      if (records.length > 0) {
        const lastRecord = records[records.length - 1];
        if (!newest || new Date(lastRecord.timestamp) > new Date(newest)) {
          newest = lastRecord.timestamp;
        }
      }
    }
    
    return newest;
  }
}

module.exports = AuditLogger;