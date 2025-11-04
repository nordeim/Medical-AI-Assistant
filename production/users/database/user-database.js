// Production Healthcare User Database Operations
// Comprehensive database operations for user management system

const { Pool } = require('pg');
const crypto = require('crypto');
const config = require('../config/user-management-config');

class HealthcareUserDatabase {
  constructor() {
    this.pool = new Pool({
      host: process.env.DB_HOST || 'localhost',
      port: process.env.DB_PORT || 5432,
      database: process.env.DB_NAME || 'healthcare_users',
      user: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD,
      ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
      ...config.database.connectionPool
    });

    this.initPool();
  }

  async initPool() {
    try {
      // Test connection
      await this.pool.query('SELECT NOW()');
      console.log('Database pool initialized successfully');
    } catch (error) {
      console.error('Database pool initialization failed:', error);
      throw error;
    }
  }

  // User Management Operations
  async createUser(userData) {
    const client = await this.pool.connect();
    
    try {
      const { 
        userId, email, passwordHash, role, profile, verificationStatus = 'pending' 
      } = userData;

      await client.query('BEGIN');

      // Insert into auth.users (handled by Supabase, but keeping for reference)
      const authInsert = `
        INSERT INTO auth.users (
          instance_id, id, aud, role, email, encrypted_password, 
          email_confirmed_at, invited_at, confirmation_token, 
          confirmation_sent_at, recovery_token, recovery_sent_at, 
          email_change_token_new, email_change, email_change_sent_at, 
          last_sign_in_at, raw_app_meta_data, raw_user_meta_data, 
          is_super_admin, created_at, updated_at, phone, phone_confirmed_at, 
          phone_change, phone_change_token, phone_change_sent_at, 
          email_change_token_current, email_change_confirm_status, 
          banned_until, reauthentication_token, reauthentication_sent_at, 
          is_sso_user, deleted_at
        ) VALUES (
          '00000000-0000-0000-0000-000000000000', $1, 'authenticated', 'authenticated', 
          $2, $3, NOW(), NULL, '', NULL, '', NULL, '', '', NULL, NOW(), 
          '{}', '{}', false, NOW(), NOW(), NULL, NULL, '', '', NULL, '', 0, NULL, '', NULL, false, NULL
        )
      `;

      // Insert healthcare user profile
      const profileInsert = `
        INSERT INTO healthcare_users (
          user_id, email, first_name, last_name, role, medical_license,
          specialty, institution, phone_number, address, date_of_birth,
          emergency_contact, verification_status, account_status,
          registration_date, last_login, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW(), NULL, NOW(), NOW())
      `;

      await client.query(profileInsert, [
        userId, email, profile.firstName, profile.lastName, role,
        profile.medicalLicense, profile.specialty, profile.institution,
        profile.phoneNumber, JSON.stringify(profile.address),
        profile.dateOfBirth, JSON.stringify(profile.emergencyContact),
        verificationStatus, 'active'
      ]);

      // Insert user roles
      await this.insertUserRoles(client, userId, [role]);

      // Insert verification steps
      await this.insertVerificationSteps(client, userId);

      await client.query('COMMIT');

      return { success: true, userId };

    } catch (error) {
      await client.query('ROLLBACK');
      console.error('User creation error:', error);
      throw error;
    } finally {
      client.release();
    }
  }

  async getUserProfile(userId) {
    const query = `
      SELECT 
        user_id, email, first_name, last_name, role, medical_license,
        specialty, institution, phone_number, address, date_of_birth,
        emergency_contact, verification_status, account_status,
        registration_date, last_login, created_at, updated_at
      FROM healthcare_users 
      WHERE user_id = $1
    `;

    const result = await this.pool.query(query, [userId]);
    
    if (result.rows.length === 0) {
      throw new Error('User profile not found');
    }

    const user = result.rows[0];
    
    // Parse JSON fields
    user.address = JSON.parse(user.address || '{}');
    user.emergency_contact = JSON.parse(user.emergency_contact || '{}');

    return user;
  }

  async updateUserProfile(userId, updates) {
    const allowedFields = [
      'phone_number', 'address', 'emergency_contact', 'specialty', 'institution'
    ];

    const setClause = [];
    const values = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(updates)) {
      if (allowedFields.includes(key)) {
        const columnName = key.replace(/([A-Z])/g, '_$1').toLowerCase();
        setClause.push(`${columnName} = $${paramIndex}`);
        
        if (typeof value === 'object') {
          values.push(JSON.stringify(value));
        } else {
          values.push(value);
        }
        paramIndex++;
      }
    }

    if (setClause.length === 0) {
      throw new Error('No valid fields to update');
    }

    setClause.push(`updated_at = NOW()`);
    values.push(userId);

    const query = `
      UPDATE healthcare_users 
      SET ${setClause.join(', ')}
      WHERE user_id = $${paramIndex}
      RETURNING *
    `;

    const result = await this.pool.query(query, values);
    return result.rows[0];
  }

  async updateUserAccountStatus(userId, status, details = {}) {
    const query = `
      UPDATE healthcare_users 
      SET 
        account_status = $1,
        ${details.activatedAt ? 'activated_at = $2,' : ''}
        ${details.suspendedAt ? 'suspended_at = $3,' : ''}
        ${details.deactivatedAt ? 'deactivated_at = $4,' : ''}
        updated_at = NOW()
      WHERE user_id = $5
      RETURNING *
    `;

    const params = [status];
    if (details.activatedAt) params.push(details.activatedAt);
    if (details.suspendedAt) params.push(details.suspendedAt);
    if (details.deactivatedAt) params.push(details.deactivatedAt);
    params.push(userId);

    const result = await this.pool.query(query, params);
    return result.rows[0];
  }

  // Role Management Operations
  async getUserRoles(userId) {
    const query = `
      SELECT 
        ur.role_id, ur.role_name, ur.granted_at, ur.granted_by,
        ur.revoked_at, ur.revoked_by, ur.is_active,
        h_roles.role_level, h_roles.role_description
      FROM user_roles ur
      LEFT JOIN healthcare_roles h_roles ON ur.role_name = h_roles.role_name
      WHERE ur.user_id = $1 AND (ur.revoked_at IS NULL OR ur.is_active = true)
      ORDER BY h_roles.role_level DESC
    `;

    const result = await this.pool.query(query, [userId]);
    return result.rows;
  }

  async insertUserRoles(client, userId, roles) {
    for (const roleName of roles) {
      await client.query(`
        INSERT INTO user_roles (
          user_id, role_name, granted_at, granted_by, is_active
        ) VALUES ($1, $2, NOW(), $3, true)
      `, [userId, roleName, userId]); // granted_by would be system user in production
    }
  }

  async grantRole({ userId, roleName, grantedBy, reason, grantedAt }) {
    const client = await this.pool.connect();
    
    try {
      await client.query('BEGIN');

      // Insert new role
      const insertQuery = `
        INSERT INTO user_roles (
          user_id, role_name, granted_at, granted_by, reason, is_active
        ) VALUES ($1, $2, $3, $4, $5, true)
        RETURNING *
      `;

      const result = await client.query(insertQuery, [
        userId, roleName, grantedAt, grantedBy, reason
      ]);

      await client.query('COMMIT');
      return result.rows[0];

    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async revokeRole({ userId, roleName, revokedBy, reason, revokedAt }) {
    const query = `
      UPDATE user_roles 
      SET 
        is_active = false,
        revoked_at = $1,
        revoked_by = $2,
        reason = $3
      WHERE user_id = $4 AND role_name = $5 AND is_active = true
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      revokedAt, revokedBy, reason, userId, roleName
    ]);

    return result.rows[0];
  }

  // Onboarding Operations
  async createOnboardingRecord(onboardingData) {
    const query = `
      INSERT INTO user_onboarding (
        user_id, status, initiated_at, estimated_completion,
        current_step, onboarding_data, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      onboardingData.userId,
      onboardingData.status,
      onboardingData.initiatedAt,
      onboardingData.estimatedCompletion,
      onboardingData.currentStep,
      JSON.stringify(onboardingData.data)
    ]);

    return result.rows[0];
  }

  async getOnboardingRecord(userId) {
    const query = `
      SELECT * FROM user_onboarding 
      WHERE user_id = $1 
      ORDER BY created_at DESC 
      LIMIT 1
    `;

    const result = await this.pool.query(query, [userId]);
    
    if (result.rows.length > 0) {
      const record = result.rows[0];
      record.onboarding_data = JSON.parse(record.onboarding_data || '{}');
      return record;
    }

    return null;
  }

  async updateOnboardingStatus(userId, status, details = {}) {
    const query = `
      UPDATE user_onboarding 
      SET 
        status = $1,
        ${details.completedAt ? 'completed_at = $2,' : ''}
        ${details.failedStep ? 'failed_step = $3,' : ''}
        ${details.failedAt ? 'failed_at = $4,' : ''}
        ${details.failureReason ? 'failure_reason = $5,' : ''}
        updated_at = NOW()
      WHERE user_id = $6
      RETURNING *
    `;

    const params = [status];
    if (details.completedAt) params.push(details.completedAt);
    if (details.failedStep) params.push(details.failedStep);
    if (details.failedAt) params.push(details.failedAt);
    if (details.failureReason) params.push(details.failureReason);
    params.push(userId);

    const result = await this.pool.query(query, params);
    return result.rows[0];
  }

  async createVerificationStep(stepData) {
    const query = `
      INSERT INTO verification_steps (
        user_id, step_name, status, created_at
      ) VALUES ($1, $2, $3, $4)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      stepData.userId,
      stepData.stepName,
      stepData.status,
      stepData.createdAt
    ]);

    return result.rows[0];
  }

  async updateVerificationStep(userId, stepName, updates) {
    const setClause = [];
    const values = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(updates)) {
      const columnName = key.replace(/([A-Z])/g, '_$1').toLowerCase();
      setClause.push(`${columnName} = $${paramIndex}`);
      values.push(value);
      paramIndex++;
    }

    setClause.push(`updated_at = NOW()`);
    values.push(userId, stepName);

    const query = `
      UPDATE verification_steps 
      SET ${setClause.join(', ')}
      WHERE user_id = $${paramIndex} AND step_name = $${paramIndex + 1}
      RETURNING *
    `;

    const result = await this.pool.query(query, values);
    return result.rows[0];
  }

  async getVerificationSteps(userId) {
    const query = `
      SELECT * FROM verification_steps 
      WHERE user_id = $1 
      ORDER BY created_at
    `;

    const result = await this.pool.query(query, [userId]);
    return result.rows;
  }

  // Background Check Operations
  async createBackgroundCheckRequest(requestData) {
    const query = `
      INSERT INTO background_checks (
        user_id, request_id, request_type, priority, check_data,
        requested_at, estimated_completion, status, provider_id, reference_id
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      requestData.userId,
      requestData.requestId,
      requestData.requestType,
      requestData.priority,
      JSON.stringify(requestData.data),
      requestData.requestedAt,
      requestData.estimatedCompletion,
      requestData.status,
      requestData.providerId,
      requestData.referenceId
    ]);

    return result.rows[0];
  }

  // Reference Validation Operations
  async createReferenceRecords(references) {
    const client = await this.pool.connect();
    
    try {
      await client.query('BEGIN');

      const insertedRecords = [];
      
      for (const ref of references) {
        const query = `
          INSERT INTO professional_references (
            user_id, reference_id, name, email, phone, relationship,
            institution, years_known, status, requested_at, response_deadline
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
          RETURNING *
        `;

        const result = await client.query(query, [
          ref.userId, ref.referenceId, ref.name, ref.email, ref.phone,
          ref.relationship, ref.institution, ref.yearsKnown, ref.status,
          ref.requestedAt, ref.responseDeadline
        ]);

        insertedRecords.push(result.rows[0]);
      }

      await client.query('COMMIT');
      return insertedRecords;

    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  // Approval Process Operations
  async createApprovalRequest(approvalData) {
    const query = `
      INSERT INTO approval_requests (
        user_id, request_id, verification_summary, submitted_at,
        status, priority
      ) VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      approvalData.userId,
      approvalData.requestId,
      JSON.stringify(approvalData.verificationSummary),
      approvalData.submittedAt,
      approvalData.status,
      approvalData.priority
    ]);

    return result.rows[0];
  }

  async getApprovalRequest(requestId) {
    const query = `
      SELECT * FROM approval_requests 
      WHERE request_id = $1
    `;

    const result = await this.pool.query(query, [requestId]);
    
    if (result.rows.length > 0) {
      const request = result.rows[0];
      request.verification_summary = JSON.parse(request.verification_summary || '{}');
      return request;
    }

    return null;
  }

  async assignReviewer(userId, requestId) {
    const query = `
      INSERT INTO reviewer_assignments (
        reviewer_id, request_id, assigned_at, status
      ) VALUES ($1, $2, NOW(), 'assigned')
      RETURNING *
    `;

    const result = await this.pool.query(query, [userId, requestId]);
    return result.rows[0];
  }

  async getReviewerAssignment(reviewerId, requestId) {
    const query = `
      SELECT * FROM reviewer_assignments 
      WHERE reviewer_id = $1 AND request_id = $2 AND status = 'assigned'
    `;

    const result = await this.pool.query(query, [reviewerId, requestId]);
    return result.rows[0];
  }

  async recordReviewDecision(reviewData) {
    const query = `
      INSERT INTO review_decisions (
        request_id, reviewer_id, decision, comments, reviewed_at, status
      ) VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      reviewData.requestId,
      reviewData.reviewerId,
      reviewData.decision,
      reviewData.comments,
      reviewData.reviewedAt,
      reviewData.status
    ]);

    return result.rows[0];
  }

  async getAllReviews(requestId) {
    const query = `
      SELECT * FROM review_decisions 
      WHERE request_id = $1
    `;

    const result = await this.pool.query(query, [requestId]);
    return result.rows;
  }

  async updateApprovalRequestStatus(requestId, status, details = {}) {
    const query = `
      UPDATE approval_requests 
      SET 
        status = $1,
        ${details.completedAt ? 'completed_at = $2,' : ''}
        ${details.approvalCount ? 'approval_count = $3,' : ''}
        updated_at = NOW()
      WHERE request_id = $4
      RETURNING *
    `;

    const params = [status];
    if (details.completedAt) params.push(details.completedAt);
    if (details.approvalCount) params.push(details.approvalCount);
    params.push(requestId);

    const result = await this.pool.query(query, params);
    return result.rows[0];
  }

  // MFA Operations
  async createMFAChallenge(challengeData) {
    const query = `
      INSERT INTO mfa_challenges (
        challenge_id, user_id, type, expires_at, attempts, status, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      challengeData.challengeId,
      challengeData.userId,
      challengeData.type,
      challengeData.expiresAt,
      challengeData.attempts,
      challengeData.status
    ]);

    return result.rows[0];
  }

  // Session Management
  async createSession(sessionData) {
    const query = `
      INSERT INTO user_sessions (
        session_id, user_id, device_info, created_at, expires_at, status
      ) VALUES ($1, $2, $3, $4, $5, 'active')
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      sessionData.sessionId,
      sessionData.userId,
      JSON.stringify(sessionData.deviceInfo),
      sessionData.createdAt,
      sessionData.expiresAt
    ]);

    return result.rows[0];
  }

  // Audit and Compliance Operations
  async insertAuditBatch(events) {
    const client = await this.pool.connect();
    
    try {
      await client.query('BEGIN');

      for (const event of events) {
        await client.query(`
          INSERT INTO audit_events (
            event_id, user_id, event, category, source, timestamp,
            details, compliance_type, requires_review, retention_period
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        `, [
          event.eventId,
          event.userId || null,
          event.event,
          event.category,
          event.source,
          event.timestamp,
          JSON.stringify(event.details),
          event.complianceType || null,
          event.requiresReview || false,
          event.retentionPeriod || 2555
        ]);
      }

      await client.query('COMMIT');

    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async insertImmediateEvent(event) {
    const query = `
      INSERT INTO immediate_audit_events (
        event_id, user_id, event, category, timestamp, details, severity
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      ON CONFLICT (event_id) DO NOTHING
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      event.eventId,
      event.userId || null,
      event.event,
      event.category,
      event.timestamp,
      JSON.stringify(event.details),
      event.severity || 'medium'
    ]);

    return result.rows[0];
  }

  async queryAuditEvents(queryBuilder) {
    let query = `
      SELECT * FROM audit_events
    `;

    const conditions = [];
    const params = [];

    if (queryBuilder.conditions.length > 0) {
      conditions.push('WHERE ' + queryBuilder.conditions.join(' AND '));
      params.push(...queryBuilder.parameters);
    }

    conditions.push(`ORDER BY ${queryBuilder.orderBy}`);
    conditions.push(`LIMIT ${queryBuilder.limit}`);

    query += ' ' + conditions.join(' ');

    const result = await this.pool.query(query, params);
    
    return result.rows.map(row => {
      row.details = JSON.parse(row.details || '{}');
      return row;
    });
  }

  async getEvents(filters) {
    let query = `
      SELECT * FROM audit_events 
      WHERE 1=1
    `;

    const params = [];
    let paramIndex = 1;

    if (filters.userId) {
      query += ` AND user_id = $${paramIndex}`;
      params.push(filters.userId);
      paramIndex++;
    }

    if (filters.eventType) {
      query += ` AND event = $${paramIndex}`;
      params.push(filters.eventType);
      paramIndex++;
    }

    if (filters.startDate) {
      query += ` AND timestamp >= $${paramIndex}`;
      params.push(filters.startDate);
      paramIndex++;
    }

    if (filters.endDate) {
      query += ` AND timestamp <= $${paramIndex}`;
      params.push(filters.endDate);
      paramIndex++;
    }

    query += ` ORDER BY timestamp DESC LIMIT ${filters.limit || 100}`;

    const result = await this.pool.query(query, params);
    
    return result.rows.map(row => {
      row.details = JSON.parse(row.details || '{}');
      return row;
    });
  }

  // Privacy and GDPR Operations
  async recordGDPRConsent(consentRecord) {
    const query = `
      INSERT INTO gdpr_consents (
        consent_id, user_id, consent_type, granted, consent_details,
        ip_address_hash, user_agent, timestamp, version, expires_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      consentRecord.consentId,
      consentRecord.userId,
      consentRecord.consentType,
      consentRecord.granted,
      JSON.stringify(consentRecord.consentDetails),
      consentRecord.ipAddress,
      consentRecord.userAgent,
      consentRecord.timestamp,
      consentRecord.version,
      consentRecord.expiresAt
    ]);

    return result.rows[0];
  }

  async createDataAccessRequest(requestData) {
    const query = `
      INSERT INTO data_access_requests (
        request_id, user_id, request_type, status, created_at,
        estimated_completion, priority
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      requestData.requestId,
      requestData.userId,
      requestData.requestType,
      requestData.status,
      requestData.createdAt,
      requestData.estimatedCompletion,
      requestData.priority
    ]);

    return result.rows[0];
  }

  async updateDataAccessRequestStatus(requestId, status, details = {}) {
    const query = `
      UPDATE data_access_requests 
      SET 
        status = $1,
        ${details.completedAt ? 'completed_at = $2,' : ''}
        ${details.exportFile ? 'export_file = $3,' : ''}
        ${details.expiresAt ? 'expires_at = $4,' : ''}
        ${details.erasureResults ? 'erasure_results = $5,' : ''}
        updated_at = NOW()
      WHERE request_id = $6
      RETURNING *
    `;

    const params = [status];
    if (details.completedAt) params.push(details.completedAt);
    if (details.exportFile) params.push(details.exportFile);
    if (details.expiresAt) params.push(details.expiresAt);
    if (details.erasureResults) params.push(JSON.stringify(details.erasureResults));
    params.push(requestId);

    const result = await this.pool.query(query, params);
    return result.rows[0];
  }

  async storePrivacyImpactAssessment(assessment) {
    const query = `
      INSERT INTO privacy_impact_assessments (
        assessment_id, user_id, data_processing, assessment_date,
        risk_level, risk_factors, mitigation_measures, compliance_status
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      assessment.assessmentId,
      assessment.userId,
      JSON.stringify(assessment.dataProcessing),
      assessment.assessmentDate,
      assessment.riskLevel,
      JSON.stringify(assessment.riskFactors),
      JSON.stringify(assessment.mitigationMeasures),
      assessment.complianceStatus
    ]);

    return result.rows[0];
  }

  async storeComplianceReport(report) {
    const query = `
      INSERT INTO compliance_reports (
        report_id, compliance_type, period_start, period_end,
        generated_at, metrics, summary, violations, compliance_score,
        recommendations, generated_by
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      report.reportId,
      report.complianceType,
      report.period.start,
      report.period.end,
      report.generatedAt,
      JSON.stringify(report.metrics),
      JSON.stringify(report.summary),
      JSON.stringify(report.violations),
      report.complianceScore,
      JSON.stringify(report.recommendations),
      report.generatedBy
    ]);

    return result.rows[0];
  }

  // Security Operations
  async createAccountLock(lockData) {
    const query = `
      INSERT INTO account_locks (
        user_id, reason, lock_type, expires_at, locked_at
      ) VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      lockData.userId,
      lockData.reason,
      lockData.lockType,
      lockData.expiresAt,
      lockData.lockedAt
    ]);

    return result.rows[0];
  }

  // Emergency Access Operations
  async createEmergencyAccessRecord(accessData) {
    const query = `
      INSERT INTO emergency_access_records (
        user_id, permission, context, emergency_reason,
        urgency_level, created_at, status
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      accessData.userId,
      accessData.permission,
      JSON.stringify(accessData.context),
      accessData.emergencyReason,
      accessData.urgencyLevel,
      accessData.createdAt,
      accessData.status
    ]);

    return result.rows[0];
  }

  // Helper Operations
  async getUserProfileByEmail(email) {
    const query = `SELECT * FROM healthcare_users WHERE email = $1`;
    const result = await this.pool.query(query, [email]);
    return result.rows[0];
  }

  async updatePrivacyPreferences(userId, preferences) {
    const query = `
      UPDATE healthcare_users 
      SET privacy_preferences = $1, updated_at = NOW()
      WHERE user_id = $2
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      JSON.stringify(preferences), userId
    ]);

    return result.rows[0];
  }

  async removeUserField(userId, field) {
    const query = `
      UPDATE healthcare_users 
      SET ${field} = NULL, updated_at = NOW()
      WHERE user_id = $1
      RETURNING *
    `;

    const result = await this.pool.query(query, [userId]);
    return result.rows[0];
  }

  async updateUserField(userId, field, value) {
    const query = `
      UPDATE healthcare_users 
      SET ${field} = $1, updated_at = NOW()
      WHERE user_id = $2
      RETURNING *
    `;

    const result = await this.pool.query(query, [value, userId]);
    return result.rows[0];
  }

  async deleteUserDataSet(userId, datasetName) {
    // Implement dataset deletion based on dataset type
    const query = `
      DELETE FROM ${datasetName}_data 
      WHERE user_id = $1
    `;

    await this.pool.query(query, [userId]);
    return { success: true, datasetName };
  }

  async anonymizeUserDataSet(userId, datasetName) {
    const query = `
      UPDATE ${datasetName}_data 
      SET anonymized = true, anonymized_at = NOW()
      WHERE user_id = $1
    `;

    await this.pool.query(query, [userId]);
    return { success: true, datasetName, action: 'anonymized' };
  }

  async restrictUserDataSet(userId, datasetName) {
    const query = `
      UPDATE ${datasetName}_data 
      SET restricted = true, restricted_at = NOW()
      WHERE user_id = $1
    `;

    await this.pool.query(query, [userId]);
    return { success: true, datasetName, action: 'restricted' };
  }

  async getUserPreferences(userId) {
    const query = `SELECT privacy_preferences FROM healthcare_users WHERE user_id = $1`;
    const result = await this.pool.query(query, [userId]);
    return result.rows[0]?.privacy_preferences || {};
  }

  async getUserActivity(userId) {
    const query = `
      SELECT * FROM audit_events 
      WHERE user_id = $1 
      ORDER BY timestamp DESC 
      LIMIT 100
    `;

    const result = await this.pool.query(query, [userId]);
    return result.rows;
  }

  async getUserCompliance(userId) {
    const query = `
      SELECT * FROM compliance_events 
      WHERE user_id = $1 
      ORDER BY timestamp DESC 
      LIMIT 50
    `;

    const result = await this.pool.query(query, [userId]);
    return result.rows;
  }

  async createAccessRestriction(restrictionData) {
    const query = `
      INSERT INTO access_restrictions (
        user_id, patient_id, restriction_type, expires_at, created_at
      ) VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;

    const result = await this.pool.query(query, [
      restrictionData.userId,
      restrictionData.patientId,
      restrictionData.restrictionType,
      restrictionData.expiresAt,
      restrictionData.createdAt
    ]);

    return result.rows[0];
  }

  async getPatientCareTeam(patientId) {
    const query = `
      SELECT * FROM patient_care_team 
      WHERE patient_id = $1 AND status = 'active'
    `;

    const result = await this.pool.query(query, [patientId]);
    return result.rows;
  }

  // Utility Methods
  async close() {
    await this.pool.end();
  }

  async healthCheck() {
    try {
      const result = await this.pool.query('SELECT NOW()');
      return { 
        status: 'healthy', 
        timestamp: result.rows[0].now,
        connectionCount: this.pool.totalCount
      };
    } catch (error) {
      return { 
        status: 'unhealthy', 
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }
}

module.exports = new HealthcareUserDatabase();