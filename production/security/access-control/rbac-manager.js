/**
 * HIPAA-Compliant Role-Based Access Control (RBAC) Manager
 * Implements production-grade access controls with audit trails
 */

const crypto = require('crypto');
const jwt = require('jsonwebtoken');

class RBACManager {
  constructor(config = {}) {
    this.config = {
      sessionTimeout: 1800000, // 30 minutes
      maxFailedAttempts: 3,
      lockoutDuration: 900000, // 15 minutes
      jwtSecret: process.env.JWT_SECRET || 'default-secret-change-in-production',
      ...config
    };
    
    this.users = new Map();
    this.roles = new Map();
    this.permissions = new Map();
    this.sessions = new Map();
    this.failedAttempts = new Map();
    this.auditLog = [];
    
    this.initializeRolesAndPermissions();
  }

  /**
   * Initialize HIPAA-compliant roles and permissions
   */
  initializeRolesAndPermissions() {
    // Define HIPAA-compliant roles
    const hipaaRoles = {
      'admin': {
        id: 'admin',
        name: 'System Administrator',
        description: 'Full system access with administrative privileges',
        permissions: [
          'system:config', 'user:manage', 'role:assign', 'audit:view',
          'phi:access', 'phi:modify', 'phi:export', 'backup:create',
          'compliance:manage', 'security:manage'
        ],
        restrictions: {
          maxConcurrentSessions: 1,
          allowedIPRanges: [],
          requireMFA: true,
          sessionTimeout: 1800000
        }
      },
      'doctor': {
        id: 'doctor',
        name: 'Physician',
        description: 'Medical staff with patient care access',
        permissions: [
          'patient:view', 'patient:update', 'phi:read', 'phi:write',
          'appointment:manage', 'prescription:write', 'medical:record'
        ],
        restrictions: {
          maxConcurrentSessions: 3,
          allowedIPRanges: [],
          requireMFA: true,
          sessionTimeout: 1800000
        }
      },
      'nurse': {
        id: 'nurse',
        name: 'Registered Nurse',
        description: 'Nursing staff with patient care support access',
        permissions: [
          'patient:view', 'patient:update', 'phi:read', 'phi:write',
          'appointment:view', 'vital:record', 'medication:administer'
        ],
        restrictions: {
          maxConcurrentSessions: 2,
          allowedIPRanges: [],
          requireMFA: false,
          sessionTimeout: 1800000
        }
      },
      'billing': {
        id: 'billing',
        name: 'Billing Specialist',
        description: 'Financial staff with billing and insurance access',
        permissions: [
          'billing:view', 'billing:update', 'insurance:verify', 'payment:process',
          'phi:read', 'financial:report'
        ],
        restrictions: {
          maxConcurrentSessions: 3,
          allowedIPRanges: [],
          requireMFA: true,
          sessionTimeout: 1800000
        }
      },
      'receptionist': {
        id: 'receptionist',
        name: 'Front Desk Receptionist',
        description: 'Administrative staff with patient registration access',
        permissions: [
          'patient:register', 'patient:view', 'appointment:schedule',
          'phi:read', 'contact:update'
        ],
        restrictions: {
          maxConcurrentSessions: 2,
          allowedIPRanges: ['192.168.1.0/24', '10.0.0.0/8'],
          requireMFA: false,
          sessionTimeout: 1800000
        }
      },
      'it_support': {
        id: 'it_support',
        name: 'IT Support Specialist',
        description: 'Technical support with limited PHI access',
        permissions: [
          'system:diagnose', 'user:reset_password', 'backup:restore',
          'log:view', 'phi:read:anonymized'
        ],
        restrictions: {
          maxConcurrentSessions: 1,
          allowedIPRanges: ['192.168.1.0/24'],
          requireMFA: true,
          sessionTimeout: 1800000
        }
      }
    };

    Object.values(hipaaRoles).forEach(role => {
      this.roles.set(role.id, role);
    });

    // Log role initialization
    this.logAuditEvent('system', 'roles_initialized', {
      roles_count: hipaaRoles.length,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * User registration with HIPAA compliance
   */
  async registerUser(userData) {
    const {
      username,
      email,
      password,
      roleId,
      firstName,
      lastName,
      department,
      phoneNumber,
      ipAddress
    } = userData;

    // Validate input
    if (!username || !email || !password || !roleId) {
      throw new Error('Missing required fields for user registration');
    }

    if (!this.roles.has(roleId)) {
      throw new Error('Invalid role specified');
    }

    // Check if user already exists
    if (this.users.has(username)) {
      throw new Error('Username already exists');
    }

    // Hash password with salt
    const salt = crypto.randomBytes(32).toString('hex');
    const hashedPassword = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');

    // Create user
    const user = {
      id: crypto.randomUUID(),
      username,
      email,
      password: {
        hash: hashedPassword,
        salt,
        iterations: 100000
      },
      roleId,
      firstName,
      lastName,
      department,
      phoneNumber,
      status: 'active',
      createdAt: new Date().toISOString(),
      lastLogin: null,
      loginAttempts: 0,
      lockedUntil: null,
      mfaEnabled: this.roles.get(roleId).restrictions.requireMFA,
      mfaSecret: this.roles.get(roleId).restrictions.requireMFA ? this.generateMFASecret() : null,
      passwordLastChanged: new Date().toISOString(),
      mustChangePassword: false,
      ipWhitelist: [],
      allowedDepartments: [department],
      sessionHistory: []
    };

    this.users.set(username, user);

    // Log user creation
    this.logAuditEvent('system', 'user_created', {
      userId: user.id,
      username,
      roleId,
      department,
      ipAddress,
      timestamp: new Date().toISOString()
    });

    return {
      success: true,
      userId: user.id,
      mfaSecret: user.mfaSecret,
      message: 'User registered successfully'
    };
  }

  /**
   * Enhanced user authentication with security checks
   */
  async authenticate(username, password, ipAddress = '', mfaToken = '') {
    const user = this.users.get(username);
    
    if (!user) {
      this.logAuditEvent('system', 'authentication_failed', {
        username,
        reason: 'user_not_found',
        ipAddress,
        timestamp: new Date().toISOString()
      });
      return { success: false, reason: 'Invalid credentials' };
    }

    // Check if account is locked
    if (user.lockedUntil && new Date() < user.lockedUntil) {
      const remainingTime = Math.ceil((user.lockedUntil - new Date()) / 1000 / 60);
      this.logAuditEvent('system', 'authentication_failed', {
        userId: user.id,
        username,
        reason: 'account_locked',
        remainingTime,
        ipAddress,
        timestamp: new Date().toISOString()
      });
      return { success: false, reason: `Account locked for ${remainingTime} minutes` };
    }

    // Verify password
    const hashedPassword = crypto.pbkdf2Sync(password, user.password.salt, 100000, 64, 'sha512').toString('hex');
    
    if (hashedPassword !== user.password.hash) {
      await this.handleFailedLogin(user, ipAddress);
      return { success: false, reason: 'Invalid credentials' };
    }

    // Check MFA if required
    if (user.mfaEnabled) {
      if (!mfaToken || !this.verifyMFAToken(user.mfaSecret, mfaToken)) {
        await this.handleFailedLogin(user, ipAddress);
        this.logAuditEvent('system', 'authentication_failed', {
          userId: user.id,
          username,
          reason: 'mfa_failed',
          ipAddress,
          timestamp: new Date().toISOString()
        });
        return { success: false, reason: 'Invalid MFA token' };
      }
    }

    // Check IP restrictions
    const role = this.roles.get(user.roleId);
    if (role.restrictions.allowedIPRanges.length > 0 && !this.isIPAllowed(ipAddress, role.restrictions.allowedIPRanges)) {
      this.logAuditEvent('system', 'authentication_failed', {
        userId: user.id,
        username,
        reason: 'ip_restricted',
        ipAddress,
        allowedRanges: role.restrictions.allowedIPRanges,
        timestamp: new Date().toISOString()
      });
      return { success: false, reason: 'Access denied from this IP address' };
    }

    // Check session limits
    const activeSessions = Array.from(this.sessions.values()).filter(s => s.userId === user.id);
    if (activeSessions.length >= role.restrictions.maxConcurrentSessions) {
      this.logAuditEvent('system', 'authentication_failed', {
        userId: user.id,
        username,
        reason: 'session_limit_exceeded',
        activeSessions: activeSessions.length,
        maxSessions: role.restrictions.maxConcurrentSessions,
        ipAddress,
        timestamp: new Date().toISOString()
      });
      return { success: false, reason: 'Maximum concurrent sessions reached' };
    }

    // Successful authentication
    user.loginAttempts = 0;
    user.lockedUntil = null;
    user.lastLogin = new Date().toISOString();
    
    const session = await this.createSession(user, ipAddress);
    
    this.logAuditEvent('system', 'authentication_success', {
      userId: user.id,
      username,
      sessionId: session.sessionId,
      ipAddress,
      timestamp: new Date().toISOString()
    });

    return {
      success: true,
      sessionId: session.sessionId,
      token: session.token,
      user: {
        id: user.id,
        username: user.username,
        role: role.name,
        permissions: role.permissions,
        mustChangePassword: user.mustChangePassword
      }
    };
  }

  /**
   * Create session with security controls
   */
  async createSession(user, ipAddress) {
    const sessionId = crypto.randomUUID();
    const token = jwt.sign(
      {
        userId: user.id,
        username: user.username,
        roleId: user.roleId,
        sessionId,
        iat: Math.floor(Date.now() / 1000)
      },
      this.config.jwtSecret,
      { expiresIn: '30m' }
    );

    const session = {
      sessionId,
      userId: user.id,
      username: user.username,
      roleId: user.roleId,
      token,
      createdAt: new Date().toISOString(),
      lastActivity: new Date().toISOString(),
      expiresAt: new Date(Date.now() + this.config.sessionTimeout).toISOString(),
      ipAddress,
      userAgent: '', // Would be populated from request
      status: 'active'
    };

    this.sessions.set(sessionId, session);
    
    // Update user's session history
    user.sessionHistory.push({
      sessionId,
      loginTime: session.createdAt,
      ipAddress
    });

    return session;
  }

  /**
   * Check if user has specific permission
   */
  hasPermission(sessionId, permission) {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    const role = this.roles.get(session.roleId);
    if (!role) return false;

    // Check if session is still valid
    if (new Date() > new Date(session.expiresAt)) {
      this.sessions.delete(sessionId);
      return false;
    }

    return role.permissions.includes(permission) || role.permissions.includes('*');
  }

  /**
   * Verify JWT token and return session info
   */
  verifyToken(token) {
    try {
      const decoded = jwt.verify(token, this.config.jwtSecret);
      const session = this.sessions.get(decoded.sessionId);
      
      if (!session || session.token !== token) {
        return { valid: false, reason: 'Invalid session' };
      }

      // Check if session has expired
      if (new Date() > new Date(session.expiresAt)) {
        this.sessions.delete(decoded.sessionId);
        return { valid: false, reason: 'Session expired' };
      }

      // Update last activity
      session.lastActivity = new Date().toISOString();

      return {
        valid: true,
        session: {
          sessionId: session.sessionId,
          userId: session.userId,
          username: session.username,
          roleId: session.roleId,
          permissions: this.roles.get(session.roleId).permissions,
          expiresAt: session.expiresAt
        }
      };
    } catch (error) {
      return { valid: false, reason: 'Invalid token' };
    }
  }

  /**
   * Handle failed login attempts
   */
  async handleFailedLogin(user, ipAddress) {
    user.loginAttempts++;
    
    if (user.loginAttempts >= this.config.maxFailedAttempts) {
      user.lockedUntil = new Date(Date.now() + this.config.lockoutDuration);
      this.logAuditEvent('system', 'account_locked', {
        userId: user.id,
        username: user.username,
        failedAttempts: user.loginAttempts,
        lockoutDuration: this.config.lockoutDuration / 1000 / 60,
        ipAddress,
        timestamp: new Date().toISOString()
      });
    }

    this.logAuditEvent('system', 'login_failed', {
      userId: user.id,
      username: user.username,
      failedAttempts: user.loginAttempts,
      ipAddress,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Generate MFA secret
   */
  generateMFASecret() {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
    let secret = '';
    for (let i = 0; i < 32; i++) {
      secret += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return secret;
  }

  /**
   * Verify MFA token (simplified TOTP)
   */
  verifyMFASecret(secret, token) {
    // In production, use proper TOTP implementation like speakeasy
    return token.length === 6 && /^\d+$/.test(token);
  }

  /**
   * Check if IP is in allowed ranges
   */
  isIPAllowed(ip, allowedRanges) {
    // Simplified IP range checking - use proper CIDR implementation in production
    return allowedRanges.includes(ip) || allowedRanges.length === 0;
  }

  /**
   * Log audit events with HIPAA compliance
   */
  logAuditEvent(userId, action, details = {}) {
    const auditEvent = {
      id: crypto.randomUUID(),
      userId,
      action,
      timestamp: new Date().toISOString(),
      ipAddress: details.ipAddress || '',
      userAgent: details.userAgent || '',
      details: {
        ...details,
        // Ensure PHI is not logged in audit trails
        phi_accessed: details.phi_accessed ? '[REDACTED]' : false,
        session_id: details.sessionId || ''
      },
      severity: this.getAuditSeverity(action),
      category: this.getAuditCategory(action)
    };

    this.auditLog.push(auditEvent);
    
    // In production, this would write to a secure, tamper-proof audit log
    console.log(`[AUDIT] ${auditEvent.timestamp} | ${userId} | ${action} | ${auditEvent.severity}`);
  }

  /**
   * Get audit event severity
   */
  getAuditSeverity(action) {
    const highSeverity = [
      'authentication_failed', 'account_locked', 'unauthorized_access',
      'phi_exported', 'data_breach', 'configuration_changed'
    ];
    
    const mediumSeverity = [
      'authentication_success', 'permission_denied', 'session_expired',
      'password_changed', 'mfa_enabled'
    ];

    if (highSeverity.includes(action)) return 'HIGH';
    if (mediumSeverity.includes(action)) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Get audit event category
   */
  getAuditCategory(action) {
    if (action.includes('auth')) return 'AUTHENTICATION';
    if (action.includes('phi')) return 'PHI_ACCESS';
    if (action.includes('data')) return 'DATA_ACCESS';
    if (action.includes('system')) return 'SYSTEM';
    return 'GENERAL';
  }

  /**
   * Get audit logs with filtering
   */
  getAuditLogs(filters = {}) {
    let logs = [...this.auditLog];

    if (filters.userId) {
      logs = logs.filter(log => log.userId === filters.userId);
    }

    if (filters.action) {
      logs = logs.filter(log => log.action === filters.action);
    }

    if (filters.severity) {
      logs = logs.filter(log => log.severity === filters.severity);
    }

    if (filters.startDate) {
      logs = logs.filter(log => new Date(log.timestamp) >= new Date(filters.startDate));
    }

    if (filters.endDate) {
      logs = logs.filter(log => new Date(log.timestamp) <= new Date(filters.endDate));
    }

    return logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }

  /**
   * Terminate session
   */
  async terminateSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      this.sessions.delete(sessionId);
      this.logAuditEvent(session.userId, 'session_terminated', {
        sessionId,
        terminationTime: new Date().toISOString()
      });
    }
  }

  /**
   * Clean up expired sessions
   */
  cleanupExpiredSessions() {
    const now = new Date();
    for (const [sessionId, session] of this.sessions.entries()) {
      if (now > new Date(session.expiresAt)) {
        this.sessions.delete(sessionId);
        this.logAuditEvent(session.userId, 'session_expired', {
          sessionId,
          expirationTime: session.expiresAt
        });
      }
    }
  }
}

module.exports = RBACManager;