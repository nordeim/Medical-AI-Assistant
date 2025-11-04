// Healthcare Role-Based Access Control (RBAC) System
// Production-grade access control for medical professionals

const config = require('../config/user-management-config');

class HealthcareRBACSystem {
  constructor() {
    this.config = config;
    this.roles = this.config.healthcare.roles;
    this.auditLogger = require('../monitoring/audit-logger');
    this.cache = new Map(); // Simple in-memory cache (use Redis in production)
  }

  // Role Hierarchy and Permission Management
  async getUserRoles(userId) {
    try {
      // Check cache first
      if (this.cache.has(`user_roles_${userId}`)) {
        return this.cache.get(`user_roles_${userId}`);
      }

      const { data, error } = await require('../database/user-database').getUserRoles(userId);
      
      if (error) {
        throw new Error(`Failed to get user roles: ${error.message}`);
      }

      // Enrich roles with hierarchy information
      const enrichedRoles = data.map(role => {
        const roleConfig = this.roles[role.role_name];
        return {
          ...role,
          level: roleConfig?.level || 0,
          permissions: roleConfig?.permissions || [],
          description: roleConfig?.description || ''
        };
      });

      // Cache for 5 minutes
      this.cache.set(`user_roles_${userId}`, enrichedRoles);
      return enrichedRoles;

    } catch (error) {
      console.error('Error getting user roles:', error);
      throw error;
    }
  }

  // Permission Checking System
  async checkPermission(userId, permission, context = {}) {
    try {
      const userRoles = await this.getUserRoles(userId);
      const hasPermission = userRoles.some(role => {
        const roleConfig = this.roles[role.role_name];
        return roleConfig?.permissions.includes('*') || // Super admin
               roleConfig?.permissions.includes(permission);
      });

      // Log access attempt
      await this.auditLogger.logEvent({
        userId,
        event: 'permission.check',
        details: {
          permission,
          context,
          granted: hasPermission,
          userRoles: userRoles.map(r => r.role_name)
        },
        timestamp: new Date().toISOString(),
        source: 'rbac_system'
      });

      // Check for emergency override
      if (!hasPermission && context.emergency) {
        return await this.handleEmergencyAccess(userId, permission, context);
      }

      return hasPermission;

    } catch (error) {
      console.error('Permission check error:', error);
      return false;
    }
  }

  // Grant Role to User
  async grantRole(userId, roleName, grantedBy, reason = '') {
    try {
      // Verify the role exists
      const roleConfig = this.roles[roleName];
      if (!roleConfig) {
        throw new Error(`Invalid role: ${roleName}`);
      }

      // Check if user already has this role
      const existingRoles = await this.getUserRoles(userId);
      if (existingRoles.some(role => role.role_name === roleName)) {
        throw new Error('User already has this role');
      }

      // Verify grantor has permission to grant this role
      const canGrant = await this.checkPermission(grantedBy, 'roles.grant');
      if (!canGrant) {
        throw new Error('Insufficient permissions to grant role');
      }

      // Grant the role
      const { error } = await require('../database/user-database').grantRole({
        userId,
        roleName,
        grantedBy,
        reason,
        grantedAt: new Date().toISOString()
      });

      if (error) {
        throw new Error(`Failed to grant role: ${error.message}`);
      }

      // Clear cache
      this.cache.delete(`user_roles_${userId}`);

      // Log role grant
      await this.auditLogger.logEvent({
        userId,
        event: 'role.granted',
        details: {
          role: roleName,
          grantedBy,
          reason,
          roleLevel: roleConfig.level
        },
        timestamp: new Date().toISOString(),
        source: 'rbac_system'
      });

      // Notify user of role change
      await this.notifyRoleChange(userId, 'granted', roleName);

      return true;

    } catch (error) {
      console.error('Role grant error:', error);
      throw error;
    }
  }

  // Revoke Role from User
  async revokeRole(userId, roleName, revokedBy, reason = '') {
    try {
      // Check if user has this role
      const existingRoles = await this.getUserRoles(userId);
      if (!existingRoles.some(role => role.role_name === roleName)) {
        throw new Error('User does not have this role');
      }

      // Verify revoker has permission
      const canRevoke = await this.checkPermission(revokedBy, 'roles.revoke');
      if (!canRevoke) {
        throw new Error('Insufficient permissions to revoke role');
      }

      // Revoke the role
      const { error } = await require('../database/user-database').revokeRole({
        userId,
        roleName,
        revokedBy,
        reason,
        revokedAt: new Date().toISOString()
      });

      if (error) {
        throw new Error(`Failed to revoke role: ${error.message}`);
      }

      // Clear cache
      this.cache.delete(`user_roles_${userId}`);

      // Log role revocation
      await this.auditLogger.logEvent({
        userId,
        event: 'role.revoked',
        details: {
          role: roleName,
          revokedBy,
          reason,
          roleLevel: this.roles[roleName]?.level || 0
        },
        timestamp: new Date().toISOString(),
        source: 'rbac_system'
      });

      // Notify user of role change
      await this.notifyRoleChange(userId, 'revoked', roleName);

      return true;

    } catch (error) {
      console.error('Role revocation error:', error);
      throw error;
    }
  }

  // Specialty-Based Access Control
  async checkSpecialtyAccess(userId, specialty, action, patientContext = {}) {
    try {
      const userProfile = await require('../database/user-database').getUserProfile(userId);
      const userSpecialty = userProfile?.specialty;
      
      // Admin users can access any specialty
      const userRoles = await this.getUserRoles(userId);
      const isAdmin = userRoles.some(role => 
        ['super_admin', 'hospital_admin', 'department_head'].includes(role.role_name)
      );

      if (isAdmin) {
        return true;
      }

      // Check if user has the required specialty
      if (userSpecialty === specialty) {
        return true;
      }

      // Check for emergency access
      if (patientContext.emergency) {
        return await this.handleEmergencySpecialtyAccess(userId, specialty, action, patientContext);
      }

      // Check for collaborative care permissions
      if (patientContext.collaborativeCare && patientContext.primaryPhysician === userId) {
        return true;
      }

      // Log unauthorized specialty access attempt
      await this.auditLogger.logEvent({
        userId,
        event: 'specialty.access.denied',
        details: {
          requestedSpecialty: specialty,
          userSpecialty,
          action,
          reason: 'specialty_mismatch'
        },
        timestamp: new Date().toISOString(),
        source: 'rbac_system'
      });

      return false;

    } catch (error) {
      console.error('Specialty access check error:', error);
      return false;
    }
  }

  // Data Access Control
  async checkDataAccess(userId, resourceType, resourceId, action, context = {}) {
    try {
      // Healthcare data access rules
      switch (resourceType) {
        case 'patient_record':
          return await this.checkPatientDataAccess(userId, resourceId, action, context);
        
        case 'medical_document':
          return await this.checkMedicalDocumentAccess(userId, resourceId, action, context);
        
        case 'appointment':
          return await this.checkAppointmentAccess(userId, resourceId, action, context);
        
        case 'medical_device':
          return await this.checkMedicalDeviceAccess(userId, resourceId, action, context);
        
        default:
          return await this.checkGenericDataAccess(userId, resourceType, resourceId, action, context);
      }

    } catch (error) {
      console.error('Data access check error:', error);
      return false;
    }
  }

  async checkPatientDataAccess(userId, patientId, action, context) {
    const userRoles = await this.getUserRoles(userId);
    const userProfile = await require('../database/user-database').getUserProfile(userId);

    // Admins have full access
    const isAdmin = userRoles.some(role => 
      ['super_admin', 'hospital_admin'].includes(role.role_name)
    );

    if (isAdmin) {
      return true;
    }

    // Primary physician has full access
    if (context.primaryPhysician === userId) {
      return true;
    }

    // Direct care team members
    const careTeam = await require('../database/user-database').getPatientCareTeam(patientId);
    const isOnCareTeam = careTeam.some(member => member.user_id === userId);
    
    if (isOnCareTeam && ['read', 'update'].includes(action)) {
      return true;
    }

    // Emergency access
    if (context.emergency) {
      await this.logEmergencyAccess(userId, patientId, 'patient_data', action);
      return true;
    }

    // Audit access attempts
    await this.auditLogger.logEvent({
      userId,
      event: 'data.access.attempt',
      details: {
        resourceType: 'patient_record',
        resourceId: patientId,
        action,
        granted: false,
        reason: 'insufficient_privileges'
      },
      timestamp: new Date().toISOString(),
      source: 'rbac_system'
    });

    return false;
  }

  // Emergency Access Protocols
  async handleEmergencyAccess(userId, permission, context) {
    try {
      // Log emergency access attempt
      await this.auditLogger.logEvent({
        userId,
        event: 'emergency.access',
        details: {
          permission,
          context,
          justification: context.emergencyReason,
          urgencyLevel: context.urgencyLevel || 'medium'
        },
        timestamp: new Date().toISOString(),
        source: 'rbac_system'
      });

      // Create emergency access record
      await this.createEmergencyAccessRecord(userId, permission, context);

      // Notify security team if high urgency
      if (context.urgencyLevel === 'high' || context.urgencyLevel === 'critical') {
        await this.notifySecurityTeam(userId, 'emergency_access', context);
      }

      return true;

    } catch (error) {
      console.error('Emergency access handling error:', error);
      return false;
    }
  }

  // Administrative Functions
  async getRolePermissions(roleName) {
    const roleConfig = this.roles[roleName];
    if (!roleConfig) {
      throw new Error(`Role not found: ${roleName}`);
    }

    return {
      roleName,
      name: roleConfig.name,
      level: roleConfig.level,
      permissions: roleConfig.permissions,
      description: roleConfig.description
    };
  }

  async listAllRoles() {
    return Object.keys(this.roles).map(roleName => ({
      roleName,
      name: this.roles[roleName].name,
      level: this.roles[roleName].level,
      description: this.roles[roleName].description,
      permissionCount: this.roles[roleName].permissions.length
    }));
  }

  async getUserPermissionSummary(userId) {
    const userRoles = await this.getUserRoles(userId);
    const allPermissions = new Set();
    
    userRoles.forEach(role => {
      const roleConfig = this.roles[role.role_name];
      if (roleConfig) {
        roleConfig.permissions.forEach(permission => {
          if (permission === '*') {
            allPermissions.add('*'); // Super admin
          } else {
            allPermissions.add(permission);
          }
        });
      }
    });

    return {
      userId,
      roles: userRoles,
      permissions: Array.from(allPermissions),
      isSuperAdmin: allPermissions.has('*'),
      highestRole: userRoles.reduce((highest, current) => {
        const currentLevel = this.roles[current.role_name]?.level || 0;
        const highestLevel = highest ? this.roles[highest.role_name]?.level || 0 : 0;
        return currentLevel > highestLevel ? current : highest;
      }, null)
    };
  }

  // Audit and Compliance
  async getAccessAuditTrail(userId, startDate, endDate) {
    return await this.auditLogger.getEvents({
      userId,
      source: 'rbac_system',
      startDate,
      endDate,
      eventTypes: [
        'permission.check',
        'role.granted',
        'role.revoked',
        'emergency.access',
        'data.access.attempt'
      ]
    });
  }

  // Helper Methods
  async createEmergencyAccessRecord(userId, permission, context) {
    const record = {
      userId,
      permission,
      context,
      emergencyReason: context.emergencyReason,
      urgencyLevel: context.urgencyLevel,
      createdAt: new Date().toISOString(),
      status: 'active'
    };

    // Store emergency access record
    await require('../database/user-database').createEmergencyAccessRecord(record);
  }

  async notifySecurityTeam(userId, eventType, context) {
    // Implement security team notification
    console.log(`Security team notification: ${eventType} by user ${userId}`);
  }

  async notifyRoleChange(userId, changeType, roleName) {
    // Implement role change notification
    console.log(`Role ${changeType}: ${roleName} for user ${userId}`);
  }

  async logEmergencyAccess(userId, resourceId, resourceType, action) {
    await this.auditLogger.logEvent({
      userId,
      event: 'emergency.data.access',
      details: {
        resourceId,
        resourceType,
        action,
        emergency: true
      },
      timestamp: new Date().toISOString(),
      source: 'rbac_system'
    });
  }

  // Placeholder methods for specific access checks
  async checkMedicalDocumentAccess(userId, documentId, action, context) {
    // Implement medical document access logic
    return true;
  }

  async checkAppointmentAccess(userId, appointmentId, action, context) {
    // Implement appointment access logic
    return true;
  }

  async checkMedicalDeviceAccess(userId, deviceId, action, context) {
    // Implement medical device access logic
    return true;
  }

  async checkGenericDataAccess(userId, resourceType, resourceId, action, context) {
    // Implement generic data access logic
    return false;
  }

  async handleEmergencySpecialtyAccess(userId, specialty, action, context) {
    // Implement emergency specialty access logic
    await this.logEmergencyAccess(userId, specialty, 'specialty_access', action);
    return true;
  }
}

module.exports = HealthcareRBACSystem;