// Production User Management Configuration
// Healthcare-specific user management and access control system

module.exports = {
  // Healthcare Role Definitions
  healthcare: {
    roles: {
      SUPER_ADMIN: {
        id: 'super_admin',
        name: 'Super Administrator',
        permissions: ['*'],
        level: 10,
        description: 'Full system access'
      },
      COMPLIANCE_OFFICER: {
        id: 'compliance_officer',
        name: 'Compliance Officer',
        permissions: [
          'audit.view',
          'compliance.manage',
          'users.review',
          'reports.generate'
        ],
        level: 9,
        description: 'Compliance and audit management'
      },
      SECURITY_ADMIN: {
        id: 'security_admin',
        name: 'Security Administrator',
        permissions: [
          'security.manage',
          'access.control',
          'users.suspend',
          'audit.configure'
        ],
        level: 8,
        description: 'System security management'
      },
      HOSPITAL_ADMIN: {
        id: 'hospital_admin',
        name: 'Hospital Administrator',
        permissions: [
          'users.manage',
          'facility.configure',
          'staff.supervise',
          'resources.allocate'
        ],
        level: 7,
        description: 'Hospital-level administration'
      },
      DEPARTMENT_HEAD: {
        id: 'department_head',
        name: 'Department Head',
        permissions: [
          'users.review',
          'department.manage',
          'staff.supervise',
          'reports.department'
        ],
        level: 6,
        description: 'Medical department leadership'
      },
      SENIOR_DOCTOR: {
        id: 'senior_doctor',
        name: 'Senior Doctor',
        permissions: [
          'patients.manage',
          'medical.manage',
          'staff.supervise',
          'reports.medical'
        ],
        level: 5,
        description: 'Senior medical practitioner'
      },
      DOCTOR: {
        id: 'doctor',
        name: 'Doctor',
        permissions: [
          'patients.manage',
          'medical.read',
          'medical.create',
          'appointments.manage'
        ],
        level: 4,
        description: 'Licensed medical doctor'
      },
      SENIOR_NURSE: {
        id: 'senior_nurse',
        name: 'Senior Nurse',
        permissions: [
          'patients.read',
          'patients.update',
          'nursing.supervise',
          'care.plans'
        ],
        level: 3,
        description: 'Senior nursing staff'
      },
      NURSE: {
        id: 'nurse',
        name: 'Nurse',
        permissions: [
          'patients.read',
          'patients.update',
          'care.execute',
          'medication.administer'
        ],
        level: 2,
        description: 'Licensed nurse'
      },
      MEDICAL_ASSISTANT: {
        id: 'medical_assistant',
        name: 'Medical Assistant',
        permissions: [
          'patients.read',
          'appointments.schedule',
          'basic.care',
          'documentation'
        ],
        level: 1,
        description: 'Medical support staff'
      }
    },

    // Specialty-specific permissions
    specialties: {
      CARDIOLOGY: { baseRole: 'doctor', additionalPermissions: ['cardiology.special'] },
      ONCOLOGY: { baseRole: 'doctor', additionalPermissions: ['oncology.special'] },
      PEDIATRICS: { baseRole: 'doctor', additionalPermissions: ['pediatrics.special'] },
      EMERGENCY: { baseRole: 'doctor', additionalPermissions: ['emergency.special'] },
      SURGERY: { baseRole: 'doctor', additionalPermissions: ['surgery.special'] },
      RADIOLOGY: { baseRole: 'doctor', additionalPermissions: ['radiology.special'] }
    }
  },

  // Authentication Configuration
  authentication: {
    sessionTimeout: 3600, // 1 hour for healthcare
    maxLoginAttempts: 3,
    lockoutDuration: 1800, // 30 minutes
    passwordPolicy: {
      minLength: 12,
      requireUppercase: true,
      requireLowercase: true,
      requireNumbers: true,
      requireSpecialChars: true,
      maxAge: 2592000, // 30 days
      historyCount: 12 // Prevent password reuse
    },
    mfa: {
      enabled: true,
      methods: ['totp', 'sms', 'email'],
      requiredRoles: ['doctor', 'senior_doctor', 'hospital_admin']
    },
    supabase: {
      authProviders: ['email', 'sso'],
      redirectUrls: [
        'https://healthcare.app/login',
        'https://healthcare.app/admin',
        'https://healthcare.app/doctor'
      ]
    }
  },

  // User Registration Configuration
  registration: {
    requireMedicalLicense: true,
    requireBackgroundCheck: true,
    requireProfessionalReferences: 2,
    verificationSteps: [
      'basic_information',
      'medical_credentials',
      'background_check',
      'reference_validation',
      'institutional_affiliation',
      'final_approval'
    ],
    approvalWorkflow: {
      autoApproval: false,
      requireHumanReview: true,
      reviewerRoles: ['hospital_admin', 'compliance_officer']
    }
  },

  // Privacy and Compliance Configuration
  privacy: {
    gdpr: {
      requireConsent: true,
      dataMinimization: true,
      rightToErasure: true,
      dataPortability: true,
      retentionPeriod: 2555 // 7 years for medical records
    },
    hipaa: {
      minimumNecessary: true,
      auditRequired: true,
      encryptionRequired: true,
      accessControls: true
    },
    dataRetention: {
      patientData: 2555, // 7 years
      auditLogs: 2555, // 7 years
      userAccounts: null, // Until termination
      sessionData: 86400 // 24 hours
    }
  },

  // Activity Monitoring Configuration
  monitoring: {
    auditTrail: {
      enabled: true,
      realTime: true,
      retention: 2555, // 7 years
      events: [
        'user.login',
        'user.logout',
        'user.failed_login',
        'user.password_change',
        'user.role_change',
        'data.access',
        'data.modification',
        'data.export',
        'system.configuration'
      ]
    },
    anomalyDetection: {
      enabled: true,
      suspiciousActivities: [
        'multiple_failed_logins',
        'unusual_access_patterns',
        'off_hours_access',
        'large_data_exports',
        'privilege_escalation'
      ],
      alertLevels: ['low', 'medium', 'high', 'critical']
    }
  },

  // Security Configuration
  security: {
    encryption: {
      algorithm: 'AES-256-GCM',
      keyRotation: 2592000, // 30 days
      saltLength: 32
    },
    api: {
      rateLimit: {
        windowMs: 900000, // 15 minutes
        maxRequests: 1000,
        skipSuccessfulRequests: false
      },
      cors: {
        origin: ['https://healthcare.app'],
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
      }
    },
    intrusion: {
      enabled: true,
      maxFailedAttempts: 5,
      lockoutTime: 3600,
      monitorIPReputation: true
    }
  },

  // Support System Configuration
  support: {
    tiers: {
      TIER_1: {
        name: 'Help Desk',
        responseTime: 1800, // 30 minutes
        escalationTime: 3600 // 1 hour
      },
      TIER_2: {
        name: 'Technical Support',
        responseTime: 900, // 15 minutes
        escalationTime: 1800 // 30 minutes
      },
      TIER_3: {
        name: 'Medical IT Support',
        responseTime: 300, // 5 minutes
        escalationTime: 900 // 15 minutes
      }
    },
    escalation: {
      medical_emergency: {
        priority: 'CRITICAL',
        responseTime: 60, // 1 minute
        contactRoles: ['hospital_admin', 'security_admin']
      },
      system_down: {
        priority: 'HIGH',
        responseTime: 300, // 5 minutes
        contactRoles: ['security_admin']
      },
      security_incident: {
        priority: 'CRITICAL',
        responseTime: 300, // 5 minutes
        contactRoles: ['security_admin', 'compliance_officer']
      }
    }
  },

  // Database Configuration
  database: {
    connectionPool: {
      minConnections: 10,
      maxConnections: 100,
      acquireTimeout: 60000,
      createTimeout: 30000,
      destroyTimeout: 5000,
      idleTimeout: 30000,
      reapInterval: 1000,
      createRetryInterval: 200
    },
    backup: {
      enabled: true,
      schedule: '0 2 * * *', // Daily at 2 AM
      retention: 2555 // 7 years
    }
  },

  // Notification Configuration
  notifications: {
    channels: ['email', 'sms', 'in_app'],
    events: [
      'user.registered',
      'user.approved',
      'user.suspended',
      'role.changed',
      'password.expiring',
      'suspicious_activity',
      'system.maintenance'
    ]
  },

  // Performance Configuration
  performance: {
    caching: {
      enabled: true,
      ttl: 300, // 5 minutes
      maxSize: 1000,
      compression: true
    },
    optimization: {
      enableCompression: true,
      enableMinification: true,
      enableCdn: true
    }
  }
};