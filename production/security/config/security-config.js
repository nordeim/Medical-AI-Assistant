/**
 * Production-Grade Security Configuration Manager
 * Centralized configuration for all security and compliance components
 */

const path = require('path');

class SecurityConfig {
  constructor() {
    this.config = {
      // Application Information
      application: {
        name: 'Healthcare Security Platform',
        version: '1.0.0',
        environment: process.env.NODE_ENV || 'production',
        compliance_frameworks: ['HIPAA', 'SOC2', 'ISO27001'],
        data_classification: ['PHI', 'PII', 'Financial', 'Internal'],
        jurisdictions: ['US', 'EU']
      },

      // Access Control Configuration
      access_control: {
        session_timeout: 30 * 60 * 1000, // 30 minutes
        max_failed_attempts: 3,
        lockout_duration: 15 * 60 * 1000, // 15 minutes
        password_policy: {
          min_length: 12,
          require_uppercase: true,
          require_lowercase: true,
          require_numbers: true,
          require_special_chars: true,
          prevent_reuse: 5,
          max_age: 90 * 24 * 60 * 60 * 1000 // 90 days
        },
        mfa_required: {
          roles: ['admin', 'doctor', 'billing', 'it_support'],
          high_value_operations: true
        },
        role_definitions: {
          admin: {
            permissions: ['*'],
            restrictions: {
              max_concurrent_sessions: 1,
              allowed_ip_ranges: [],
              require_mfa: true
            }
          },
          doctor: {
            permissions: ['patient:read', 'patient:write', 'phi:access'],
            restrictions: {
              max_concurrent_sessions: 3,
              allowed_ip_ranges: [],
              require_mfa: true
            }
          },
          nurse: {
            permissions: ['patient:read', 'patient:write', 'phi:read'],
            restrictions: {
              max_concurrent_sessions: 2,
              allowed_ip_ranges: [],
              require_mfa: false
            }
          },
          billing: {
            permissions: ['billing:read', 'billing:write', 'financial:read'],
            restrictions: {
              max_concurrent_sessions: 3,
              allowed_ip_ranges: [],
              require_mfa: true
            }
          }
        }
      },

      // Audit Logging Configuration
      audit_logging: {
        enabled: true,
        retention_period: 7 * 365 * 24 * 60 * 60 * 1000, // 7 years for HIPAA
        max_log_size: 100 * 1024 * 1024, // 100MB
        encryption_enabled: true,
        compression_enabled: true,
        log_levels: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'],
        event_types: [
          'authentication',
          'authorization',
          'data_access',
          'system_changes',
          'security_incidents',
          'phi_access',
          'compliance_events'
        ],
        real_time_alerts: {
          critical_events: true,
          failed_logins: true,
          unauthorized_access: true,
          data_exfiltration: true
        }
      },

      // Encryption Configuration
      encryption: {
        algorithm: 'aes-256-gcm',
        key_derivation: 'pbkdf2',
        iterations: 100000,
        key_rotation_interval: 90 * 24 * 60 * 60 * 1000, // 90 days
        hsm_enabled: false, // Hardware Security Module
        key_backup: {
          enabled: true,
          locations: ['secure_vault', 'geographic_backup'],
          encryption: 'aes-256'
        },
        data_at_rest: {
          database_encryption: true,
          file_system_encryption: true,
          backup_encryption: true
        },
        data_in_transit: {
          tls_version: '1.3',
          certificate_validation: 'strict',
          cipher_suites: [
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256'
          ]
        }
      },

      // Penetration Testing Configuration
      penetration_testing: {
        enabled: true,
        schedule: {
          automated_scans: 'weekly',
          comprehensive_assessment: 'quarterly',
          external_audit: 'annually'
        },
        scope: {
          internal_networks: true,
          external_facing: true,
          web_applications: true,
          mobile_applications: true,
          wireless_networks: true,
          social_engineering: false // Requires special approval
        },
        tools: {
          nmap: { enabled: true, scripts: ['vuln', 'default'] },
          nikto: { enabled: true, timeout: '30s' },
          sqlmap: { enabled: true, level: 5, risk: 3 },
          dirb: { enabled: true, extensions: 'php,html,htm,js' },
          hydra: { enabled: true, threads: 16 }
        },
        reporting: {
          formats: ['PDF', 'HTML', 'JSON'],
          distribution: ['security_team', 'management', 'audit'],
          retention: '7 years'
        }
      },

      // Incident Response Configuration
      incident_response: {
        enabled: true,
        escalation_timeout: 30 * 60 * 1000, // 30 minutes
        auto_containment: false, // Manual approval required
        communication_channels: [
          'email',
          'sms',
          'slack',
          'pagerduty',
          'phone_call'
        ],
        response_teams: {
          security_team: {
            members: ['security@company.com'],
            on_call: true,
            escalation_level: 1
          },
          incident_commander: {
            members: ['incident.commander@company.com'],
            on_call: true,
            escalation_level: 2
          },
          legal_compliance: {
            members: ['legal@company.com'],
            on_call: false,
            escalation_level: 3
          }
        },
        playbooks: {
          data_breach: { severity: 'CRITICAL', estimated_duration: '4-8 hours' },
          malware: { severity: 'HIGH', estimated_duration: '2-6 hours' },
          unauthorized_access: { severity: 'HIGH', estimated_duration: '1-3 hours' },
          ddos: { severity: 'MEDIUM', estimated_duration: '30min-2hrs' }
        }
      },

      // Compliance Documentation Configuration
      compliance: {
        frameworks: {
          hipaa: {
            enabled: true,
            security_rule: true,
            privacy_rule: true,
            breach_notification_rule: true
          },
          soc2: {
            enabled: true,
            trust_services: ['Security', 'Availability', 'Confidentiality']
          },
          iso27001: {
            enabled: false, // Future implementation
            required_controls: []
          }
        },
        reporting: {
          frequency: 'quarterly',
          formats: ['PDF', 'HTML', 'JSON'],
          distribution: ['compliance_team', 'audit', 'management'],
          automated_generation: true
        },
        audits: {
          internal_frequency: 'quarterly',
          external_frequency: 'annually',
          preparation_time: '30 days',
          documentation_retention: '7 years'
        }
      },

      // Security Monitoring Configuration
      monitoring: {
        enabled: true,
        interval: 60 * 1000, // 1 minute
        real_time_alerts: true,
        alert_thresholds: {
          failed_logins: 5,
          unauthorized_access: 1,
          data_exfiltration: 1,
          malware_detection: 1
        },
        integrations: {
          siem: {
            enabled: true,
            endpoint: 'http://localhost:8080/siem',
            format: 'CEF'
          },
          splunk: {
            enabled: true,
            endpoint: 'https://splunk.company.com:8088',
            index: 'security'
          },
          elasticsearch: {
            enabled: true,
            endpoint: 'http://localhost:9200',
            index: 'security-logs-*'
          }
        },
        threat_intelligence: {
          enabled: true,
          feeds: [
            'alienvault_otx',
            'virustotal',
            'abuse_ch',
            'custom_feeds'
          ],
          update_frequency: 'hourly'
        }
      },

      // Network Security Configuration
      network_security: {
        firewalls: {
          enabled: true,
          rules: {
            default_deny: true,
            allow_internal: true,
            allow_admin_access: false,
            allow_established: true
          }
        },
        intrusion_detection: {
          enabled: true,
          mode: 'prevention',
          rules_update: 'daily'
        },
        network_segmentation: {
          enabled: true,
          zones: [
            { name: 'DMZ', description: 'Demilitarized Zone' },
            { name: 'Internal', description: 'Internal Network' },
            { name: 'PHI', description: 'PHI Processing Zone' },
            { name: 'Management', description: 'Management Network' }
          ]
        }
      },

      // Data Protection Configuration
      data_protection: {
        classification: {
          enabled: true,
          levels: ['Public', 'Internal', 'Confidential', 'Restricted']
        },
        data_loss_prevention: {
          enabled: true,
          policies: [
            {
              name: 'PHI Data Protection',
              pattern: 'PHI_DATA',
              action: 'block',
              encryption_required: true
            },
            {
              name: 'Credit Card Protection',
              pattern: 'CREDIT_CARD',
              action: 'mask',
              tokenization: true
            }
          ]
        },
        backup_strategy: {
          frequency: 'daily',
          retention: '7 years',
          encryption: 'aes-256',
          testing: 'monthly',
          geographic_distribution: true
        }
      },

      // Security Policies Configuration
      policies: {
        security_policy: {
          version: '2.0',
          last_reviewed: '2024-10-01',
          next_review: '2025-10-01',
          approval_required: true,
          acknowledgment_required: true
        },
        access_control_policy: {
          principle_of_least_privilege: true,
          role_based_access: true,
          periodic_access_review: 'quarterly'
        },
        incident_response_policy: {
          classification_levels: ['Low', 'Medium', 'High', 'Critical'],
          notification_requirements: 'immediate',
          documentation_requirements: 'comprehensive'
        }
      },

      // Third-Party Risk Management
      third_party: {
        vendor_assessment: {
          required: true,
          frequency: 'annually',
          security_questionnaire: true,
          penetration_testing: 'risk_based'
        },
        business_associate_agreements: {
          required_for_phi: true,
          review_frequency: 'annually',
          compliance_validation: true
        },
        vendor_monitoring: {
          security_posture: true,
          compliance_status: true,
          incident_notification: true
        }
      },

      // Security Training and Awareness
      training: {
        mandatory_training: {
          frequency: 'annually',
          completion_required: true,
          topics: [
            'HIPAA Privacy and Security',
            'Information Security Awareness',
            'Phishing and Social Engineering',
            'Data Handling and Classification',
            'Incident Response Procedures'
          ]
        },
        role_specific_training: {
          admin: ['System Administration Security', 'Privileged Access Management'],
          developer: ['Secure Coding Practices', 'Application Security Testing'],
          healthcare_worker: ['PHI Handling', 'Patient Privacy']
        },
        awareness_campaigns: {
          phishing_simulation: 'quarterly',
          security_tips: 'monthly',
          threat_updates: 'as_needed'
        }
      },

      // Business Continuity and Disaster Recovery
      business_continuity: {
        rto_rpo: {
          critical_systems: { rto: '4 hours', rpo: '1 hour' },
          important_systems: { rto: '24 hours', rpo: '4 hours' },
          non_critical_systems: { rto: '72 hours', rpo: '24 hours' }
        },
        backup_testing: {
          frequency: 'quarterly',
          scope: 'full_system',
          documentation: 'required'
        },
        disaster_recovery: {
          hot_site: true,
          cold_site: true,
          geographic_distribution: true,
          testing_frequency: 'annually'
        }
      }
    };

    // Validate configuration
    this.validateConfiguration();
  }

  /**
   * Validate security configuration
   */
  validateConfiguration() {
    const requiredSections = [
      'access_control',
      'audit_logging',
      'encryption',
      'incident_response',
      'compliance',
      'monitoring'
    ];

    for (const section of requiredSections) {
      if (!this.config[section]) {
        throw new Error(`Missing required configuration section: ${section}`);
      }
    }

    // Validate specific settings
    this.validateAccessControl();
    this.validateEncryption();
    this.validateCompliance();

    console.log('[CONFIG] Security configuration validated successfully');
  }

  /**
   * Validate access control configuration
   */
  validateAccessControl() {
    const ac = this.config.access_control;

    if (ac.session_timeout < 5 * 60 * 1000) {
      console.warn('[CONFIG] Warning: Session timeout is very short');
    }

    if (ac.max_failed_attempts > 5) {
      console.warn('[CONFIG] Warning: High failed login attempts allowed');
    }

    if (!ac.password_policy || ac.password_policy.min_length < 8) {
      console.warn('[CONFIG] Warning: Password policy may be too weak');
    }
  }

  /**
   * Validate encryption configuration
   */
  validateEncryption() {
    const enc = this.config.encryption;

    if (!enc.algorithm.includes('256')) {
      console.warn('[CONFIG] Warning: Encryption may not meet HIPAA requirements');
    }

    if (enc.iterations < 10000) {
      console.warn('[CONFIG] Warning: Key derivation iterations may be too low');
    }

    if (!enc.data_at_rest.database_encryption) {
      console.warn('[CONFIG] Warning: Database encryption not enabled');
    }
  }

  /**
   * Validate compliance configuration
   */
  validateCompliance() {
    const comp = this.config.compliance;

    if (!comp.frameworks.hipaa?.enabled) {
      console.warn('[CONFIG] Warning: HIPAA compliance not enabled');
    }

    if (comp.reporting.frequency !== 'quarterly') {
      console.warn('[CONFIG] Warning: Compliance reporting frequency may not meet requirements');
    }
  }

  /**
   * Get configuration value by path
   */
  get(path) {
    return this.getNestedValue(this.config, path);
  }

  /**
   * Get nested configuration value
   */
  getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  /**
   * Update configuration value
   */
  set(path, value) {
    const keys = path.split('.');
    const lastKey = keys.pop();
    const target = this.getNestedValue(this.config, keys.join('.'));
    
    if (target) {
      target[lastKey] = value;
    }
  }

  /**
   * Get environment-specific configuration
   */
  getEnvironmentConfig() {
    const env = this.config.application.environment;
    
    if (env === 'development') {
      return {
        ...this.config,
        audit_logging: {
          ...this.config.audit_logging,
          retention_period: 30 * 24 * 60 * 60 * 1000 // 30 days for dev
        },
        incident_response: {
          ...this.config.incident_response,
          auto_containment: false
        },
        monitoring: {
          ...this.config.monitoring,
          real_time_alerts: false
        }
      };
    }
    
    if (env === 'staging') {
      return {
        ...this.config,
        access_control: {
          ...this.config.access_control,
          mfa_required: {
            roles: ['admin'], // Only admin in staging
            high_value_operations: false
          }
        }
      };
    }
    
    return this.config; // Production configuration
  }

  /**
   * Get configuration for specific component
   */
  getComponentConfig(component) {
    const componentMap = {
      'rbac-manager': this.config.access_control,
      'audit-logger': this.config.audit_logging,
      'phi-encryption': this.config.encryption,
      'pen-test-framework': this.config.penetration_testing,
      'incident-response': this.config.incident_response,
      'compliance-docs': this.config.compliance,
      'security-monitor': this.config.monitoring
    };

    return componentMap[component] || {};
  }

  /**
   * Generate configuration summary
   */
  getConfigurationSummary() {
    return {
      application: this.config.application,
      security_controls: {
        access_control: this.config.access_control.enabled !== false,
        audit_logging: this.config.audit_logging.enabled,
        encryption: this.config.encryption.data_at_rest.database_encryption,
        incident_response: this.config.incident_response.enabled,
        compliance_reporting: this.config.compliance.reporting.automated_generation,
        security_monitoring: this.config.monitoring.enabled
      },
      compliance_frameworks: this.config.application.compliance_frameworks,
      encryption_level: this.config.encryption.algorithm,
      data_protection: this.config.data_protection.classification.enabled,
      network_security: this.config.network_security.firewalls.enabled,
      monitoring_integrations: Object.keys(this.config.monitoring.integrations).filter(
        key => this.config.monitoring.integrations[key].enabled
      ).length,
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Export configuration for external systems
   */
  exportConfiguration(format = 'json') {
    const config = this.getEnvironmentConfig();
    
    switch (format) {
      case 'json':
        return JSON.stringify(config, null, 2);
      
      case 'yaml':
        return this.convertToYAML(config);
      
      case 'env':
        return this.convertToEnvVars(config);
      
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  /**
   * Convert configuration to YAML format
   */
  convertToYAML(obj, indent = 0) {
    const spaces = '  '.repeat(indent);
    let yaml = '';
    
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'object' && value !== null) {
        yaml += `${spaces}${key}:\n`;
        yaml += this.convertToYAML(value, indent + 1);
      } else {
        yaml += `${spaces}${key}: ${value}\n`;
      }
    }
    
    return yaml;
  }

  /**
   * Convert configuration to environment variables
   */
  convertToEnvVars(obj, prefix = '') {
    let envVars = '';
    
    for (const [key, value] of Object.entries(obj)) {
      const envName = prefix ? `${prefix}_${key.toUpperCase()}` : key.toUpperCase();
      
      if (typeof value === 'object' && value !== null) {
        envVars += this.convertToEnvVars(value, envName);
      } else {
        envVars += `${envName}=${value}\n`;
      }
    }
    
    return envVars;
  }

  /**
   * Load configuration from file
   */
  async loadFromFile(filePath) {
    const fs = require('fs').promises;
    const ext = path.extname(filePath).toLowerCase();
    
    const content = await fs.readFile(filePath, 'utf8');
    
    let config;
    switch (ext) {
      case '.json':
        config = JSON.parse(content);
        break;
      case '.yaml':
      case '.yml':
        // In production, use js-yaml library
        console.warn('[CONFIG] YAML loading not implemented, using JSON');
        config = JSON.parse(content);
        break;
      default:
        throw new Error(`Unsupported configuration file format: ${ext}`);
    }
    
    // Merge with default configuration
    this.config = { ...this.config, ...config };
    this.validateConfiguration();
    
    console.log(`[CONFIG] Configuration loaded from: ${filePath}`);
  }

  /**
   * Save configuration to file
   */
  async saveToFile(filePath) {
    const fs = require('fs').promises;
    const content = this.exportConfiguration('json');
    
    await fs.writeFile(filePath, content);
    console.log(`[CONFIG] Configuration saved to: ${filePath}`);
  }

  /**
   * Get configuration validation report
   */
  getValidationReport() {
    const report = {
      timestamp: new Date().toISOString(),
      overall_status: 'VALID',
      warnings: [],
      errors: [],
      recommendations: []
    };

    // Check for security best practices
    if (!this.config.encryption.data_in_transit.tls_version) {
      report.warnings.push('TLS version not specified');
    }

    if (this.config.access_control.session_timeout > 60 * 60 * 1000) {
      report.warnings.push('Session timeout is longer than recommended');
    }

    if (!this.config.monitoring.threat_intelligence.enabled) {
      report.recommendations.push('Enable threat intelligence feeds');
    }

    if (this.config.compliance.reporting.automated_generation) {
      report.recommendations.push('Consider automated compliance reporting');
    }

    // Set overall status
    if (report.errors.length > 0) {
      report.overall_status = 'INVALID';
    } else if (report.warnings.length > 0) {
      report.overall_status = 'WARNING';
    }

    return report;
  }
}

module.exports = SecurityConfig;