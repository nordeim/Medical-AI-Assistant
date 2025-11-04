/**
 * Information Security Policy
 * Healthcare Data Protection and HIPAA Compliance
 * Version: 2.0
 * Last Updated: October 2024
 */

const SecurityPolicy = {
  policy_info: {
    title: 'Information Security Policy',
    version: '2.0',
    effective_date: '2024-10-01',
    next_review_date: '2025-10-01',
    owner: 'Chief Information Security Officer',
    approval_authority: 'Chief Executive Officer',
    classification: 'Internal',
    compliance_requirements: ['HIPAA', 'SOC2', 'ISO27001']
  },

  scope: {
    description: 'This policy applies to all workforce members, contractors, volunteers, students, and other persons who have access to Protected Health Information (PHI) or organizational information systems.',
    systems_covered: [
      'All computer systems and networks',
      'Mobile devices and portable media',
      'Cloud services and third-party applications',
      'Physical facilities and workspaces',
      'Communication systems'
    ],
    data_types: [
      'Protected Health Information (PHI)',
      'Personally Identifiable Information (PII)',
      'Financial information',
      'Proprietary business information',
      'Authentication credentials'
    ]
  },

  policy_statements: {
    // Administrative Safeguards
    security_officer: {
      requirement: '164.308(a)(2)',
      statement: 'The organization shall designate a security official responsible for developing and implementing security policies and procedures.',
      implementation: 'Chief Information Security Officer (CISO) with full authority and organizational support'
    },

    workforce_security: {
      requirement: '164.308(a)(3)',
      statement: 'Implement policies and procedures for authorizing and supervising workforce members who access PHI.',
      implementation: 'Role-based access control with principle of least privilege'
    },

    information_access_management: {
      requirement: '164.308(a)(4)',
      statement: 'Implement policies and procedures for authorizing access to PHI.',
      implementation: 'RBAC system with regular access reviews and automated provisioning'
    },

    security_awareness_training: {
      requirement: '164.308(a)(5)',
      statement: 'Implement a security awareness and training program for all workforce members.',
      implementation: 'Annual mandatory training with role-specific modules and quarterly updates'
    },

    incident_response: {
      requirement: '164.308(a)(6)',
      statement: 'Implement procedures for responding to security incidents.',
      implementation: '24/7 incident response team with automated detection and escalation'
    },

    contingency_plan: {
      requirement: '164.308(a)(7)',
      statement: 'Establish data backup and disaster recovery procedures.',
      implementation: 'Daily automated backups with 7-year retention and tested recovery procedures'
    },

    evaluation: {
      requirement: '164.308(a)(8)',
      statement: 'Conduct regular evaluation of security measures.',
      implementation: 'Quarterly security assessments and annual penetration testing'
    },

    // Physical Safeguards
    facility_access: {
      requirement: '164.310(a)(1)',
      statement: 'Implement physical safeguards for facilities containing PHI.',
      implementation: 'Biometric access controls with visitor logging and security monitoring'
    },

    workstation_security: {
      requirement: '164.310(a)(2)',
      statement: 'Implement policies for workstation use and access.',
      implementation: 'Clean desk policy with automatic screen locking after inactivity'
    },

    device_controls: {
      requirement: '164.310(d)(1)',
      statement: 'Implement policies for device and media disposal and reuse.',
      implementation: 'Secure data destruction with certificates of destruction'
    },

    // Technical Safeguards
    access_control: {
      requirement: '164.312(a)(1)',
      statement: 'Implement technical policies and procedures for PHI access.',
      implementation: 'Multi-factor authentication with role-based permissions and session management'
    },

    audit_controls: {
      requirement: '164.312(b)',
      statement: 'Implement audit controls for system activity.',
      implementation: 'Comprehensive audit logging with 7-year retention and tamper-proof storage'
    },

    integrity: {
      requirement: '164.312(c)(1)',
      statement: 'Implement controls to protect PHI from unauthorized alteration.',
      implementation: 'Digital signatures and integrity verification for all PHI transactions'
    },

    transmission_security: {
      requirement: '164.312(d)',
      statement: 'Implement technical security measures for data in transit.',
      implementation: 'TLS 1.3 encryption for all network communications'
    },

    encryption: {
      requirement: '164.312(e)(1)',
      statement: 'Implement encryption for PHI at rest.',
      implementation: 'AES-256 encryption for all PHI in databases and file systems'
    }
  },

  roles_and_responsibilities: {
    chief_executive_officer: [
      'Overall accountability for security program',
      'Approve security policies and procedures',
      'Ensure adequate resources for security initiatives'
    ],

    chief_information_security_officer: [
      'Develop and implement security policies',
      'Manage security operations and incident response',
      'Report on security posture to executive leadership'
    ],

    security_team: [
      'Monitor security systems and respond to incidents',
      'Conduct security assessments and penetration testing',
      'Maintain security controls and configurations'
    ],

    it_operations: [
      'Implement and maintain technical security controls',
      'Manage access provisioning and deprovisioning',
      'Backup and recovery operations'
    ],

    legal_compliance: [
      'Ensure regulatory compliance with HIPAA and other requirements',
      'Manage business associate agreements',
      'Coordinate breach notification procedures'
    ],

    human_resources: [
      'Include security requirements in job descriptions',
      'Conduct background checks for workforce members',
      'Manage security training and awareness programs'
    ],

    department_managers: [
      'Ensure workforce members complete required training',
      'Report security incidents and concerns',
      'Participate in access review processes'
    ],

    workforce_members: [
      'Complete required security training',
      'Follow security policies and procedures',
      'Report security incidents immediately',
      'Protect PHI and organizational information'
    ]
  },

  specific_policies: {
    access_control_policy: {
      title: 'Access Control Policy',
      sections: [
        {
          title: 'User Account Management',
          requirements: [
            'All user accounts must be created through approved provisioning process',
            'Access is granted based on job function and principle of least privilege',
            'User access reviews conducted quarterly',
            'Privileged access requires additional approval and monitoring'
          ]
        },
        {
          title: 'Password Requirements',
          requirements: [
            'Minimum 12 characters with complexity requirements',
            'Passwords changed every 90 days',
            'Previous 5 passwords cannot be reused',
            'Multi-factor authentication required for privileged accounts'
          ]
        },
        {
          title: 'Session Management',
          requirements: [
            'Sessions timeout after 30 minutes of inactivity',
            'Maximum concurrent sessions limited by role',
            'Remote access requires VPN and multi-factor authentication'
          ]
        }
      ]
    },

    data_protection_policy: {
      title: 'Data Protection Policy',
      sections: [
        {
          title: 'Data Classification',
          requirements: [
            'All data must be classified according to sensitivity level',
            'PHI and other sensitive data requires encryption at rest and in transit',
            'Data sharing requires approval and appropriate safeguards'
          ]
        },
        {
          title: 'Data Retention',
          requirements: [
            'PHI retained for minimum 7 years as required by HIPAA',
            'Data destruction follows secure disposal procedures',
            'Legal holds supersede normal retention schedules'
          ]
        }
      ]
    },

    incident_response_policy: {
      title: 'Incident Response Policy',
      sections: [
        {
          title: 'Incident Classification',
          requirements: [
            'Security incidents classified by severity (Low, Medium, High, Critical)',
            'Critical incidents require immediate escalation',
            'All incidents must be documented and tracked'
          ]
        },
        {
          title: 'Response Procedures',
          requirements: [
            'Incident response team available 24/7',
            'Containment procedures executed within 15 minutes for critical incidents',
            'Forensic evidence preserved for investigation',
            'Stakeholders notified according to escalation procedures'
          ]
        }
      ]
    },

    vendor_management_policy: {
      title: 'Vendor Management Policy',
      sections: [
        {
          title: 'Third-Party Security',
          requirements: [
            'Security assessments required for all vendors handling PHI',
            'Business Associate Agreements required for PHI access',
            'Regular monitoring of vendor security posture',
            'Incident notification requirements in all contracts'
          ]
        }
      ]
    }
  },

  enforcement: {
    violations: [
      'Unauthorized access to PHI or systems',
      'Sharing passwords or accessing systems with another user\'s credentials',
      'Failure to report security incidents',
      'Bypassing or disabling security controls',
      'Unauthorized disclosure of PHI',
      'Failure to complete required security training'
    ],

    disciplinary_actions: {
      minor_violations: 'Verbal warning and retraining',
      moderate_violations: 'Written warning and probation',
      serious_violations: 'Suspension pending investigation',
      severe_violations: 'Immediate termination and legal action'
    }
  },

  exceptions: {
    process: 'Security policy exceptions require written approval from CISO',
    criteria: 'Exceptions granted only for business necessity with compensating controls',
    review: 'All exceptions reviewed quarterly for continued validity',
    documentation: 'Exception requests documented with risk assessment'
  },

  compliance_monitoring: {
    audits: [
      'Internal security assessments conducted quarterly',
      'Annual penetration testing by qualified external firm',
      'Annual HIPAA risk assessment',
      'SOC 2 Type II audit conducted annually'
    ],

    reporting: [
      'Monthly security metrics reported to executive team',
      'Quarterly compliance reports to board of directors',
      'Annual security program review with leadership',
      'Immediate notification of critical incidents to CEO'
    ]
  },

  policy_review: {
    frequency: 'This policy is reviewed annually or as needed',
    responsible_party: 'Chief Information Security Officer',
    approval_process: 'Policy changes require CISO approval and executive sign-off',
    communication: 'Policy updates communicated to all workforce members',
    training: 'Updated policies incorporated into security training program'
  },

  related_policies: [
    'Access Control Policy',
    'Data Protection Policy',
    'Incident Response Policy',
    'Vendor Management Policy',
    'Acceptable Use Policy',
    'Remote Access Policy',
    'Bring Your Own Device (BYOD) Policy'
  ],

  references: [
    'HIPAA Security Rule (45 CFR Part 164.312)',
    'HIPAA Privacy Rule (45 CFR Part 164.500)',
    'NIST Cybersecurity Framework',
    'ISO 27001 Information Security Management',
    'SOC 2 Trust Services Criteria'
  ]
};

module.exports = SecurityPolicy;