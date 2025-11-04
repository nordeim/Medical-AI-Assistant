/**
 * Incident Response Procedures and Playbook
 * Healthcare Security Incident Response Framework
 * Version: 1.3
 */

const IncidentResponseProcedures = {
  document_info: {
    title: 'Incident Response Procedures and Playbook',
    version: '1.3',
    effective_date: '2024-10-01',
    next_review_date: '2025-08-30',
    owner: 'Security Team',
    approval_authority: 'Chief Information Security Officer',
    classification: 'Internal - Security Sensitive',
    related_policies: ['Information Security Policy', 'Business Continuity Plan']
  },

  incident_classification: {
    severity_levels: {
      critical: {
        level: 'P1 - Critical',
        description: 'Major security breach with immediate business impact',
        examples: [
          'Data breach involving PHI of 500+ individuals',
          'Complete system compromise',
          'Ransomware attack affecting operations',
          'Network infrastructure breach',
          'Insider threat with data exfiltration'
        ],
        response_time: 'Immediate (within 15 minutes)',
        notification_required: ['CEO', 'CISO', 'Legal', 'Compliance', 'Board'],
        business_impact: 'Severe - operations may be halted'
      },
      high: {
        level: 'P2 - High',
        description: 'Significant security incident with notable impact',
        examples: [
          'Unauthorized access to PHI of 50-500 individuals',
          'Malware infection on critical systems',
          'Successful phishing attack with credential theft',
          'Data exfiltration attempts',
          'Website defacement or DDoS attack'
        ],
        response_time: 'Within 1 hour',
        notification_required: ['CISO', 'IT Manager', 'Legal'],
        business_impact: 'Moderate to severe - operations may be degraded'
      },
      medium: {
        level: 'P3 - Medium',
        description: 'Security incident with limited impact',
        examples: [
          'Failed intrusion attempts',
          'Policy violations by workforce members',
          'Lost or stolen devices without confirmed data access',
          'Suspicious network activity',
          'Minor data spills or misconfigurations'
        ],
        response_time: 'Within 4 hours',
        notification_required: ['Security Team', 'IT Manager'],
        business_impact: 'Low to moderate - minimal operational impact'
      },
      low: {
        level: 'P4 - Low',
        description: 'Minor security events and informational alerts',
        examples: [
          'Automated security alerts with no confirmed threat',
          'Routine security scanning results',
          'Security awareness questions',
          'Minor policy clarifications',
          'Preventive security measures'
        ],
        response_time: 'Within 24 hours',
        notification_required: ['Security Team'],
        business_impact: 'Minimal - no operational impact'
      }
    }
  },

  response_team_structure: {
    incident_commander: {
      role: 'Overall incident coordination and decision making',
      responsibilities: [
        'Declare incident severity level',
        'Coordinate response activities',
        'Make critical decisions',
        'Communicate with executive leadership',
        'Authorize external communications'
      ],
      primary: 'Security Team Lead',
      backup: 'IT Manager',
      contact: 'Available 24/7 via on-call rotation'
    },

    security_analysts: {
      role: 'Technical investigation and containment',
      responsibilities: [
        'Analyze security events and indicators',
        'Implement containment measures',
        'Collect and preserve forensic evidence',
        'Document technical findings',
        'Monitor for additional threats'
      ],
      primary: 'Security Operations Team',
      contact: 'Available 24/7'
    },

    forensics_specialist: {
      role: 'Digital forensics and evidence collection',
      responsibilities: [
        'Preserve digital evidence',
        'Conduct forensic analysis',
        'Maintain chain of custody',
        'Prepare forensic reports',
        'Testify if legal proceedings occur'
      ],
      primary: 'External forensic consultant',
      backup: 'Security team member with forensics training'
    },

    legal_counsel: {
      role: 'Legal guidance and regulatory compliance',
      responsibilities: [
        'Provide legal advice on incident response',
        'Determine regulatory notification requirements',
        'Coordinate with law enforcement',
        'Review public communications',
        'Manage legal holds and evidence'
      ],
      primary: 'Corporate Legal Department',
      contact: 'Available during business hours'
    },

    compliance_officer: {
      role: 'Regulatory compliance and notification',
      responsibilities: [
        'Determine HIPAA breach notification requirements',
        'Coordinate with regulatory bodies',
        'Manage compliance documentation',
        'Coordinate with external auditors',
        'Track regulatory deadlines'
      ],
      primary: 'Compliance Team',
      contact: 'Available during business hours'
    },

    communications_lead: {
      role: 'Internal and external communications',
      responsibilities: [
        'Draft internal incident communications',
        'Prepare external statements',
        'Coordinate with PR team',
        'Manage stakeholder updates',
        'Handle media inquiries'
      ],
      primary: 'Corporate Communications',
      backup: 'Human Resources'
    }
  },

  incident_response_phases: {
    phase_1_preparation: {
      duration: 'Ongoing',
      activities: [
        'Maintain incident response team roster and contact information',
        'Ensure incident response tools and technologies are current',
        'Conduct regular tabletop exercises and simulations',
        'Maintain updated contact lists for stakeholders',
        'Pre-draft notification templates and procedures',
        'Establish communication channels (conference bridges, chat rooms)',
        'Train incident response team members',
        'Maintain incident response documentation and playbooks'
      ],
      success_criteria: [
        'All team members trained and aware of roles',
        'Response tools tested and operational',
        'Communication channels established and tested',
        'Documentation current and accessible'
      ]
    },

    phase_2_detection_analysis: {
      duration: 'Variable - minutes to hours',
      activities: [
        'Identify and validate security incidents',
        'Classify incident severity and type',
        'Collect initial information and context',
        'Analyze scope and potential impact',
        'Determine if incident meets breach criteria',
        'Establish incident response team',
        'Initiate incident tracking and documentation',
        'Begin stakeholder notification process'
      ],
      decision_points: [
        'Is this a legitimate security incident?',
        'What is the preliminary severity level?',
        'Are PHI or other sensitive data involved?',
        'Should external parties be notified?',
        'What immediate actions are required?'
      ],
      documentation_required: [
        'Incident detection date and time',
        'Initial assessment of scope and severity',
        'Systems and data potentially affected',
        'Preliminary impact assessment',
        'Response team activation',
        'Initial containment measures'
      ]
    },

    phase_3_containment_eradication_recovery: {
      duration: 'Variable - hours to days',
      activities: [
        'Implement immediate containment measures',
        'Prevent further damage or data exposure',
        'Preserve evidence for investigation',
        'Eradicate threat actor and malicious code',
        'Restore systems to normal operations',
        'Verify system integrity and security',
        'Implement additional security measures',
        'Monitor for signs of continued threat activity'
      ],
      containment_strategies: [
        'Network isolation of affected systems',
        'Account lockdown and password resets',
        'Firewall rule changes',
        'Antivirus and malware removal',
        'System reimaging if necessary',
        'Certificate and key revocation'
      ],
      success_criteria: [
        'Threat contained and neutralized',
        'All affected systems restored and secured',
        'No signs of continued malicious activity',
        'Normal operations resumed',
        'Additional monitoring in place'
      ]
    },

    phase_4_post_incident_activity: {
      duration: 'Variable - days to weeks',
      activities: [
        'Conduct post-incident review and lessons learned',
        'Document complete incident timeline',
        'Prepare regulatory notifications if required',
        'Update incident response procedures based on lessons learned',
        'Conduct team training on incident findings',
        'Implement additional security controls',
        'Monitor for long-term effects and reoccurrence',
        'Close incident and archive documentation'
      ],
      deliverables: [
        'Post-incident report',
        'Lessons learned document',
        'Updated procedures and policies',
        'Regulatory notifications (if applicable)',
        'Training materials updates',
        'Security improvement recommendations'
      ],
      success_criteria: [
        'All stakeholders satisfied with response',
        'Regulatory requirements met',
        'Lessons learned implemented',
        'Security posture improved',
        'Documentation complete and archived'
      ]
    }
  },

  specific_incident_playbooks: {
    data_breach: {
      triggers: [
        'Unauthorized access to PHI',
        'Lost or stolen devices containing PHI',
        'Accidental disclosure of PHI',
        'System compromise with data access',
        'Insider threat with data exfiltration'
      ],
      immediate_actions: [
        'Isolate affected systems to prevent further access',
        'Preserve forensic evidence including logs',
        'Notify incident commander and legal team',
        'Begin impact assessment to determine number of individuals affected',
        'Document all actions taken',
        'Do NOT notify affected individuals until legal approval'
      ],
      assessment_criteria: [
        'Types of PHI involved (demographics, medical records, financial)',
        'Number of individuals affected',
        'Whether PHI was actually acquired or viewed',
        'Risk of harm to affected individuals',
        'Likelihood of re-identification',
        'Whether breach was internal or external'
      ],
      notification_requirements: {
        internal: ['CEO', 'CISO', 'Legal', 'Compliance', 'IT Manager'],
        regulatory: ['HHS OCR if 500+ affected', 'State AG if applicable'],
        external: ['Affected individuals (required)', 'Media (if required)'],
        timeline: '60 days from discovery for HIPAA notification'
      },
      recovery_steps: [
        'Identify and fix the vulnerability that allowed the breach',
        'Implement additional safeguards to prevent recurrence',
        'Conduct comprehensive security review',
        'Provide credit monitoring if financial data involved',
        'Update policies and procedures'
      ]
    },

    ransomware_attack: {
      triggers: [
        'Systems encrypted with ransom demands',
        'Files inaccessible due to encryption',
        'Ransom notes found on systems',
        'Suspicious file extensions',
        'Unusual system behavior indicating malware'
      ],
      immediate_actions: [
        'Disconnect affected systems from network immediately',
        'Preserve evidence including ransom notes',
        'Notify incident commander and executive leadership',
        'Do NOT attempt to negotiate or pay ransom without legal approval',
        'Begin assessment of backup systems and data integrity',
        'Document ransom demands and communications exactly as received'
      ],
      business_continuity: [
        'Activate business continuity plan immediately',
        'Use backup systems to maintain operations',
        'Implement manual procedures if necessary',
        'Communicate with patients/clients about service delays',
        'Coordinate with insurance carrier if cyber coverage exists'
      ],
      law_enforcement: [
        'Notify FBI and local law enforcement',
        'Provide all evidence and documentation',
        'Coordinate investigation activities',
        'Do not publicize attack until approved by authorities'
      ],
      recovery_steps: [
        'Clean rebuild all affected systems from known good backups',
        'Implement additional endpoint protection',
        'Enhance network segmentation',
        'Review and update backup strategies',
        'Conduct comprehensive security assessment'
      ]
    },

    insider_threat: {
      triggers: [
        'Unusual access patterns by authorized users',
        'Large downloads or data exports',
        'Access to systems outside normal job requirements',
        'Policy violations or suspicious behavior',
        'Whistleblower reports of misconduct'
      ],
      investigation_approach: [
        'Quietly gather evidence without alerting the individual',
        'Review audit logs for unusual activity patterns',
        'Interview supervisors and colleagues if appropriate',
        'Coordinate with Human Resources',
        'Preserve evidence including emails and file access logs'
      ],
      containment_actions: [
        'Immediately suspend system access if threat is active',
        'Preserve evidence and audit logs',
        'Coordinate with law enforcement if criminal activity suspected',
        'Prepare termination procedures if employee separation required',
        'Secure physical access and collect badges/keys'
      ],
      legal_considerations: [
        'Consult with legal team before any confrontations',
        'Ensure compliance with employment laws',
        'Preserve evidence for potential legal proceedings',
        'Coordinate with law enforcement if criminal activity',
        'Prepare for potential litigation'
      ]
    }
  },

  communication_procedures: {
    internal_communication: {
      incident_commander: [
        'Notify executive leadership immediately for P1/P2 incidents',
        'Provide regular status updates every 2-4 hours during active incidents',
        'Escalate issues requiring executive decisions promptly',
        'Coordinate media response and public communications'
      ],

      it_security_team: [
        'Maintain real-time communication during response',
        'Document all technical actions and decisions',
        'Provide regular technical status updates',
        'Escalate technical issues requiring additional resources'
      ],

      department_managers: [
        'Inform affected departments of service impacts',
        'Coordinate workforce communications and support',
        'Provide status updates to department staff',
        'Support business continuity efforts'
      ]
    },

    external_communication: {
      regulatory_bodies: [
        'HIPAA breach notifications to HHS OCR (60 days)',
        'State notification requirements (varies by state)',
        'Industry-specific reporting requirements',
        'Coordinate all regulatory communications through legal'
      ],

      affected_individuals: [
        'Prepare clear, factual communication about the incident',
        'Include steps individuals can take to protect themselves',
        'Provide contact information for questions',
        'Coordinate with legal and compliance on timing and content'
      ],

      media_public: [
        'Prepare official statement approved by legal and PR',
        'Designate single spokesperson for media inquiries',
        'Coordinate with corporate communications team',
        'Monitor social media for false information'
      ]
    }
  },

  documentation_requirements: {
    incident_log: [
      'Date and time of initial detection',
      'Initial reporter and contact information',
      'Initial assessment of incident type and severity',
      'Systems and data potentially affected',
      'Response team members activated',
      'Timeline of all significant actions taken',
      'Decisions made and rationale',
      'Communication log including recipients and times',
      'Evidence collected and chain of custody',
      'Lessons learned and improvement recommendations'
    ],

    evidence_preservation: [
      'System logs and audit trails',
      'Network traffic captures',
      'Disk images and memory dumps',
      'Email communications',
      'Phone call logs and recordings',
      'Physical evidence (if applicable)',
      'Chain of custody documentation',
      'Forensic analysis reports'
    ]
  },

  metrics_and_reporting: {
    response_metrics: [
      'Time from detection to initial response',
      'Time from detection to containment',
      'Time from detection to eradication',
      'Time from detection to full recovery',
      'Number of systems affected',
      'Number of individuals affected (if data breach)',
      'Total cost of incident response and recovery',
      'Business operations downtime'
    ],

    monthly_reporting: [
      'Total incidents by severity level',
      'Average response times',
      'Most common incident types',
      'System availability and uptime',
      'Security training completion rates',
      'Incident trends and patterns',
      'Improvement initiatives and status'
    ]
  },

  continuous_improvement: {
    tabletop_exercises: [
      'Conduct quarterly tabletop exercises',
      'Test different incident scenarios',
      'Include all relevant stakeholders',
      'Document gaps and improvement areas',
      'Update procedures based on exercise findings'
    ],

    lessons_learned: [
      'Conduct lessons learned session after each major incident',
      'Identify process improvements',
      'Update training materials',
      'Implement additional security controls',
      'Share knowledge across security team'
    ],

    training_updates: [
      'Incorporate lessons learned into training programs',
      'Update role-specific training based on incidents',
      'Conduct refresher training annually',
      'Test knowledge through tabletop exercises',
      'Maintain training records and certifications'
    ]
  }
};

module.exports = IncidentResponseProcedures;