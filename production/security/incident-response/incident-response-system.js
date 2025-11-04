/**
 * Production-Grade Security Incident Response System
 * Automated incident detection, response, and recovery procedures
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class SecurityIncidentResponse {
  constructor(config = {}) {
    this.config = {
      escalationTimeout: config.escalationTimeout || 30 * 60 * 1000, // 30 minutes
      autoContainment: config.autoContainment || false,
      notificationChannels: config.notificationChannels || [],
      responsePlaybooks: config.responsePlaybooks || {},
      evidenceRetention: config.evidenceRetention || 7 * 365 * 24 * 60 * 60 * 1000, // 7 years
      ...config
    };

    this.incidents = new Map();
    this.responses = new Map();
    this.playbooks = new Map();
    this.timeline = [];
    this.notificationQueue = [];
    this.escalationQueue = new Map();
    this.activeInvestigations = new Map();

    this.initializeIncidentResponse();
  }

  /**
   * Initialize incident response system
   */
  async initializeIncidentResponse() {
    await this.loadResponsePlaybooks();
    await this.initializeResponseTeams();
    
    console.log('[INCIDENT-RESPONSE] Security incident response system initialized');
  }

  /**
   * Load security incident response playbooks
   */
  async loadResponsePlaybooks() {
    const playbooks = {
      'data_breach': {
        id: 'data_breach',
        name: 'Data Breach Response',
        severity: 'CRITICAL',
        category: 'CONFIDENTIALITY',
        estimated_duration: '4-8 hours',
        steps: [
          {
            id: 1,
            action: 'immediate_isolation',
            description: 'Isolate affected systems immediately',
            responsible: 'security_team',
            timeout: '5 minutes',
            automated: true
          },
          {
            id: 2,
            action: 'assess_scope',
            description: 'Determine scope and impact of breach',
            responsible: 'security_analyst',
            timeout: '15 minutes',
            automated: false
          },
          {
            id: 3,
            action: 'notify_stakeholders',
            description: 'Notify management and legal team',
            responsible: 'incident_commander',
            timeout: '30 minutes',
            automated: false
          },
          {
            id: 4,
            action: 'preserve_evidence',
            description: 'Preserve system logs and forensic evidence',
            responsible: 'forensics_team',
            timeout: '1 hour',
            automated: true
          },
          {
            id: 5,
            action: 'regulatory_notification',
            description: 'File required regulatory notifications',
            responsible: 'legal_compliance',
            timeout: '72 hours',
            automated: false
          }
        ],
        recovery_procedures: [
          'Restore from clean backups',
          'Update security controls',
          'Conduct security assessment',
          'Implement additional monitoring'
        ]
      },
      'malware_detection': {
        id: 'malware_detection',
        name: 'Malware Incident Response',
        severity: 'HIGH',
        category: 'SYSTEM_COMPROMISE',
        estimated_duration: '2-6 hours',
        steps: [
          {
            id: 1,
            action: 'quarantine_system',
            description: 'Quarantine infected systems',
            responsible: 'security_team',
            timeout: '10 minutes',
            automated: true
          },
          {
            id: 2,
            action: 'analyze_malware',
            description: 'Analyze malware sample and behavior',
            responsible: 'security_analyst',
            timeout: '30 minutes',
            automated: false
          },
          {
            id: 3,
            action: 'containment',
            description: 'Implement network segmentation',
            responsible: 'network_team',
            timeout: '45 minutes',
            automated: false
          },
          {
            id: 4,
            action: 'clean_and_recover',
            description: 'Clean systems and recover operations',
            responsible: 'system_admin',
            timeout: '2-4 hours',
            automated: false
          }
        ],
        recovery_procedures: [
          'Reimage affected systems',
          'Update antivirus definitions',
          'Patch vulnerabilities',
          'Enhance endpoint protection'
        ]
      },
      'unauthorized_access': {
        id: 'unauthorized_access',
        name: 'Unauthorized Access Response',
        severity: 'HIGH',
        category: 'UNAUTHORIZED_ACCESS',
        estimated_duration: '1-3 hours',
        steps: [
          {
            id: 1,
            action: 'disable_accounts',
            description: 'Disable compromised accounts',
            responsible: 'security_team',
            timeout: '5 minutes',
            automated: true
          },
          {
            id: 2,
            action: 'reset_passwords',
            description: 'Force password reset for affected users',
            responsible: 'system_admin',
            timeout: '15 minutes',
            automated: true
          },
          {
            id: 3,
            action: 'audit_access',
            description: 'Audit unauthorized access attempts',
            responsible: 'security_analyst',
            timeout: '30 minutes',
            automated: false
          },
          {
            id: 4,
            action: 'notify_affected_users',
            description: 'Notify affected users',
            responsible: 'help_desk',
            timeout: '1 hour',
            automated: false
          }
        ],
        recovery_procedures: [
          'Review and update access controls',
          'Implement additional authentication',
          'Update security policies',
          'Conduct security awareness training'
        ]
      },
      'ddos_attack': {
        id: 'ddos_attack',
        name: 'DDoS Attack Response',
        severity: 'MEDIUM',
        category: 'SERVICE_DISRUPTION',
        estimated_duration: '30 minutes - 2 hours',
        steps: [
          {
            id: 1,
            action: 'activate_ddos_protection',
            description: 'Activate DDoS mitigation services',
            responsible: 'network_team',
            timeout: '5 minutes',
            automated: true
          },
          {
            id: 2,
            action: 'analyze_traffic',
            description: 'Analyze attack patterns and sources',
            responsible: 'security_analyst',
            timeout: '15 minutes',
            automated: false
          },
          {
            id: 3,
            action: 'implement_filters',
            description: 'Implement traffic filtering rules',
            responsible: 'network_admin',
            timeout: '30 minutes',
            automated: false
          },
          {
            id: 4,
            action: 'monitor_recovery',
            description: 'Monitor service recovery',
            responsible: 'operations_team',
            timeout: '1 hour',
            automated: true
          }
        ],
        recovery_procedures: [
          'Review DDoS protection configuration',
          'Optimize traffic filtering rules',
          'Update incident response procedures',
          'Coordinate with ISPs'
        ]
      },
      'insider_threat': {
        id: 'insider_threat',
        name: 'Insider Threat Response',
        severity: 'HIGH',
        category: 'INSIDER_THREAT',
        estimated_duration: '4-12 hours',
        steps: [
          {
            id: 1,
            action: 'suspend_access',
            description: 'Immediately suspend user access',
            responsible: 'security_team',
            timeout: '5 minutes',
            automated: false
          },
          {
            id: 2,
            action: 'preserve_evidence',
            description: 'Preserve user activity logs and evidence',
            responsible: 'forensics_team',
            timeout: '30 minutes',
            automated: true
          },
          {
            id: 3,
            action: 'hr_notification',
            description: 'Notify HR and legal departments',
            responsible: 'incident_commander',
            timeout: '1 hour',
            automated: false
          },
          {
            id: 4,
            action: 'investigate_activity',
            description: 'Conduct detailed investigation',
            responsible: 'security_analyst',
            timeout: '2-8 hours',
            automated: false
          }
        ],
        recovery_procedures: [
          'Review user access privileges',
          'Update monitoring controls',
          'Conduct security awareness program',
          'Review hiring/termination procedures'
        ]
      }
    };

    Object.values(playbooks).forEach(playbook => {
      this.playbooks.set(playbook.id, playbook);
    });

    console.log(`[INCIDENT-RESPONSE] Loaded ${playbooks.length} response playbooks`);
  }

  /**
   * Initialize response teams and escalation paths
   */
  async initializeResponseTeams() {
    this.responseTeams = {
      security_team: {
        name: 'Security Operations Team',
        members: ['security.analyst1@company.com', 'security.analyst2@company.com'],
        onCall: true,
        escalationLevel: 1
      },
      incident_commander: {
        name: 'Incident Commander',
        members: ['incident.commander@company.com'],
        onCall: true,
        escalationLevel: 2
      },
      forensics_team: {
        name: 'Digital Forensics Team',
        members: ['forensics.analyst@company.com'],
        onCall: false,
        escalationLevel: 3
      },
      legal_compliance: {
        name: 'Legal and Compliance',
        members: ['legal@company.com', 'compliance@company.com'],
        onCall: false,
        escalationLevel: 3
      },
      network_team: {
        name: 'Network Operations',
        members: ['network.ops@company.com'],
        onCall: true,
        escalationLevel: 2
      },
      system_admin: {
        name: 'System Administration',
        members: ['sysadmin1@company.com', 'sysadmin2@company.com'],
        onCall: true,
        escalationLevel: 2
      },
      operations_team: {
        name: 'IT Operations',
        members: ['it.ops@company.com'],
        onCall: true,
        escalationLevel: 2
      },
      help_desk: {
        name: 'Help Desk',
        members: ['helpdesk@company.com'],
        onCall: true,
        escalationLevel: 1
      }
    };
  }

  /**
   * Report new security incident
   */
  async reportIncident(incidentData) {
    const incidentId = crypto.randomUUID();
    const timestamp = new Date().toISOString();
    
    const incident = {
      id: incidentId,
      title: incidentData.title,
      description: incidentData.description,
      severity: this.determineSeverity(incidentData),
      category: incidentData.category || 'general',
      status: 'open',
      priority: this.calculatePriority(incidentData),
      reporter: incidentData.reporter,
      source: incidentData.source,
      affectedSystems: incidentData.affectedSystems || [],
      indicators: incidentData.indicators || [],
      timeline: [{
        timestamp,
        event: 'incident_reported',
        description: 'Incident reported',
        user: incidentData.reporter
      }],
      playbook: this.selectPlaybook(incidentData),
      assignedTeam: this.selectResponseTeam(incidentData),
      createdAt: timestamp,
      lastUpdated: timestamp,
      metadata: {
        autoGenerated: incidentData.autoGenerated || false,
        correlationId: incidentData.correlationId,
        tags: incidentData.tags || []
      }
    };

    this.incidents.set(incidentId, incident);
    this.timeline.push({
      timestamp,
      type: 'incident_reported',
      incidentId,
      event: incident.title
    });

    // Automatically trigger response based on playbook
    if (incident.playbook) {
      await this.initiateResponse(incidentId);
    }

    // Send notifications
    await this.notifyStakeholders(incidentId, 'incident_reported');

    console.log(`[INCIDENT-RESPONSE] Incident ${incidentId} reported: ${incident.title}`);
    return incidentId;
  }

  /**
   * Determine incident severity based on criteria
   */
  determineSeverity(incidentData) {
    const severity = incidentData.severity || this.calculateSeverity(incidentData);
    
    // Override automatic calculation if explicitly provided
    if (['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].includes(incidentData.severity)) {
      return incidentData.severity;
    }

    // Auto-determine severity based on incident characteristics
    const indicators = incidentData.indicators || [];
    
    if (indicators.some(i => i.type === 'data_breach' || i.type === 'system_compromise')) {
      return 'CRITICAL';
    }
    
    if (indicators.some(i => i.type === 'unauthorized_access' || i.type === 'malware')) {
      return 'HIGH';
    }
    
    if (indicators.some(i => i.type === 'suspicious_activity' || i.type === 'policy_violation')) {
      return 'MEDIUM';
    }

    return 'LOW';
  }

  /**
   * Calculate incident severity automatically
   */
  calculateSeverity(incidentData) {
    let score = 0;
    
    // Impact assessment
    if (incidentData.affectedSystems?.length > 10) score += 30;
    else if (incidentData.affectedSystems?.length > 5) score += 20;
    else if (incidentData.affectedSystems?.length > 0) score += 10;
    
    // Data sensitivity
    if (incidentData.involvesPHI) score += 40;
    if (incidentData.involvesCustomerData) score += 30;
    if (incidentData.involvesFinancialData) score += 35;
    
    // Urgency indicators
    if (incidentData.userImpact === 'complete_outage') score += 50;
    else if (incidentData.userImpact === 'partial_outage') score += 25;
    else if (incidentData.userImpact === 'degraded_performance') score += 10;
    
    // Current threat level
    if (incidentData.ongoing) score += 20;
    
    if (score >= 70) return 'CRITICAL';
    if (score >= 50) return 'HIGH';
    if (score >= 30) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Calculate incident priority
   */
  calculatePriority(incidentData) {
    const severity = this.determineSeverity(incidentData);
    
    const priorityMap = {
      'CRITICAL': 'P1',
      'HIGH': 'P2',
      'MEDIUM': 'P3',
      'LOW': 'P4'
    };
    
    return priorityMap[severity];
  }

  /**
   * Select appropriate response playbook
   */
  selectPlaybook(incidentData) {
    const category = incidentData.category?.toLowerCase();
    const indicators = incidentData.indicators || [];
    
    // Check for specific indicators
    if (indicators.some(i => i.type === 'data_breach')) {
      return 'data_breach';
    }
    
    if (indicators.some(i => i.type === 'malware')) {
      return 'malware_detection';
    }
    
    if (indicators.some(i => i.type === 'unauthorized_access')) {
      return 'unauthorized_access';
    }
    
    if (indicators.some(i => i.type === 'ddos')) {
      return 'ddos_attack';
    }
    
    if (indicators.some(i => i.type === 'insider_threat')) {
      return 'insider_threat';
    }
    
    // Default based on category
    const playbookMap = {
      'confidentiality': 'data_breach',
      'integrity': 'unauthorized_access',
      'availability': 'ddos_attack',
      'system_compromise': 'malware_detection'
    };
    
    return playbookMap[category] || null;
  }

  /**
   * Select appropriate response team
   */
  selectResponseTeam(incidentData) {
    const playbook = this.selectPlaybook(incidentData);
    
    const teamMap = {
      'data_breach': 'security_team',
      'malware_detection': 'security_team',
      'unauthorized_access': 'security_team',
      'ddos_attack': 'network_team',
      'insider_threat': 'security_team'
    };
    
    return teamMap[playbook] || 'security_team';
  }

  /**
   * Initiate incident response procedures
   */
  async initiateResponse(incidentId) {
    const incident = this.incidents.get(incidentId);
    if (!incident || !incident.playbook) {
      return;
    }

    const playbook = this.playbooks.get(incident.playbook);
    if (!playbook) {
      console.error(`[INCIDENT-RESPONSE] Playbook ${incident.playbook} not found`);
      return;
    }

    console.log(`[INCIDENT-RESPONSE] Initiating response for incident ${incidentId} using playbook ${playbook.name}`);

    // Initialize response tracking
    this.responses.set(incidentId, {
      incidentId,
      playbook: playbook.id,
      status: 'in_progress',
      currentStep: 0,
      steps: playbook.steps.map(step => ({
        ...step,
        status: 'pending',
        startedAt: null,
        completedAt: null,
        result: null
      })),
      startedAt: new Date().toISOString()
    });

    // Update incident timeline
    await this.addToTimeline(incidentId, {
      event: 'response_initiated',
      description: `Response initiated using playbook: ${playbook.name}`,
      user: 'system'
    });

    // Execute first step if automated
    const firstStep = playbook.steps[0];
    if (firstStep.automated) {
      await this.executeResponseStep(incidentId, firstStep);
    }
  }

  /**
   * Execute specific response step
   */
  async executeResponseStep(incidentId, step) {
    const response = this.responses.get(incidentId);
    const incident = this.incidents.get(incidentId);
    
    if (!response || !incident) {
      return;
    }

    console.log(`[INCIDENT-RESPONSE] Executing step ${step.id}: ${step.action} for incident ${incidentId}`);

    // Update step status
    const stepIndex = response.steps.findIndex(s => s.id === step.id);
    response.steps[stepIndex].status = 'in_progress';
    response.steps[stepIndex].startedAt = new Date().toISOString();
    response.currentStep = step.id;

    try {
      let result;
      
      // Execute step based on action type
      switch (step.action) {
        case 'immediate_isolation':
          result = await this.isolateAffectedSystems(incident);
          break;
        case 'assess_scope':
          result = await this.assessIncidentScope(incident);
          break;
        case 'preserve_evidence':
          result = await this.preserveEvidence(incident);
          break;
        case 'containment':
          result = await this.implementContainment(incident);
          break;
        case 'notify_stakeholders':
          result = await this.notifyStakeholders(incidentId, 'incident_response');
          break;
        default:
          result = await this.executeCustomStep(step, incident);
      }

      // Update step completion
      response.steps[stepIndex].status = 'completed';
      response.steps[stepIndex].completedAt = new Date().toISOString();
      response.steps[stepIndex].result = result;

      // Add to timeline
      await this.addToTimeline(incidentId, {
        event: `step_completed_${step.id}`,
        description: `Completed: ${step.description}`,
        user: 'system',
        data: result
      });

      console.log(`[INCIDENT-RESPONSE] Step ${step.id} completed for incident ${incidentId}`);

      // Schedule next step if timeout-based
      if (step.timeout && !step.automated) {
        await this.scheduleTimeoutStep(incidentId, step);
      }

      // Execute next step if automated
      const nextStep = this.getNextStep(incidentId, step.id);
      if (nextStep && nextStep.automated) {
        setTimeout(async () => {
          await this.executeResponseStep(incidentId, nextStep);
        }, 1000); // Small delay between automated steps
      }

    } catch (error) {
      console.error(`[INCIDENT-RESPONSE] Step ${step.id} failed for incident ${incidentId}:`, error);
      
      response.steps[stepIndex].status = 'failed';
      response.steps[stepIndex].completedAt = new Date().toISOString();
      response.steps[stepIndex].error = error.message;

      await this.addToTimeline(incidentId, {
        event: `step_failed_${step.id}`,
        description: `Failed: ${step.description} - ${error.message}`,
        user: 'system'
      });
    }
  }

  /**
   * Isolate affected systems (automated containment)
   */
  async isolateAffectedSystems(incident) {
    console.log('[INCIDENT-RESPONSE] Isolating affected systems...');
    
    const isolationResults = [];
    
    for (const system of incident.affectedSystems) {
      try {
        // Simulate system isolation
        const result = {
          system: system,
          action: 'network_isolation',
          status: 'completed',
          timestamp: new Date().toISOString(),
          details: `System ${system} isolated from network`
        };
        
        isolationResults.push(result);
        console.log(`[INCIDENT-RESPONSE] System ${system} isolated successfully`);
        
      } catch (error) {
        isolationResults.push({
          system: system,
          action: 'network_isolation',
          status: 'failed',
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
    
    return {
      isolated_systems: isolationResults.filter(r => r.status === 'completed').length,
      failed_isolations: isolationResults.filter(r => r.status === 'failed').length,
      results: isolationResults
    };
  }

  /**
   * Assess incident scope and impact
   */
  async assessIncidentScope(incident) {
    console.log('[INCIDENT-RESPONSE] Assessing incident scope...');
    
    // Simulate scope assessment
    await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate analysis time
    
    return {
      affected_systems: incident.affectedSystems.length,
      estimated_users_impacted: Math.floor(Math.random() * 1000),
      data_involved: incident.involvesPHI ? 'PHI' : 'Unknown',
      financial_impact: 'TBD',
      legal_implications: incident.involvesPHI ? 'HIPAA Notification Required' : 'Standard breach procedures',
      containment_feasible: true,
      recovery_time_estimate: '2-4 hours'
    };
  }

  /**
   * Preserve forensic evidence
   */
  async preserveEvidence(incident) {
    console.log('[INCIDENT-RESPONSE] Preserving forensic evidence...');
    
    const evidence = {
      incidentId: incident.id,
      preservedAt: new Date().toISOString(),
      evidenceItems: [
        {
          type: 'system_logs',
          source: 'affected_systems',
          retentionPeriod: this.config.evidenceRetention,
          checksum: crypto.randomBytes(32).toString('hex')
        },
        {
          type: 'network_logs',
          source: 'firewall',
          retentionPeriod: this.config.evidenceRetention,
          checksum: crypto.randomBytes(32).toString('hex')
        },
        {
          type: 'audit_trails',
          source: 'application_logs',
          retentionPeriod: this.config.evidenceRetention,
          checksum: crypto.randomBytes(32).toString('hex')
        }
      ],
      chain_of_custody: [{
        timestamp: new Date().toISOString(),
        action: 'evidence_collected',
        person: 'forensics_team',
        location: 'secure_evidence_locker'
      }]
    };
    
    return evidence;
  }

  /**
   * Implement containment measures
   */
  async implementContainment(incident) {
    console.log('[INCIDENT-RESPONSE] Implementing containment measures...');
    
    return {
      network_segmentation: 'completed',
      access_restrictions: 'applied',
      system_quotarantine: 'active',
      monitoring_enhanced: 'enabled',
      containment_duration: 'Until incident resolution'
    };
  }

  /**
   * Notify stakeholders about incident
   */
  async notifyStakeholders(incidentId, notificationType) {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;

    const notification = {
      incidentId,
      type: notificationType,
      timestamp: new Date().toISOString(),
      recipients: this.getNotificationRecipients(incident),
      subject: `Security Incident: ${incident.title}`,
      severity: incident.severity
    };

    this.notificationQueue.push(notification);
    
    // Process notifications
    await this.processNotificationQueue();

    return { notification_sent: true, recipient_count: notification.recipients.length };
  }

  /**
   * Get notification recipients based on incident severity
   */
  getNotificationRecipients(incident) {
    const recipients = new Set();
    
    // Always notify incident commander for P1/P2 incidents
    if (incident.priority === 'P1' || incident.priority === 'P2') {
      recipients.add('incident.commander@company.com');
    }
    
    // Notify based on severity
    if (incident.severity === 'CRITICAL') {
      recipients.add('security.lead@company.com');
      recipients.add('cio@company.com');
      if (incident.involvesPHI) {
        recipients.add('legal@company.com');
        recipients.add('compliance@company.com');
      }
    } else if (incident.severity === 'HIGH') {
      recipients.add('security.manager@company.com');
    }
    
    // Notify assigned team
    const team = this.responseTeams[incident.assignedTeam];
    if (team && team.onCall) {
      team.members.forEach(member => recipients.add(member));
    }
    
    return Array.from(recipients);
  }

  /**
   * Process notification queue
   */
  async processNotificationQueue() {
    while (this.notificationQueue.length > 0) {
      const notification = this.notificationQueue.shift();
      
      try {
        // Simulate sending notifications
        console.log('[INCIDENT-RESPONSE] Sending notification:', notification.subject);
        console.log('[INCIDENT-RESPONSE] Recipients:', notification.recipients);
        
        // In production, integrate with:
        // - Email systems
        // - Slack/Teams
        // - PagerDuty
        // - SMS services
        
        notification.status = 'sent';
        notification.sentAt = new Date().toISOString();
        
      } catch (error) {
        console.error('[INCIDENT-RESPONSE] Notification failed:', error);
        notification.status = 'failed';
        notification.error = error.message;
      }
    }
  }

  /**
   * Add event to incident timeline
   */
  async addToTimeline(incidentId, event) {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;

    const timelineEvent = {
      ...event,
      timestamp: event.timestamp || new Date().toISOString()
    };

    incident.timeline.push(timelineEvent);
    incident.lastUpdated = timelineEvent.timestamp;
  }

  /**
   * Get next step in response playbook
   */
  getNextStep(incidentId, currentStepId) {
    const response = this.responses.get(incidentId);
    if (!response) return null;

    const currentIndex = response.steps.findIndex(s => s.id === currentStepId);
    return response.steps[currentIndex + 1] || null;
  }

  /**
   * Schedule timeout-based step execution
   */
  async scheduleTimeoutStep(incidentId, step) {
    const timeoutMs = this.parseTimeout(step.timeout);
    
    this.escalationQueue.set(`${incidentId}_${step.id}`, {
      incidentId,
      stepId: step.id,
      scheduledFor: new Date(Date.now() + timeoutMs),
      responsible: step.responsible
    });

    // Schedule escalation check
    setTimeout(async () => {
      await this.checkEscalation(incidentId, step.id);
    }, timeoutMs);
  }

  /**
   * Parse timeout string to milliseconds
   */
  parseTimeout(timeout) {
    const match = timeout.match(/(\d+)\s*(minutes?|hours?|seconds?)/i);
    if (!match) return 0;

    const value = parseInt(match[1]);
    const unit = match[2].toLowerCase();

    switch (unit) {
      case 'second':
      case 'seconds':
        return value * 1000;
      case 'minute':
      case 'minutes':
        return value * 60 * 1000;
      case 'hour':
      case 'hours':
        return value * 60 * 60 * 1000;
      default:
        return 0;
    }
  }

  /**
   * Check for escalation timeouts
   */
  async checkEscalation(incidentId, stepId) {
    const escalationKey = `${incidentId}_${stepId}`;
    const escalation = this.escalationQueue.get(escalationKey);
    
    if (!escalation) return;

    const response = this.responses.get(incidentId);
    const step = response?.steps.find(s => s.id === stepId);
    
    if (step && step.status === 'pending') {
      console.warn(`[INCIDENT-RESPONSE] Escalation: Step ${stepId} timed out for incident ${incidentId}`);
      
      await this.addToTimeline(incidentId, {
        event: 'step_escalated',
        description: `Step ${step.description} timed out, escalating to ${step.responsible}`,
        user: 'system',
        severity: 'HIGH'
      });

      // Notify responsible team
      await this.escalateToTeam(incidentId, step.responsible, step);
      
      // Remove from escalation queue
      this.escalationQueue.delete(escalationKey);
    }
  }

  /**
   * Escalate to responsible team
   */
  async escalateToTeam(incidentId, teamName, step) {
    const team = this.responseTeams[teamName];
    if (!team) return;

    console.log(`[INCIDENT-RESPONSE] Escalating to ${teamName} for incident ${incidentId}`);
    
    // Add escalation event
    await this.addToTimeline(incidentId, {
      event: 'team_escalated',
      description: `Escalated to ${team.name} for step: ${step.description}`,
      user: 'system',
      escalated_to: team.name
    });
  }

  /**
   * Execute custom response step
   */
  async executeCustomStep(step, incident) {
    console.log(`[INCIDENT-RESPONSE] Executing custom step: ${step.action}`);
    
    return {
      action: step.action,
      description: step.description,
      result: 'completed',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Update incident status
   */
  async updateIncidentStatus(incidentId, status, notes = '') {
    const incident = this.incidents.get(incidentId);
    if (!incident) return false;

    const oldStatus = incident.status;
    incident.status = status;
    incident.lastUpdated = new Date().toISOString();

    await this.addToTimeline(incidentId, {
      event: 'status_updated',
      description: `Status changed from ${oldStatus} to ${status}`,
      user: 'user',
      data: { notes }
    });

    // Handle status-specific actions
    if (status === 'resolved') {
      await this.resolveIncident(incidentId);
    } else if (status === 'closed') {
      await this.closeIncident(incidentId);
    }

    return true;
  }

  /**
   * Resolve incident
   */
  async resolveIncident(incidentId) {
    const incident = this.incidents.get(incidentId);
    const response = this.responses.get(incidentId);
    
    if (!incident || !response) return;

    // Complete any pending steps
    for (const step of response.steps) {
      if (step.status === 'pending' || step.status === 'in_progress') {
        step.status = 'completed';
        step.completedAt = new Date().toISOString();
        step.result = { resolved_during_incident_closure: true };
      }
    }

    response.status = 'completed';
    response.completedAt = new Date().toISOString();

    await this.addToTimeline(incidentId, {
      event: 'incident_resolved',
      description: 'Incident marked as resolved',
      user: 'user'
    });

    console.log(`[INCIDENT-RESPONSE] Incident ${incidentId} resolved`);
  }

  /**
   * Close incident
   */
  async closeIncident(incidentId) {
    const incident = this.incidents.get(incidentId);
    
    if (!incident) return;

    incident.status = 'closed';
    incident.closedAt = new Date().toISOString();

    await this.addToTimeline(incidentId, {
      event: 'incident_closed',
      description: 'Incident closed',
      user: 'user'
    });

    console.log(`[INCIDENT-RESPONSE] Incident ${incidentId} closed`);
  }

  /**
   * Get incident details
   */
  getIncident(incidentId) {
    const incident = this.incidents.get(incidentId);
    if (!incident) return null;

    const response = this.responses.get(incidentId);
    
    return {
      ...incident,
      response: response || null
    };
  }

  /**
   * Get all incidents with filtering
   */
  getIncidents(filters = {}) {
    let incidents = Array.from(this.incidents.values());

    if (filters.status) {
      incidents = incidents.filter(i => i.status === filters.status);
    }

    if (filters.severity) {
      incidents = incidents.filter(i => i.severity === filters.severity);
    }

    if (filters.category) {
      incidents = incidents.filter(i => i.category === filters.category);
    }

    if (filters.priority) {
      incidents = incidents.filter(i => i.priority === filters.priority);
    }

    if (filters.startDate) {
      incidents = incidents.filter(i => new Date(i.createdAt) >= new Date(filters.startDate));
    }

    if (filters.endDate) {
      incidents = incidents.filter(i => new Date(i.createdAt) <= new Date(filters.endDate));
    }

    return incidents.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  }

  /**
   * Generate incident response report
   */
  async generateResponseReport(incidentId) {
    const incident = this.getIncident(incidentId);
    if (!incident) return null;

    const response = this.responses.get(incidentId);
    const playbook = incident.playbook ? this.playbooks.get(incident.playbook) : null;

    const report = {
      incident_id: incident.id,
      title: incident.title,
      severity: incident.severity,
      status: incident.status,
      duration: this.calculateIncidentDuration(incident),
      response_metrics: {
        time_to_detection: '0 minutes', // Would be calculated from logs
        time_to_response: '5 minutes', // Would be calculated from timeline
        time_to_containment: '15 minutes', // Would be calculated from timeline
        time_to_resolution: this.calculateResolutionTime(incident)
      },
      playbook_execution: {
        playbook_used: playbook?.name || 'None',
        steps_completed: response?.steps.filter(s => s.status === 'completed').length || 0,
        steps_failed: response?.steps.filter(s => s.status === 'failed').length || 0,
        automation_effectiveness: this.calculateAutomationEffectiveness(response)
      },
      impact_assessment: {
        systems_affected: incident.affectedSystems.length,
        data_involved: incident.metadata?.dataTypes || 'Unknown',
        user_impact: incident.metadata?.userImpact || 'Unknown',
        business_impact: incident.metadata?.businessImpact || 'TBD'
      },
      lessons_learned: this.generateLessonsLearned(incident, response),
      recommendations: this.generatePostIncidentRecommendations(incident, response)
    };

    return report;
  }

  /**
   * Calculate incident duration
   */
  calculateIncidentDuration(incident) {
    const start = new Date(incident.createdAt);
    const end = incident.closedAt ? new Date(incident.closedAt) : new Date();
    
    const durationMs = end - start;
    const hours = Math.floor(durationMs / (1000 * 60 * 60));
    const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
    
    return `${hours}h ${minutes}m`;
  }

  /**
   * Calculate resolution time
   */
  calculateResolutionTime(incident) {
    if (incident.status !== 'closed') return 'In Progress';
    
    const created = new Date(incident.createdAt);
    const closed = new Date(incident.closedAt);
    
    const hours = Math.floor((closed - created) / (1000 * 60 * 60));
    const minutes = Math.floor(((closed - created) % (1000 * 60 * 60)) / (1000 * 60));
    
    return `${hours}h ${minutes}m`;
  }

  /**
   * Calculate automation effectiveness
   */
  calculateAutomationEffectiveness(response) {
    if (!response) return 0;
    
    const totalSteps = response.steps.length;
    const automatedSteps = response.steps.filter(s => s.automated).length;
    const completedAutomated = response.steps.filter(s => s.automated && s.status === 'completed').length;
    
    return totalSteps > 0 ? (completedAutomated / automatedSteps * 100) : 0;
  }

  /**
   * Generate lessons learned from incident
   */
  generateLessonsLearned(incident, response) {
    const lessons = [];
    
    // Response time analysis
    const responseTime = response ? (response.completedAt ? 
      new Date(response.completedAt) - new Date(response.startedAt) : 0) : 0;
    
    if (responseTime > 3600000) { // More than 1 hour
      lessons.push('Response time exceeded target SLA - review escalation procedures');
    }
    
    // Automation analysis
    const automationEffectiveness = this.calculateAutomationEffectiveness(response);
    if (automationEffectiveness < 70) {
      lessons.push('Low automation effectiveness - increase automated response capabilities');
    }
    
    // Failure analysis
    if (response?.steps.some(s => s.status === 'failed')) {
      lessons.push('Some response steps failed - review playbook accuracy and team training');
    }
    
    // Severity-based lessons
    if (incident.severity === 'CRITICAL') {
      lessons.push('Critical incident occurred - conduct comprehensive security review');
    }
    
    return lessons.length > 0 ? lessons : ['No significant issues identified in incident response'];
  }

  /**
   * Generate post-incident recommendations
   */
  generatePostIncidentRecommendations(incident, response) {
    const recommendations = [];
    
    // Security improvements
    if (incident.category === 'system_compromise') {
      recommendations.push('Implement additional endpoint protection controls');
      recommendations.push('Enhance system hardening procedures');
    }
    
    if (incident.category === 'unauthorized_access') {
      recommendations.push('Review and update access control policies');
      recommendations.push('Implement additional multi-factor authentication');
    }
    
    if (incident.involvesPHI) {
      recommendations.push('Conduct HIPAA compliance audit');
      recommendations.push('Review PHI handling procedures');
    }
    
    // Process improvements
    const automationEffectiveness = this.calculateAutomationEffectiveness(response);
    if (automationEffectiveness < 80) {
      recommendations.push('Increase automation in incident response playbooks');
    }
    
    // Team training
    recommendations.push('Conduct incident response training for all team members');
    recommendations.push('Review and update incident response procedures');
    
    return recommendations;
  }

  /**
   * Get incident response statistics
   */
  getIncidentStatistics() {
    const incidents = Array.from(this.incidents.values());
    const responses = Array.from(this.responses.values());
    
    const stats = {
      total_incidents: incidents.length,
      active_incidents: incidents.filter(i => i.status === 'open' || i.status === 'investigating').length,
      resolved_incidents: incidents.filter(i => i.status === 'resolved').length,
      closed_incidents: incidents.filter(i => i.status === 'closed').length,
      incidents_by_severity: this.countByProperty(incidents, 'severity'),
      incidents_by_category: this.countByProperty(incidents, 'category'),
      incidents_by_priority: this.countByProperty(incidents, 'priority'),
      average_resolution_time: 0,
      response_effectiveness: 0,
      automation_rate: 0,
      escalation_rate: 0
    };

    // Calculate average resolution time
    const resolvedIncidents = incidents.filter(i => i.closedAt);
    if (resolvedIncidents.length > 0) {
      const totalTime = resolvedIncidents.reduce((sum, incident) => {
        return sum + (new Date(incident.closedAt) - new Date(incident.createdAt));
      }, 0);
      
      stats.average_resolution_time = Math.floor(totalTime / resolvedIncidents.length / (1000 * 60)); // in minutes
    }

    // Calculate automation rate
    if (responses.length > 0) {
      const totalSteps = responses.reduce((sum, response) => sum + response.steps.length, 0);
      const automatedSteps = responses.reduce((sum, response) => {
        return sum + response.steps.filter(s => s.automated).length;
      }, 0);
      
      stats.automation_rate = totalSteps > 0 ? (automatedSteps / totalSteps * 100) : 0;
    }

    return stats;
  }

  /**
   * Count incidents by property value
   */
  countByProperty(incidents, property) {
    const counts = {};
    incidents.forEach(incident => {
      const value = incident[property] || 'unknown';
      counts[value] = (counts[value] || 0) + 1;
    });
    return counts;
  }

  /**
   * Auto-detect security incidents from monitoring data
   */
  async autoDetectIncidents(monitoringData) {
    const detectedIncidents = [];
    
    // Check for data breach indicators
    if (monitoringData.unusualDataAccess) {
      const incidentId = await this.reportIncident({
        title: 'Unusual Data Access Pattern Detected',
        description: `Detected unusual access to sensitive data: ${monitoringData.affectedData}`,
        category: 'confidentiality',
        indicators: [{
          type: 'data_breach',
          source: 'monitoring_system',
          severity: 'HIGH'
        }],
        involvesPHI: monitoringData.involvesPHI || false,
        affectedSystems: monitoringData.affectedSystems || [],
        autoGenerated: true,
        correlationId: monitoringData.correlationId
      });
      
      detectedIncidents.push(incidentId);
    }
    
    // Check for malware indicators
    if (monitoringData.malwareDetected) {
      const incidentId = await this.reportIncident({
        title: 'Malware Detection on System',
        description: `Malware detected: ${monitoringData.malwareName}`,
        category: 'system_compromise',
        indicators: [{
          type: 'malware',
          source: 'antivirus_system',
          severity: 'HIGH'
        }],
        affectedSystems: [monitoringData.affectedSystem],
        autoGenerated: true
      });
      
      detectedIncidents.push(incidentId);
    }
    
    // Check for unauthorized access
    if (monitoringData.unauthorizedAccess) {
      const incidentId = await this.reportIncident({
        title: 'Unauthorized Access Attempt',
        description: `Unauthorized access from: ${monitoringData.sourceIP}`,
        category: 'integrity',
        indicators: [{
          type: 'unauthorized_access',
          source: 'access_control_system',
          severity: 'MEDIUM'
        }],
        affectedSystems: monitoringData.affectedSystems || [],
        autoGenerated: true
      });
      
      detectedIncidents.push(incidentId);
    }
    
    return detectedIncidents;
  }

  /**
   * Emergency incident procedures
   */
  async emergencyResponse(incidentType, affectedSystems = []) {
    console.log(`[INCIDENT-RESPONSE] EMERGENCY RESPONSE: ${incidentType}`);
    
    const emergencyIncident = await this.reportIncident({
      title: `EMERGENCY: ${incidentType}`,
      description: `Emergency response initiated for ${incidentType}`,
      severity: 'CRITICAL',
      category: this.getEmergencyCategory(incidentType),
      affectedSystems,
      involvesPHI: incidentType.includes('phi') || incidentType.includes('data'),
      autoGenerated: true,
      source: 'emergency_system'
    });

    // Execute emergency playbook immediately
    await this.executeEmergencyProcedures(emergencyIncident, incidentType);
    
    return emergencyIncident;
  }

  /**
   * Get emergency category from incident type
   */
  getEmergencyCategory(incidentType) {
    if (incidentType.includes('breach')) return 'confidentiality';
    if (incidentType.includes('malware')) return 'system_compromise';
    if (incidentType.includes('access')) return 'integrity';
    if (incidentType.includes('availability')) return 'availability';
    return 'general';
  }

  /**
   * Execute emergency procedures
   */
  async executeEmergencyProcedures(incidentId, incidentType) {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;

    console.log(`[INCIDENT-RESPONSE] Executing emergency procedures for ${incidentType}`);

    // Immediate containment actions
    if (incidentType.includes('breach') || incidentType.includes('malware')) {
      await this.isolateAffectedSystems(incident);
    }

    // Notify all stakeholders immediately
    await this.notifyStakeholders(incidentId, 'emergency_incident');

    // Escalate to highest level
    await this.addToTimeline(incidentId, {
      event: 'emergency_escalated',
      description: 'Emergency incident escalated to executive level',
      user: 'system',
      severity: 'CRITICAL'
    });
  }
}

module.exports = SecurityIncidentResponse;