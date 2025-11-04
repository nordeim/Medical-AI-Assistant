// Advanced Anomaly Detection System for Healthcare User Management
// Machine learning-based anomaly detection for security and compliance

const crypto = require('crypto');
const EventEmitter = require('events');

class AnomalyDetector extends EventEmitter {
  constructor() {
    super();
    this.anomalyModels = new Map();
    this.baselineProfiles = new Map();
    this.trainingData = new Map();
    this.detectionThresholds = this.initializeThresholds();
    this.mlEngine = null;
    this.models = {
      isolationForest: null,
      statisticalModel: null,
      timeSeriesModel: null,
      behavioralModel: null
    };
    
    this.initializeModels();
  }

  // Initialize ML Models
  initializeModels() {
    // Initialize isolation forest for outlier detection
    this.models.isolationForest = {
      type: 'isolation_forest',
      trees: 100,
      contamination: 0.1,
      maxSamples: 'auto',
      randomState: 42,
      trained: false
    };

    // Initialize statistical model for data distribution analysis
    this.models.statisticalModel = {
      type: 'statistical',
      distribution: 'mixed_gaussian',
      window_size: 1000,
      confidence_interval: 0.95,
      trained: false
    };

    // Initialize time series model for temporal anomaly detection
    this.models.timeSeriesModel = {
      type: 'time_series',
      algorithm: 'lstm',
      sequence_length: 24,
      prediction_horizon: 1,
      trained: false
    };

    // Initialize behavioral model for user behavior analysis
    this.models.behavioralModel = {
      type: 'behavioral',
      features: [
        'access_time',
        'resource_access',
        'session_duration',
        'actions_frequency',
        'error_rate'
      ],
      window_size: 7, // days
      trained: false
    };
  }

  // Comprehensive Anomaly Analysis
  async analyzeAnomaly(event) {
    try {
      const analysisStartTime = Date.now();
      
      // Multi-dimensional anomaly analysis
      const anomalyScores = await this.calculateMultiDimensionalScores(event);
      
      // Combine scores using ensemble method
      const ensembleScore = this.calculateEnsembleScore(anomalyScores);
      
      // Determine anomaly type and severity
      const anomalyType = this.identifyAnomalyType(event, anomalyScores);
      const severity = this.calculateAnomalySeverity(ensembleScore, anomalyType);
      
      // Generate anomaly explanation
      const explanation = await this.generateAnomalyExplanation(event, anomalyScores, anomalyType);
      
      // Create anomaly record
      const anomaly = {
        anomalyId: crypto.randomUUID(),
        event,
        ensembleScore,
        anomalyScores,
        anomalyType,
        severity,
        explanation,
        timestamp: new Date().toISOString(),
        confidence: this.calculateConfidence(anomalyScores),
        recommendedActions: await this.generateRecommendedActions(anomalyType, severity),
        context: await this.gatherAnomalyContext(event)
      };

      // Store anomaly
      await this.storeAnomaly(anomaly);

      // Trigger response if severe
      if (severity === 'critical' || severity === 'high') {
        this.emit('severeAnomaly', anomaly);
      }

      // Emit general anomaly event
      this.emit('anomalyDetected', anomaly);

      const analysisTime = Date.now() - analysisStartTime;
      console.log(`Anomaly analysis completed in ${analysisTime}ms`);

      return anomaly;

    } catch (error) {
      console.error('Anomaly analysis error:', error);
      throw error;
    }
  }

  // Multi-dimensional Anomaly Detection
  async calculateMultiDimensionalScores(event) {
    const scores = {};

    // 1. Statistical Anomaly Score
    scores.statistical = await this.calculateStatisticalAnomalyScore(event);

    // 2. Behavioral Anomaly Score
    scores.behavioral = await this.calculateBehavioralAnomalyScore(event);

    // 3. Temporal Anomaly Score
    scores.temporal = await this.calculateTemporalAnomalyScore(event);

    // 4. Access Pattern Anomaly Score
    scores.accessPattern = await this.calculateAccessPatternAnomalyScore(event);

    // 5. Security Anomaly Score
    scores.security = await this.calculateSecurityAnomalyScore(event);

    return scores;
  }

  // Statistical Anomaly Detection
  async calculateStatisticalAnomalyScore(event) {
    const userId = event.userId;
    const currentValue = this.extractFeatureValue(event, 'primary_metric');
    
    // Get baseline statistics for user
    const baseline = await this.getUserBaseline(userId, 'statistical');
    
    if (!baseline || baseline.count < 10) {
      return 0.3; // Low confidence for insufficient data
    }

    // Calculate z-score
    const zScore = Math.abs((currentValue - baseline.mean) / baseline.stdDev);
    
    // Calculate probability score
    const probabilityScore = this.normalCDF(zScore);
    
    // Statistical anomaly threshold
    const threshold = 2.5; // 95% confidence interval
    
    return Math.min(zScore / threshold, 1.0);
  }

  // Behavioral Anomaly Detection
  async calculateBehavioralAnomalyScore(event) {
    const userId = event.userId;
    const behaviorFeatures = this.extractBehavioralFeatures(event);
    
    // Get behavioral baseline
    const behavioralBaseline = await this.getUserBaseline(userId, 'behavioral');
    
    if (!behavioralBaseline) {
      return 0.4;
    }

    // Calculate behavioral deviation
    const deviations = [];
    
    for (const [feature, value] of Object.entries(behaviorFeatures)) {
      if (behavioralBaseline[feature]) {
        const deviation = Math.abs(value - behavioralBaseline[feature].mean) / behavioralBaseline[feature].stdDev;
        deviations.push(deviation);
      }
    }

    // Average deviation
    const avgDeviation = deviations.reduce((a, b) => a + b, 0) / deviations.length;
    
    return Math.min(avgDeviation / 2.0, 1.0);
  }

  // Temporal Anomaly Detection
  async calculateTemporalAnomalyScore(event) {
    const timestamp = new Date(event.timestamp);
    const hour = timestamp.getHours();
    const dayOfWeek = timestamp.getDay();
    
    // Get temporal baseline
    const temporalBaseline = await this.getTemporalBaseline(event.userId);
    
    if (!temporalBaseline) {
      return 0.3;
    }

    // Check unusual time patterns
    let timeScore = 0;
    
    // Off-hours access
    if (hour < 6 || hour > 22) {
      const usualOffHours = temporalBaseline.offHours / temporalBaseline.total;
      const currentOffHours = this.isOffHours(timestamp) ? 1 : 0;
      timeScore += Math.abs(currentOffHours - usualOffHours);
    }

    // Weekend access
    const weekend = dayOfWeek === 0 || dayOfWeek === 6;
    const usualWeekend = temporalBaseline.weekend / temporalBaseline.total;
    const currentWeekend = weekend ? 1 : 0;
    timeScore += Math.abs(currentWeekend - usualWeekend);

    // Check frequency patterns
    const currentHourActivity = temporalBaseline.hourlyActivity[hour] || 0;
    const avgHourlyActivity = temporalBaseline.total / 24;
    const frequencyDeviation = Math.abs(currentHourActivity - avgHourlyActivity) / avgHourlyActivity;
    timeScore += Math.min(frequencyDeviation, 1.0);

    return Math.min(timeScore / 3, 1.0);
  }

  // Access Pattern Anomaly Detection
  async calculateAccessPatternAnomalyScore(event) {
    const userId = event.userId;
    const resource = event.details?.resource || event.resource;
    const action = event.event || event.action;
    
    // Get access pattern baseline
    const accessBaseline = await this.getUserBaseline(userId, 'access_patterns');
    
    if (!accessBaseline) {
      return 0.4;
    }

    let patternScore = 0;

    // Check resource access patterns
    const resourceFrequency = accessBaseline.resources[resource] || 0;
    const avgResourceFrequency = Object.values(accessBaseline.resources).reduce((a, b) => a + b, 0) / Object.keys(accessBaseline.resources).length;
    
    if (resourceFrequency === 0 && avgResourceFrequency > 5) {
      patternScore += 0.6; // New resource access
    }

    // Check action patterns
    const actionFrequency = accessBaseline.actions[action] || 0;
    const avgActionFrequency = Object.values(accessBaseline.actions).reduce((a, b) => a + b, 0) / Object.keys(accessBaseline.actions).length;
    
    if (actionFrequency < avgActionFrequency * 0.1 && avgActionFrequency > 10) {
      patternScore += 0.4; // Unusual action frequency
    }

    // Check access velocity
    const recentAccessCount = await this.getRecentAccessCount(userId, 60); // Last 60 minutes
    const usualVelocity = accessBaseline.averageVelocity;
    
    if (recentAccessCount > usualVelocity * 3) {
      patternScore += 0.5; // High velocity access
    }

    return Math.min(patternScore, 1.0);
  }

  // Security Anomaly Detection
  async calculateSecurityAnomalyScore(event) {
    const securityIndicators = {
      failedLogins: 0,
      unusualIP: false,
      privilegeEscalation: false,
      dataExfiltration: false,
      offHoursAccess: false
    };

    // Analyze event for security indicators
    if (event.event?.includes('failed_login')) {
      securityIndicators.failedLogins = 1;
    }

    if (event.details?.ipAddress) {
      const usualIPs = await this.getUserUsualIPs(event.userId);
      securityIndicators.unusualIP = !usualIPs.includes(event.details.ipAddress);
    }

    if (event.event?.includes('role') && event.event?.includes('change')) {
      securityIndicators.privilegeEscalation = true;
    }

    if (event.event?.includes('export') || event.event?.includes('data')) {
      const exportSize = event.details?.dataSize || 0;
      securityIndicators.dataExfiltration = exportSize > 1000000; // > 1MB
    }

    const accessTime = new Date(event.timestamp);
    securityIndicators.offHoursAccess = this.isOffHours(accessTime);

    // Calculate security score
    const weights = {
      failedLogins: 0.3,
      unusualIP: 0.25,
      privilegeEscalation: 0.2,
      dataExfiltration: 0.15,
      offHoursAccess: 0.1
    };

    let securityScore = 0;
    Object.entries(securityIndicators).forEach(([indicator, value]) => {
      const weight = weights[indicator];
      if (value === true) {
        securityScore += weight;
      } else if (value > 0) {
        securityScore += weight * Math.min(value / 5, 1); // Normalize failed logins
      }
    });

    return Math.min(securityScore, 1.0);
  }

  // Ensemble Score Calculation
  calculateEnsembleScore(anomalyScores) {
    const weights = {
      statistical: 0.25,
      behavioral: 0.25,
      temporal: 0.2,
      accessPattern: 0.2,
      security: 0.1
    };

    let weightedScore = 0;
    let totalWeight = 0;

    Object.entries(anomalyScores).forEach(([type, score]) => {
      if (weights[type]) {
        weightedScore += score * weights[type];
        totalWeight += weights[type];
      }
    });

    return totalWeight > 0 ? weightedScore / totalWeight : 0;
  }

  // Anomaly Type Identification
  identifyAnomalyType(event, anomalyScores) {
    const maxScoreType = Object.entries(anomalyScores).reduce((max, [type, score]) => 
      score > max.score ? { type, score } : max, { type: null, score: 0 }
    );

    // Specific type identification based on event characteristics
    if (event.event?.includes('failed_login') && anomalyScores.security > 0.7) {
      return 'authentication_anomaly';
    }

    if (event.event?.includes('export') && anomalyScores.security > 0.6) {
      return 'data_exfiltration_attempt';
    }

    if (event.event?.includes('role') && anomalyScores.accessPattern > 0.8) {
      return 'privilege_escalation';
    }

    if (this.isOffHours(event.timestamp) && anomalyScores.temporal > 0.6) {
      return 'off_hours_access';
    }

    if (anomalyScores.behavioral > 0.7) {
      return 'behavioral_deviation';
    }

    if (maxScoreType.type) {
      return `${maxScoreType.type}_anomaly`;
    }

    return 'unknown_anomaly';
  }

  // Severity Calculation
  calculateAnomalySeverity(ensembleScore, anomalyType) {
    // Base severity from score
    let severity = 'low';
    if (ensembleScore > 0.9) {
      severity = 'critical';
    } else if (ensembleScore > 0.75) {
      severity = 'high';
    } else if (ensembleScore > 0.6) {
      severity = 'medium';
    }

    // Adjust severity based on anomaly type
    const criticalTypes = ['data_exfiltration_attempt', 'privilege_escalation', 'authentication_anomaly'];
    const highTypes = ['security_violation', 'compliance_breach'];

    if (criticalTypes.includes(anomalyType)) {
      severity = 'critical';
    } else if (highTypes.includes(anomalyType) && severity !== 'critical') {
      severity = 'high';
    }

    return severity;
  }

  // Confidence Calculation
  calculateConfidence(anomalyScores) {
    const scores = Object.values(anomalyScores);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    const standardDeviation = Math.sqrt(variance);

    // Higher confidence for consistent high scores across models
    const consistency = 1 - (standardDeviation / mean);
    
    return Math.max(0.1, Math.min(1.0, consistency));
  }

  // Generate Anomaly Explanation
  async generateAnomalyExplanation(event, anomalyScores, anomalyType) {
    const explanations = {
      authentication_anomaly: 'Multiple failed login attempts detected from unusual location or time',
      data_exfiltration_attempt: 'Unusual data export pattern detected exceeding normal thresholds',
      privilege_escalation: 'Unauthorized role or permission change detected',
      off_hours_access: 'System access during unusual hours outside normal work patterns',
      behavioral_deviation: 'User behavior significantly deviates from established patterns',
      accessPattern_anomaly: 'Access to resources or actions outside normal usage patterns',
      temporal_anomaly: 'Access patterns showing unusual timing characteristics',
      statistical_anomaly: 'Statistical metrics falling outside expected ranges',
      security_anomaly: 'Multiple security indicators suggest potential threat'
    };

    let explanation = explanations[anomalyType] || 'Anomaly detected through multi-dimensional analysis';

    // Add specific details
    const topAnomalies = Object.entries(anomalyScores)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([type, score]) => `${type} (${(score * 100).toFixed(1)}%)`);

    if (topAnomalies.length > 0) {
      explanation += ` | Top indicators: ${topAnomalies.join(', ')}`;
    }

    return explanation;
  }

  // Recommended Actions
  async generateRecommendedActions(anomalyType, severity) {
    const actions = {
      critical: {
        authentication_anomaly: ['immediate_account_lock', 'security_team_notification', 'log_retention_extension'],
        data_exfiltration_attempt: ['immediate_access_suspension', 'security_incident_creation', 'data_integrity_check'],
        privilege_escalation: ['role_reversion', 'access_audit', 'privilege_review']
      },
      high: {
        authentication_anomaly: ['temp_account_lock', 'enhanced_monitoring', 'user_notification'],
        data_exfiltration_attempt: ['access_restriction', 'export_monitoring', 'supervisor_notification'],
        behavioral_deviation: ['behavioral_review', 'enhanced_logging', 'user_interview']
      },
      medium: {
        off_hours_access: ['activity_logging', 'pattern_analysis', 'user_awareness'],
        accessPattern_anomaly: ['usage_monitoring', 'pattern_study', 'training_review']
      },
      low: {
        temporal_anomaly: ['pattern_monitoring', 'baseline_update', 'trend_analysis']
      }
    };

    return actions[severity]?.[anomalyType] || ['monitor', 'analyze_pattern', 'review_baseline'];
  }

  // Gather Anomaly Context
  async gatherAnomalyContext(event) {
    return {
      userId: event.userId,
      eventTime: event.timestamp,
      relatedEvents: await this.getRelatedEvents(event.userId, event.timestamp),
      systemState: await this.getSystemState(),
      complianceImpact: await this.assessComplianceImpact(event),
      businessImpact: await this.assessBusinessImpact(event)
    };
  }

  // Feature Extraction
  extractFeatureValue(event, featureType) {
    const featureMap = {
      primary_metric: event.details?.riskScore || 0,
      access_time: new Date(event.timestamp).getHours(),
      resource_access: event.details?.resource ? 1 : 0,
      session_duration: event.details?.duration || 0,
      actions_frequency: 1, // Per event
      error_rate: event.event?.includes('failed') ? 1 : 0
    };

    return featureMap[featureType] || 0;
  }

  extractBehavioralFeatures(event) {
    const timestamp = new Date(event.timestamp);
    
    return {
      access_hour: timestamp.getHours(),
      access_day: timestamp.getDay(),
      access_week: Math.floor(timestamp.getTime() / (7 * 24 * 60 * 60 * 1000)),
      resource_type: this.categorizeResource(event.details?.resource),
      action_type: this.categorizeAction(event.event),
      session_length: event.details?.sessionDuration || 0,
      ip_block: this.getIPBlock(event.details?.ipAddress)
    };
  }

  categorizeResource(resource) {
    if (!resource) return 'unknown';
    
    if (resource.includes('patient')) return 'patient_data';
    if (resource.includes('admin')) return 'administrative';
    if (resource.includes('report')) return 'reporting';
    return 'general';
  }

  categorizeAction(action) {
    if (!action) return 'unknown';
    
    if (action.includes('read') || action.includes('view')) return 'read';
    if (action.includes('create') || action.includes('add')) return 'create';
    if (action.includes('update') || action.includes('edit')) return 'update';
    if (action.includes('delete') || action.includes('remove')) return 'delete';
    if (action.includes('export') || action.includes('download')) return 'export';
    return 'other';
  }

  getIPBlock(ipAddress) {
    if (!ipAddress) return 'unknown';
    return ipAddress.split('.').slice(0, 2).join('.');
  }

  // Utility Functions
  isOffHours(timestamp) {
    const hour = timestamp.getHours();
    return hour < 6 || hour > 22;
  }

  normalCDF(x) {
    // Approximation of standard normal cumulative distribution function
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  erf(x) {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  // Database Operations (Placeholder implementations)
  async getUserBaseline(userId, baselineType) {
    return await require('../database/user-database').getUserBaseline(userId, baselineType);
  }

  async getTemporalBaseline(userId) {
    return await require('../database/user-database').getTemporalBaseline(userId);
  }

  async getUserUsualIPs(userId) {
    return await require('../database/user-database').getUserUsualIPs(userId);
  }

  async getRecentAccessCount(userId, minutes) {
    const cutoff = new Date(Date.now() - (minutes * 60 * 1000)).toISOString();
    return await require('../database/user-database').getRecentAccessCount(userId, cutoff);
  }

  async getRelatedEvents(userId, timestamp) {
    const window = 60 * 60 * 1000; // 1 hour
    const start = new Date(new Date(timestamp).getTime() - window).toISOString();
    const end = new Date(new Date(timestamp).getTime() + window).toISOString();
    
    return await require('../database/user-database').getEventsInTimeRange(userId, start, end);
  }

  async getSystemState() {
    return await require('../database/user-database').getSystemState();
  }

  async assessComplianceImpact(event) {
    return await require('../database/user-database').assessComplianceImpact(event);
  }

  async assessBusinessImpact(event) {
    return await require('../database/user-database').assessBusinessImpact(event);
  }

  async storeAnomaly(anomaly) {
    await require('../database/user-database').storeAnomaly(anomaly);
  }

  // Initialize thresholds
  initializeThresholds() {
    return {
      statistical: 0.7,
      behavioral: 0.8,
      temporal: 0.6,
      accessPattern: 0.75,
      security: 0.8,
      ensemble: 0.65
    };
  }

  // Model training methods (placeholders)
  async trainModels(trainingData) {
    console.log('Training anomaly detection models...');
    // Implementation would include actual ML model training
    this.models.isolationForest.trained = true;
    this.models.statisticalModel.trained = true;
    this.models.timeSeriesModel.trained = true;
    this.models.behavioralModel.trained = true;
  }

  async updateBaselines(userId, event) {
    // Update user baselines with new event data
    const baseline = await this.getUserBaseline(userId, 'general') || {};
    
    // Update statistical baseline
    baseline.statistical = baseline.statistical || { count: 0, mean: 0, stdDev: 0 };
    baseline.statistical.count++;
    
    // Incremental mean and standard deviation update
    const value = this.extractFeatureValue(event, 'primary_metric');
    baseline.statistical.mean += (value - baseline.statistical.mean) / baseline.statistical.count;
    baseline.statistical.stdDev = Math.sqrt(
      ((baseline.statistical.stdDev ** 2) * (baseline.statistical.count - 1) + 
       (value - baseline.statistical.mean) ** 2) / baseline.statistical.count
    );
    
    await require('../database/user-database').updateUserBaseline(userId, baseline);
  }
}

module.exports = AnomalyDetector;