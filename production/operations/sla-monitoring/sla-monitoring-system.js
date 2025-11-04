#!/usr/bin/env node

/**
 * SLA Monitoring System for Medical AI Assistant
 * Tracks healthcare-specific service level agreements
 */

class SLAMonitoringSystem {
    constructor() {
        this.slAs = new Map();
        this.metrics = new Map();
        this.thresholds = new Map();
        this.reports = [];
        this.alertChannels = [];
        this.isInitialized = false;
    }

    /**
     * Initialize SLA monitoring system
     */
    async initialize() {
        console.log('📊 Initializing SLA Monitoring System...');
        
        await this.setupHealthcareSLAs();
        await this.setupMetricsCollection();
        await this.setupThresholds();
        await this.setupReporting();
        await this.setupAlerting();
        
        this.isInitialized = true;
        console.log('✅ SLA Monitoring System initialized successfully');
    }

    /**
     * Setup healthcare-specific SLAs
     */
    async setupHealthcareSLAs() {
        const slaDefinitions = [
            {
                name: 'patient_response_time',
                description: 'Response time for patient queries',
                target: 2000, // milliseconds
                current: 1850,
                status: 'met',
                severity: 'low',
                category: 'performance'
            },
            {
                name: 'system_availability',
                description: 'System uptime for clinical operations',
                target: 99.9, // percentage
                current: 99.94,
                status: 'met',
                severity: 'low',
                category: 'availability'
            },
            {
                name: 'diagnosis_accuracy',
                description: 'Accuracy of AI-assisted diagnoses',
                target: 95.0, // percentage
                current: 96.2,
                status: 'exceeded',
                severity: 'medium',
                category: 'clinical'
            },
            {
                name: 'phi_access_compliance',
                description: 'Compliance with PHI access controls',
                target: 100.0, // percentage
                current: 99.98,
                status: 'violated',
                severity: 'high',
                category: 'compliance'
            },
            {
                name: 'clinical_workflow_efficiency',
                description: 'Efficiency of clinical decision support',
                target: 90.0, // percentage
                current: 93.1,
                status: 'met',
                severity: 'low',
                category: 'clinical'
            },
            {
                name: 'emergency_response_time',
                description: 'Response time for emergency alerts',
                target: 30, // seconds
                current: 25,
                status: 'met',
                severity: 'medium',
                category: 'critical'
            },
            {
                name: 'data_backup_integrity',
                description: 'Backup system integrity and recovery',
                target: 100.0, // percentage
                current: 100.0,
                status: 'met',
                severity: 'medium',
                category: 'availability'
            },
            {
                name: 'audit_trail_completeness',
                description: 'Completeness of compliance audit trails',
                target: 100.0, // percentage
                current: 100.0,
                status: 'met',
                severity: 'medium',
                category: 'compliance'
            },
            {
                name: 'user_satisfaction_score',
                description: 'Healthcare provider satisfaction rating',
                target: 4.0, // out of 5
                current: 4.3,
                status: 'exceeded',
                severity: 'low',
                category: 'satisfaction'
            },
            {
                name: 'regulatory_compliance_score',
                description: 'Overall regulatory compliance score',
                target: 100.0, // percentage
                current: 100.0,
                status: 'met',
                severity: 'high',
                category: 'compliance'
            }
        ];

        for (const sla of slaDefinitions) {
            this.slAs.set(sla.name, {
                ...sla,
                violations: 0,
                lastViolation: null,
                history: []
            });
        }
    }

    /**
     * Setup metrics collection
     */
    async setupMetricsCollection() {
        const metricsCollectors = [
            {
                name: 'responseTimeCollector',
                interval: 30000, // 30 seconds
                method: this.collectResponseTimeMetrics.bind(this)
            },
            {
                name: 'availabilityCollector',
                interval: 60000, // 1 minute
                method: this.collectAvailabilityMetrics.bind(this)
            },
            {
                name: 'clinicalMetricsCollector',
                interval: 300000, // 5 minutes
                method: this.collectClinicalMetrics.bind(this)
            },
            {
                name: 'complianceMetricsCollector',
                interval: 600000, // 10 minutes
                method: this.collectComplianceMetrics.bind(this)
            },
            {
                name: 'securityMetricsCollector',
                interval: 60000, // 1 minute
                method: this.collectSecurityMetrics.bind(this)
            }
        ];

        for (const collector of metricsCollectors) {
            this.startMetricsCollection(collector);
        }
    }

    /**
     * Start metrics collection for a specific collector
     */
    startMetricsCollection(collector) {
        setInterval(async () => {
            try {
                await collector.method();
                await this.updateSLAMetrics(collector.name);
                await this.checkSLACompliance();
            } catch (error) {
                console.error(`❌ Metrics collection error for ${collector.name}:`, error);
            }
        }, collector.interval);
    }

    /**
     * Collect response time metrics
     */
    async collectResponseTimeMetrics() {
        // Simulate response time collection
        this.metrics.set('patient_response_time', {
            timestamp: new Date(),
            value: Math.random() * 500 + 1600, // 1600-2100ms range
            source: 'api_monitoring',
            breakdown: {
                api_gateway: 150,
                model_inference: 1200,
                database: 200,
                caching: 300
            }
        });
    }

    /**
     * Collect availability metrics
     */
    async collectAvailabilityMetrics() {
        // Simulate availability collection
        this.metrics.set('system_availability', {
            timestamp: new Date(),
            value: 99.94 + (Math.random() * 0.1 - 0.05), // 99.89-99.99% range
            source: 'uptime_monitoring',
            services: {
                api_gateway: 99.99,
                model_serving: 99.95,
                database: 99.98,
                cache: 99.92
            }
        });
    }

    /**
     * Collect clinical metrics
     */
    async collectClinicalMetrics() {
        // Simulate clinical metrics collection
        this.metrics.set('diagnosis_accuracy', {
            timestamp: new Date(),
            value: 96.2 + (Math.random() * 0.6 - 0.3), // 95.9-96.5% range
            source: 'clinical_assessment',
            sample_size: 1250,
            accuracy_by_specialty: {
                cardiology: 97.1,
                radiology: 96.8,
                pathology: 95.9,
                general_medicine: 96.4
            }
        });
    }

    /**
     * Collect compliance metrics
     */
    async collectComplianceMetrics() {
        // Simulate compliance metrics collection
        this.metrics.set('phi_access_compliance', {
            timestamp: new Date(),
            value: 99.98 + (Math.random() * 0.04 - 0.02), // 99.96-100.0% range
            source: 'compliance_monitoring',
            violations: 1,
            violations_detail: [
                { timestamp: '2025-11-04T08:15:22Z', type: 'after_hours_access', severity: 'low' }
            ]
        });
    }

    /**
     * Collect security metrics
     */
    async collectSecurityMetrics() {
        // Simulate security metrics collection
        this.metrics.set('security_status', {
            timestamp: new Date(),
            value: 'secure',
            incidents: 0,
            failed_attempts: 0,
            encryption_status: 'active',
            access_violations: 0
        });
    }

    /**
     * Setup performance thresholds
     */
    async setupThresholds() {
        const thresholds = {
            patient_response_time: {
                warning: 2500,
                critical: 5000,
                escalation_time: 300 // 5 minutes
            },
            system_availability: {
                warning: 99.5,
                critical: 99.0,
                escalation_time: 600 // 10 minutes
            },
            diagnosis_accuracy: {
                warning: 93.0,
                critical: 90.0,
                escalation_time: 1800 // 30 minutes
            },
            phi_access_compliance: {
                warning: 99.9,
                critical: 99.0,
                escalation_time: 60 // 1 minute
            },
            clinical_workflow_efficiency: {
                warning: 85.0,
                critical: 80.0,
                escalation_time: 900 // 15 minutes
            }
        };

        for (const [name, config] of Object.entries(thresholds)) {
            this.thresholds.set(name, config);
        }
    }

    /**
     * Setup reporting system
     */
    async setupReporting() {
        // Daily SLA report
        setInterval(() => {
            this.generateDailySLAReport();
        }, 24 * 60 * 60 * 1000); // Every 24 hours

        // Weekly SLA summary
        setInterval(() => {
            this.generateWeeklySLASummary();
        }, 7 * 24 * 60 * 60 * 1000); // Every 7 days

        // Monthly SLA analysis
        setInterval(() => {
            this.generateMonthlySLAAnalysis();
        }, 30 * 24 * 60 * 60 * 1000); // Every 30 days
    }

    /**
     * Setup alerting system
     */
    async setupAlerting() {
        this.alertChannels = [
            { type: 'email', recipients: ['sla-alerts@company.com'] },
            { type: 'slack', channel: '#sla-alerts' },
            { type: 'pagerduty', service_key: process.env.SLA_PAGERDUTY_KEY }
        ];
    }

    /**
     * Update SLA metrics with collected data
     */
    async updateSLAMetrics(collectorName) {
        const latestMetrics = Array.from(this.metrics.values())
            .filter(m => m.timestamp)
            .sort((a, b) => b.timestamp - a.timestamp);

        for (const sla of this.slAs.values()) {
            const matchingMetric = latestMetrics.find(m => 
                m.source === this.getMetricSource(sla.name)
            );

            if (matchingMetric) {
                await this.updateSLAValue(sla.name, matchingMetric);
            }
        }
    }

    /**
     * Get metric source for SLA
     */
    getMetricSource(slaName) {
        const sourceMap = {
            'patient_response_time': 'api_monitoring',
            'system_availability': 'uptime_monitoring',
            'diagnosis_accuracy': 'clinical_assessment',
            'phi_access_compliance': 'compliance_monitoring'
        };
        return sourceMap[slaName] || 'general_monitoring';
    }

    /**
     * Update SLA value and history
     */
    async updateSLAValue(slaName, metric) {
        const sla = this.slAs.get(slaName);
        if (!sla) return;

        const oldValue = sla.current;
        sla.current = typeof metric.value === 'number' ? metric.value : 0;
        sla.lastUpdate = metric.timestamp;

        // Add to history
        sla.history.push({
            timestamp: metric.timestamp,
            value: sla.current,
            status: this.evaluateSLAStatus(sla)
        });

        // Limit history to last 100 entries
        if (sla.history.length > 100) {
            sla.history = sla.history.slice(-100);
        }

        // Check for violations
        await this.checkSLAViolation(slaName, oldValue, sla.current);
    }

    /**
     * Evaluate SLA status based on current value and target
     */
    evaluateSLAStatus(sla) {
        const threshold = this.thresholds.get(sla.name);
        if (!threshold) return 'unknown';

        const { target } = sla;
        const { warning, critical } = threshold;

        if (sla.current >= target) return 'exceeded';
        if (sla.current >= warning) return 'met';
        if (sla.current >= critical) return 'warning';
        return 'violated';
    }

    /**
     * Check for SLA violations
     */
    async checkSLACompliance() {
        for (const [name, sla] of this.slAs) {
            const status = this.evaluateSLAStatus(sla);
            
            if (status === 'violated' && sla.status !== 'violated) {
                await this.handleSLAViolation(name, sla);
            }
            
            sla.status = status;
        }
    }

    /**
     * Handle SLA violation
     */
    async handleSLAViolation(slaName, sla) {
        sla.violations++;
        sla.lastViolation = new Date();
        
        const violation = {
            timestamp: sla.lastViolation,
            sla: slaName,
            current: sla.current,
            target: sla.target,
            severity: sla.severity,
            category: sla.category,
            description: sla.description
        };

        await this.sendSLAAlert(violation);
        this.reports.push(violation);
    }

    /**
     * Send SLA violation alert
     */
    async sendSLAAlert(violation) {
        console.log('🚨 SLA VIOLATION ALERT:');
        console.log(`   SLA: ${violation.sla}`);
        console.log(`   Current: ${violation.current}%`);
        console.log(`   Target: ${violation.target}%`);
        console.log(`   Severity: ${violation.severity}`);
        console.log(`   Category: ${violation.category}`);
        console.log(`   Description: ${violation.description}`);

        // Send to all alert channels
        for (const channel of this.alertChannels) {
            await this.sendAlertToChannel(violation, channel);
        }
    }

    /**
     * Send alert to specific channel
     */
    async sendAlertToChannel(violation, channel) {
        switch (channel.type) {
            case 'email':
                console.log(`   📧 Email alert sent to: ${channel.recipients.join(', ')}`);
                break;
            case 'slack':
                console.log(`   💬 Slack alert sent to #${channel.channel}`);
                break;
            case 'pagerduty':
                console.log(`   📞 PagerDuty escalation triggered`);
                break;
        }
    }

    /**
     * Generate daily SLA report
     */
    generateDailySLAReport() {
        const report = {
            date: new Date().toISOString().split('T')[0],
            generatedAt: new Date(),
            summary: this.generateSLAReportSummary(),
            detailedMetrics: this.getDetailedSLAMetrics(),
            violations: this.getDailyViolations(),
            trends: this.calculateSLATrends(),
            recommendations: this.generateSLARecommendations()
        };

        console.log('📊 Daily SLA Report Generated');
        console.log(JSON.stringify(report, null, 2));

        return report;
    }

    /**
     * Generate SLA report summary
     */
    generateSLAReportSummary() {
        const summary = {
            totalSLAs: this.slAs.size,
            met: 0,
            exceeded: 0,
            violated: 0,
            overallCompliance: 0
        };

        let totalCompliance = 0;
        for (const sla of this.slAs.values()) {
            totalCompliance += (sla.current / sla.target) * 100;
            
            if (sla.status === 'met') summary.met++;
            else if (sla.status === 'exceeded') summary.exceeded++;
            else if (sla.status === 'violated') summary.violated++;
        }

        summary.overallCompliance = totalCompliance / this.slAs.size;

        return summary;
    }

    /**
     * Get detailed SLA metrics
     */
    getDetailedSLAMetrics() {
        const metrics = {};
        for (const [name, sla] of this.slAs) {
            metrics[name] = {
                target: sla.target,
                current: sla.current,
                status: sla.status,
                violations: sla.violations,
                history: sla.history.slice(-24) // Last 24 hours
            };
        }
        return metrics;
    }

    /**
     * Get violations from today
     */
    getDailyViolations() {
        const today = new Date().toISOString().split('T')[0];
        return this.reports.filter(v => 
            v.timestamp.toISOString().split('T')[0] === today
        );
    }

    /**
     * Calculate SLA trends
     */
    calculateSLATrends() {
        const trends = {};
        for (const [name, sla] of this.slAs) {
            if (sla.history.length >= 2) {
                const recent = sla.history.slice(-10);
                const older = sla.history.slice(-20, -10);
                
                const recentAvg = recent.reduce((sum, h) => sum + h.value, 0) / recent.length;
                const olderAvg = older.reduce((sum, h) => sum + h.value, 0) / older.length;
                
                trends[name] = {
                    direction: recentAvg > olderAvg ? 'improving' : 'declining',
                    change: recentAvg - olderAvg,
                    percentage: ((recentAvg - olderAvg) / olderAvg) * 100
                };
            }
        }
        return trends;
    }

    /**
     * Generate SLA recommendations
     */
    generateSLARecommendations() {
        const recommendations = [];
        
        for (const [name, sla] of this.slAs) {
            if (sla.status === 'violated') {
                recommendations.push({
                    sla: name,
                    priority: 'high',
                    action: 'Immediate investigation required',
                    impact: sla.category === 'clinical' ? 'Patient care impact' : 'Operational impact'
                });
            } else if (sla.status === 'warning') {
                recommendations.push({
                    sla: name,
                    priority: 'medium',
                    action: 'Monitor closely and optimize',
                    impact: 'Prevent SLA violation'
                });
            } else if (sla.status === 'exceeded') {
                recommendations.push({
                    sla: name,
                    priority: 'low',
                    action: 'Document best practices',
                    impact: 'Maintain excellent performance'
                });
            }
        }
        
        return recommendations;
    }

    /**
     * Generate weekly SLA summary
     */
    generateWeeklySLASummary() {
        const weekStart = new Date();
        weekStart.setDate(weekStart.getDate() - 7);
        
        const weeklyViolations = this.reports.filter(v => v.timestamp >= weekStart);
        
        const summary = {
            week: `${weekStart.toISOString().split('T')[0]} to ${new Date().toISOString().split('T')[0]}`,
            totalViolations: weeklyViolations.length,
            slaPerformance: this.generateWeeklyPerformanceMetrics(),
            topIssues: this.identifyTopIssues(weeklyViolations),
            improvementPlan: this.generateImprovementPlan(weeklyViolations)
        };

        console.log('📈 Weekly SLA Summary Generated');
        console.log(JSON.stringify(summary, null, 2));

        return summary;
    }

    /**
     * Generate weekly performance metrics
     */
    generateWeeklyPerformanceMetrics() {
        const metrics = {};
        for (const [name, sla] of this.slAs) {
            const weekHistory = sla.history.filter(h => 
                h.timestamp >= new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
            );
            
            if (weekHistory.length > 0) {
                const avgPerformance = weekHistory.reduce((sum, h) => sum + (h.value / sla.target) * 100, 0) / weekHistory.length;
                metrics[name] = {
                    averagePerformance: avgPerformance.toFixed(2),
                    complianceRate: (weekHistory.filter(h => this.evaluateSLAStatus(sla) !== 'violated').length / weekHistory.length * 100).toFixed(2),
                    trend: this.calculateWeeklyTrend(weekHistory, sla.target)
                };
            }
        }
        return metrics;
    }

    /**
     * Calculate weekly trend
     */
    calculateWeeklyTrend(history, target) {
        if (history.length < 2) return 'insufficient_data';
        
        const firstHalf = history.slice(0, Math.floor(history.length / 2));
        const secondHalf = history.slice(Math.floor(history.length / 2));
        
        const firstAvg = firstHalf.reduce((sum, h) => sum + h.value, 0) / firstHalf.length;
        const secondAvg = secondHalf.reduce((sum, h) => sum + h.value, 0) / secondHalf.length;
        
        if (secondAvg > firstAvg * 1.02) return 'improving';
        if (secondAvg < firstAvg * 0.98) return 'declining';
        return 'stable';
    }

    /**
     * Identify top issues from violations
     */
    identifyTopIssues(violations) {
        const issueCounts = {};
        for (const violation of violations) {
            issueCounts[violation.sla] = (issueCounts[violation.sla] || 0) + 1;
        }
        
        return Object.entries(issueCounts)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5)
            .map(([sla, count]) => ({
                sla,
                violationCount: count,
                slaDetails: this.slAs.get(sla)
            }));
    }

    /**
     * Generate improvement plan based on violations
     */
    generateImprovementPlan(violations) {
        const plan = {
            immediateActions: [],
            shortTermGoals: [],
            longTermStrategy: [],
            resourceRequirements: []
        };

        const criticalViolations = violations.filter(v => v.severity === 'high' || v.severity === 'critical');
        const performanceViolations = violations.filter(v => v.category === 'performance');
        
        if (criticalViolations.length > 0) {
            plan.immediateActions.push('Address critical SLA violations within 24 hours');
            plan.resourceRequirements.push('Dedicated engineering resources for critical issues');
        }
        
        if (performanceViolations.length > 0) {
            plan.shortTermGoals.push('Implement performance optimization measures');
            plan.resourceRequirements.push('Performance monitoring and optimization tools');
        }
        
        plan.longTermStrategy.push('Establish proactive SLA monitoring and predictive alerting');
        plan.longTermStrategy.push('Regular SLA review and optimization processes');
        
        return plan;
    }

    /**
     * Generate monthly SLA analysis
     */
    generateMonthlySLAAnalysis() {
        const monthStart = new Date();
        monthStart.setDate(monthStart.getDate() - 30);
        
        const monthlyData = this.reports.filter(v => v.timestamp >= monthStart);
        
        const analysis = {
            month: monthStart.toISOString().split('T')[0],
            totalViolations: monthlyData.length,
            complianceScore: this.calculateMonthlyCompliance(),
            performanceTrends: this.calculateMonthlyTrends(),
            regulatoryImpact: this.assessRegulatoryImpact(),
            executiveSummary: this.generateExecutiveSummary(monthlyData)
        };

        console.log('📊 Monthly SLA Analysis Generated');
        console.log(JSON.stringify(analysis, null, 2));

        return analysis;
    }

    /**
     * Calculate monthly compliance score
     */
    calculateMonthlyCompliance() {
        let totalCompliance = 0;
        let slaCount = 0;
        
        for (const sla of this.slAs.values()) {
            const monthHistory = sla.history.filter(h => 
                h.timestamp >= new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
            );
            
            if (monthHistory.length > 0) {
                const compliance = monthHistory.filter(h => 
                    this.evaluateSLAStatus(sla) !== 'violated'
                ).length / monthHistory.length * 100;
                totalCompliance += compliance;
                slaCount++;
            }
        }
        
        return slaCount > 0 ? totalCompliance / slaCount : 0;
    }

    /**
     * Calculate monthly trends
     */
    calculateMonthlyTrends() {
        const trends = {};
        for (const [name, sla] of this.slAs) {
            const monthHistory = sla.history.filter(h => 
                h.timestamp >= new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
            );
            
            if (monthHistory.length > 7) {
                const weeks = Math.floor(monthHistory.length / 7);
                const weeklyData = [];
                
                for (let i = 0; i < weeks; i++) {
                    const weekStart = i * 7;
                    const weekEnd = Math.min(weekStart + 7, monthHistory.length);
                    const weekAvg = monthHistory.slice(weekStart, weekEnd)
                        .reduce((sum, h) => sum + h.value, 0) / (weekEnd - weekStart);
                    weeklyData.push(weekAvg);
                }
                
                trends[name] = {
                    weeklyData,
                    direction: weeklyData[weeklyData.length - 1] > weeklyData[0] ? 'improving' : 'declining',
                    consistency: this.calculateConsistency(weeklyData)
                };
            }
        }
        return trends;
    }

    /**
     * Calculate consistency score
     */
    calculateConsistency(data) {
        if (data.length < 2) return 0;
        
        const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
        const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
        const stdDev = Math.sqrt(variance);
        
        return Math.max(0, 100 - (stdDev / mean) * 100);
    }

    /**
     * Assess regulatory impact
     */
    assessRegulatoryImpact() {
        const complianceSLAs = Array.from(this.slAs.values()).filter(sla => 
            sla.category === 'compliance' || sla.category === 'critical'
        );
        
        const violations = complianceSLAs.filter(sla => sla.violations > 0);
        
        return {
            totalComplianceSLAs: complianceSLAs.length,
            violatedSLAs: violations.length,
            riskLevel: violations.length > 0 ? 'medium' : 'low',
            regulatoryFrameworks: ['HIPAA', 'FDA 21 CFR Part 820', 'HITECH'],
            complianceScore: complianceSLAs.length > 0 ? 
                (complianceSLAs.filter(sla => sla.status !== 'violated').length / complianceSLAs.length * 100) : 100
        };
    }

    /**
     * Generate executive summary
     */
    generateExecutiveSummary(violations) {
        const totalSLAs = this.slAs.size;
        const violatedSLAs = new Set(violations.map(v => v.sla)).size;
        const complianceRate = ((totalSLAs - violatedSLAs) / totalSLAs * 100).toFixed(1);
        
        const topViolations = violations.reduce((acc, v) => {
            acc[v.sla] = (acc[v.sla] || 0) + 1;
            return acc;
        }, {});
        
        const mostProblematic = Object.entries(topViolations)
            .sort(([,a], [,b]) => b - a)[0];
        
        return {
            overallCompliance: `${complianceRate}%`,
            criticalIssues: violations.filter(v => v.severity === 'high' || v.severity === 'critical').length,
            mostProblematicSLA: mostProblematic ? mostProblematic[0] : 'None',
            recommendation: violations.length > 0 ? 
                'Immediate attention required for SLA violations' : 
                'Maintained excellent SLA compliance',
            trendDirection: 'stable' // Simplified for demo
        };
    }

    /**
     * Get current SLA status dashboard
     */
    getSLADashboard() {
        return {
            timestamp: new Date(),
            overallStatus: this.calculateOverallSLAStatus(),
            slas: Object.fromEntries(this.slAs),
            activeAlerts: this.reports.filter(r => 
                r.timestamp > new Date(Date.now() - 3600000) // Last hour
            ),
            complianceSummary: this.generateSLAReportSummary()
        };
    }

    /**
     * Calculate overall SLA status
     */
    calculateOverallSLAStatus() {
        const statuses = Array.from(this.slAs.values()).map(sla => sla.status);
        
        if (statuses.includes('violated')) return 'critical';
        if (statuses.includes('warning')) return 'warning';
        if (statuses.every(s => s === 'met' || s === 'exceeded')) return 'healthy';
        return 'unknown';
    }
}

// CLI Interface
if (require.main === module) {
    const sla = new SLAMonitoringSystem();
    
    sla.initialize().then(() => {
        console.log('📊 SLA Monitoring System is running...');
        
        // Generate initial dashboard
        setTimeout(() => {
            const dashboard = sla.getSLADashboard();
            console.log('\n🎯 SLA Dashboard:');
            console.log(JSON.stringify(dashboard, null, 2));
        }, 3000);
        
    }).catch(error => {
        console.error('❌ Failed to initialize SLA Monitoring:', error);
        process.exit(1);
    });
    
    process.on('SIGINT', () => {
        console.log('\n🛑 Shutting down SLA Monitoring System...');
        process.exit(0);
    });
}

module.exports = SLAMonitoringSystem;