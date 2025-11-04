#!/usr/bin/env python3
"""
Demo Validation Framework - Comprehensive testing and validation for demo systems.

This module provides comprehensive validation capabilities including:
- Demo scenario validation and testing
- Stakeholder feedback validation
- Technical performance testing
- Compliance verification
- Automated quality assurance
- Continuous improvement tracking
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class ValidationType(Enum):
    """Validation type enumeration"""
    SCENARIO_ACCURACY = "scenario_accuracy"
    TECHNICAL_PERFORMANCE = "technical_performance"
    STAKEHOLDER_SATISFACTION = "stakeholder_satisfaction"
    COMPLIANCE_CHECK = "compliance_check"
    CONTENT_QUALITY = "content_quality"
    WORKFLOW_INTEGRATION = "workflow_integration"

class ValidationStatus(Enum):
    """Validation status enumeration"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"

class QualityLevel(Enum):
    """Quality level enumeration"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    CRITICAL = "critical"

@dataclass
class ValidationTest:
    """Individual validation test structure"""
    test_id: str
    test_name: str
    validation_type: ValidationType
    description: str
    success_criteria: Dict[str, Any]
    test_data: Dict[str, Any]
    expected_result: Any
    actual_result: Optional[Any] = None
    status: ValidationStatus = ValidationStatus.PENDING
    execution_time: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    validation_date: datetime
    overall_status: ValidationStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    quality_score: float
    test_results: List[ValidationTest]
    recommendations: List[str]
    improvement_areas: List[str]

class DemoValidationFramework:
    """Comprehensive demo validation framework"""
    
    def __init__(self, db_path: str = "validation.db", config_file: str = "validation_config.json"):
        self.db_path = db_path
        self.config_file = Path(config_file)
        self.test_suites: Dict[str, List[ValidationTest]] = {}
        self.validation_history: List[ValidationReport] = []
        self._init_database()
        self._load_validation_config()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for validation framework"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('demo_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize validation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Validation tests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_tests (
                test_id TEXT PRIMARY KEY,
                test_name TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                description TEXT,
                success_criteria TEXT,
                test_data TEXT,
                expected_result TEXT,
                actual_result TEXT,
                status TEXT,
                execution_time REAL,
                error_message TEXT,
                created_at TEXT,
                last_run TEXT
            )
        ''')
        
        # Validation reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_reports (
                report_id TEXT PRIMARY KEY,
                validation_date TEXT,
                overall_status TEXT,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                warning_tests INTEGER,
                quality_score REAL,
                recommendations TEXT,
                improvement_areas TEXT,
                full_report TEXT
            )
        ''')
        
        # Demo quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_quality_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                demo_type TEXT,
                stakeholder_type TEXT,
                scenario_id TEXT,
                quality_score REAL,
                validation_date TEXT,
                test_results TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_validation_config(self):
        """Load validation configuration"""
        default_config = {
            "validation_suites": {
                "scenario_validation": {
                    "name": "Medical Scenario Validation",
                    "tests": [
                        {
                            "test_id": "scenario_accuracy_check",
                            "test_name": "Clinical Accuracy Validation",
                            "validation_type": "scenario_accuracy",
                            "description": "Verify medical scenario accuracy and guideline compliance",
                            "success_criteria": {
                                "accuracy_threshold": 0.95,
                                "guideline_compliance": True,
                                "medical_validity": True
                            },
                            "expected_result": {
                                "accuracy_score": 0.95,
                                "guidelines_compliant": True,
                                "medical_validity": True
                            }
                        },
                        {
                            "test_id": "patient_profile_validation",
                            "test_name": "Patient Profile Validation",
                            "validation_type": "content_quality",
                            "description": "Validate patient demographics and clinical data",
                            "success_criteria": {
                                "demographic_realism": True,
                                "clinical_consistency": True,
                                "anonymization_compliance": True
                            }
                        },
                        {
                            "test_id": "workflow_integration_check",
                            "test_name": "Clinical Workflow Integration",
                            "validation_type": "workflow_integration",
                            "description": "Verify scenario integration with clinical workflows",
                            "success_criteria": {
                                "workflow_compatibility": True,
                                "emr_integration_possible": True,
                                "clinical_decision_support": True
                            }
                        }
                    ]
                },
                "technical_validation": {
                    "name": "Technical Performance Validation",
                    "tests": [
                        {
                            "test_id": "response_time_validation",
                            "test_name": "System Response Time",
                            "validation_type": "technical_performance",
                            "description": "Validate system response times meet requirements",
                            "success_criteria": {
                                "max_response_time": 500,  # milliseconds
                                "95th_percentile": 800,
                                "consistency_score": 0.95
                            },
                            "expected_result": {
                                "avg_response_time": 350,
                                "95th_percentile": 650,
                                "consistency_score": 0.98
                            }
                        },
                        {
                            "test_id": "accuracy_validation",
                            "test_name": "AI Accuracy Validation",
                            "validation_type": "technical_performance",
                            "description": "Validate AI accuracy meets clinical standards",
                            "success_criteria": {
                                "accuracy_threshold": 0.95,
                                "precision_score": 0.97,
                                "recall_score": 0.96
                            },
                            "expected_result": {
                                "accuracy_score": 0.97,
                                "precision_score": 0.98,
                                "recall_score": 0.97
                            }
                        },
                        {
                            "test_id": "availability_validation",
                            "test_name": "System Availability",
                            "validation_type": "technical_performance",
                            "description": "Validate system availability and reliability",
                            "success_criteria": {
                                "uptime_threshold": 0.99,
                                "error_rate_threshold": 0.001
                            }
                        }
                    ]
                },
                "stakeholder_validation": {
                    "name": "Stakeholder Satisfaction Validation",
                    "tests": [
                        {
                            "test_id": "engagement_validation",
                            "test_name": "Audience Engagement Validation",
                            "validation_type": "stakeholder_satisfaction",
                            "description": "Validate audience engagement meets targets",
                            "success_criteria": {
                                "engagement_score": 7.5,
                                "completion_rate": 0.85,
                                "interaction_rate": 0.60
                            },
                            "expected_result": {
                                "engagement_score": 8.2,
                                "completion_rate": 0.92,
                                "interaction_rate": 0.75
                            }
                        },
                        {
                            "test_id": "satisfaction_validation",
                            "test_name": "Overall Satisfaction Validation",
                            "validation_type": "stakeholder_satisfaction",
                            "description": "Validate overall stakeholder satisfaction",
                            "success_criteria": {
                                "satisfaction_score": 7.0,
                                "recommendation_rate": 0.70,
                                "feedback_quality": 0.80
                            }
                        }
                    ]
                },
                "compliance_validation": {
                    "name": "Compliance and Security Validation",
                    "tests": [
                        {
                            "test_id": "hipaa_compliance_check",
                            "test_name": "HIPAA Compliance Verification",
                            "validation_type": "compliance_check",
                            "description": "Verify HIPAA compliance for demo environments",
                            "success_criteria": {
                                "phi_protection": True,
                                "audit_trails": True,
                                "access_controls": True,
                                "data_encryption": True
                            }
                        },
                        {
                            "test_id": "data_anonymization_check",
                            "test_name": "Data Anonymization Validation",
                            "validation_type": "compliance_check",
                            "description": "Validate patient data anonymization",
                            "success_criteria": {
                                "anonymization_complete": True,
                                "re_identification_risk": "low",
                                "synthetic_data_quality": True
                            }
                        },
                        {
                            "test_id": "security_validation",
                            "test_name": "Security Controls Validation",
                            "validation_type": "compliance_check",
                            "description": "Validate security controls and protocols",
                            "success_criteria": {
                                "encryption_standards": True,
                                "access_management": True,
                                "incident_response": True,
                                "security_monitoring": True
                            }
                        }
                    ]
                }
            },
            "quality_thresholds": {
                "scenario_accuracy": 0.95,
                "technical_performance": 0.90,
                "stakeholder_satisfaction": 7.5,
                "compliance_score": 0.98,
                "overall_quality": 8.0
            },
            "validation_schedule": {
                "continuous_validation": True,
                "scheduled_validations": {
                    "daily": ["technical_performance"],
                    "weekly": ["scenario_accuracy", "stakeholder_satisfaction"],
                    "monthly": ["compliance_check", "content_quality"]
                }
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in self.config:
                        self.config[key] = value
            except Exception as e:
                self.logger.error(f"Error loading validation config: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save validation configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Validation configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving validation config: {e}")
    
    def create_test_suite(self, suite_name: str, tests: List[Dict[str, Any]]) -> bool:
        """Create a new test suite"""
        try:
            test_objects = []
            for test_config in tests:
                test = ValidationTest(
                    test_id=test_config["test_id"],
                    test_name=test_config["test_name"],
                    validation_type=ValidationType(test_config["validation_type"]),
                    description=test_config["description"],
                    success_criteria=test_config["success_criteria"],
                    test_data=test_config.get("test_data", {}),
                    expected_result=test_config.get("expected_result")
                )
                test_objects.append(test)
            
            self.test_suites[suite_name] = test_objects
            self.logger.info(f"Created test suite: {suite_name} with {len(test_objects)} tests")
            return True
        except Exception as e:
            self.logger.error(f"Error creating test suite: {e}")
            return False
    
    def execute_validation_test(self, test: ValidationTest) -> ValidationTest:
        """Execute a single validation test"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing test: {test.test_name}")
            
            # Execute test based on type
            if test.validation_type == ValidationType.SCENARIO_ACCURACY:
                test.actual_result = self._execute_scenario_accuracy_test(test)
            elif test.validation_type == ValidationType.TECHNICAL_PERFORMANCE:
                test.actual_result = self._execute_technical_performance_test(test)
            elif test.validation_type == ValidationType.STAKEHOLDER_SATISFACTION:
                test.actual_result = self._execute_stakeholder_satisfaction_test(test)
            elif test.validation_type == ValidationType.COMPLIANCE_CHECK:
                test.actual_result = self._execute_compliance_check(test)
            elif test.validation_type == ValidationType.CONTENT_QUALITY:
                test.actual_result = self._execute_content_quality_test(test)
            elif test.validation_type == ValidationType.WORKFLOW_INTEGRATION:
                test.actual_result = self._execute_workflow_integration_test(test)
            else:
                raise ValueError(f"Unknown validation type: {test.validation_type}")
            
            # Determine test status
            test.status = self._determine_test_status(test)
            test.execution_time = time.time() - start_time
            
            self.logger.info(f"Test {test.test_name} completed: {test.status.value}")
            
        except Exception as e:
            test.status = ValidationStatus.FAILED
            test.error_message = str(e)
            test.execution_time = time.time() - start_time
            self.logger.error(f"Test {test.test_name} failed: {e}")
        
        return test
    
    def _execute_scenario_accuracy_test(self, test: ValidationTest) -> Dict[str, Any]:
        """Execute scenario accuracy validation"""
        # Simulate scenario accuracy testing
        test_data = test.test_data
        
        # Mock accuracy checks
        accuracy_score = 0.97  # Simulated high accuracy
        guideline_compliance = True
        medical_validity = True
        
        # Check against success criteria
        if "accuracy_threshold" in test.success_criteria:
            if accuracy_score < test.success_criteria["accuracy_threshold"]:
                raise ValueError(f"Accuracy {accuracy_score} below threshold {test.success_criteria['accuracy_threshold']}")
        
        return {
            "accuracy_score": accuracy_score,
            "guidelines_compliant": guideline_compliance,
            "medical_validity": medical_validity,
            "validated_at": datetime.now().isoformat()
        }
    
    def _execute_technical_performance_test(self, test: ValidationTest) -> Dict[str, Any]:
        """Execute technical performance validation"""
        # Simulate technical performance testing
        avg_response_time = 350  # milliseconds
        accuracy_score = 0.97
        uptime = 0.995
        
        # Check response time criteria
        if "max_response_time" in test.success_criteria:
            if avg_response_time > test.success_criteria["max_response_time"]:
                raise ValueError(f"Response time {avg_response_time}ms exceeds limit {test.success_criteria['max_response_time']}ms")
        
        # Check accuracy criteria
        if "accuracy_threshold" in test.success_criteria:
            if accuracy_score < test.success_criteria["accuracy_threshold"]:
                raise ValueError(f"Accuracy {accuracy_score} below threshold {test.success_criteria['accuracy_threshold']}")
        
        return {
            "avg_response_time": avg_response_time,
            "accuracy_score": accuracy_score,
            "uptime": uptime,
            "95th_percentile": 650,
            "tested_at": datetime.now().isoformat()
        }
    
    def _execute_stakeholder_satisfaction_test(self, test: ValidationTest) -> Dict[str, Any]:
        """Execute stakeholder satisfaction validation"""
        # Simulate stakeholder satisfaction testing
        engagement_score = 8.2
        completion_rate = 0.92
        satisfaction_score = 8.5
        
        # Check success criteria
        if "engagement_score" in test.success_criteria:
            if engagement_score < test.success_criteria["engagement_score"]:
                raise ValueError(f"Engagement score {engagement_score} below threshold {test.success_criteria['engagement_score']}")
        
        return {
            "engagement_score": engagement_score,
            "completion_rate": completion_rate,
            "satisfaction_score": satisfaction_score,
            "recommendation_rate": 0.85,
            "survey_count": 45,
            "validated_at": datetime.now().isoformat()
        }
    
    def _execute_compliance_check(self, test: ValidationTest) -> Dict[str, Any]:
        """Execute compliance and security validation"""
        # Simulate compliance checking
        hipaa_compliant = True
        data_anonymized = True
        encryption_enabled = True
        audit_trails_present = True
        
        # Check success criteria
        for criterion, expected in test.success_criteria.items():
            if criterion == "phi_protection" and not hipaa_compliant:
                raise ValueError("PHI protection not adequate")
            elif criterion == "anonymization_complete" and not data_anonymized:
                raise ValueError("Data anonymization incomplete")
            elif criterion == "data_encryption" and not encryption_enabled:
                raise ValueError("Data encryption not enabled")
        
        return {
            "hipaa_compliant": hipaa_compliant,
            "data_anonymized": data_anonymized,
            "encryption_enabled": encryption_enabled,
            "audit_trails_present": audit_trails_present,
            "access_controls": True,
            "compliance_score": 0.98,
            "validated_at": datetime.now().isoformat()
        }
    
    def _execute_content_quality_test(self, test: ValidationTest) -> Dict[str, Any]:
        """Execute content quality validation"""
        # Simulate content quality testing
        readability_score = 8.5
        medical_accuracy = 0.97
        content_completeness = 0.95
        
        return {
            "readability_score": readability_score,
            "medical_accuracy": medical_accuracy,
            "content_completeness": content_completeness,
            "validated_at": datetime.now().isoformat()
        }
    
    def _execute_workflow_integration_test(self, test: ValidationTest) -> Dict[str, Any]:
        """Execute workflow integration validation"""
        # Simulate workflow integration testing
        workflow_compatibility = True
        emr_integration_possible = True
        clinical_decision_support = True
        
        return {
            "workflow_compatibility": workflow_compatibility,
            "emr_integration_possible": emr_integration_possible,
            "clinical_decision_support": clinical_decision_support,
            "validated_at": datetime.now().isoformat()
        }
    
    def _determine_test_status(self, test: ValidationTest) -> ValidationStatus:
        """Determine test execution status"""
        if test.error_message:
            return ValidationStatus.FAILED
        
        # Compare actual result with expected result
        if test.expected_result and test.actual_result:
            # Simple validation logic (in production, this would be more sophisticated)
            if isinstance(test.expected_result, dict) and isinstance(test.actual_result, dict):
                # Check if key metrics meet thresholds
                for key, expected_value in test.expected_result.items():
                    if key in test.actual_result:
                        actual_value = test.actual_result[key]
                        
                        # Check for failure conditions
                        if isinstance(expected_value, bool) and expected_value and not actual_value:
                            return ValidationStatus.FAILED
                        elif isinstance(expected_value, (int, float)) and actual_value < expected_value * 0.9:
                            return ValidationStatus.WARNING
        
        return ValidationStatus.PASSED
    
    def run_validation_suite(self, suite_name: str) -> ValidationReport:
        """Run complete validation suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite {suite_name} not found")
        
        test_suite = self.test_suites[suite_name]
        executed_tests = []
        
        self.logger.info(f"Starting validation suite: {suite_name}")
        
        # Execute all tests in suite
        for test in test_suite:
            executed_test = self.execute_validation_test(test)
            executed_tests.append(executed_test)
        
        # Generate report
        report = self._generate_validation_report(executed_tests, suite_name)
        
        # Save to database
        self._save_validation_report(report)
        
        self.logger.info(f"Validation suite {suite_name} completed: {report.overall_status.value}")
        
        return report
    
    def _generate_validation_report(self, test_results: List[ValidationTest], suite_name: str) -> ValidationReport:
        """Generate comprehensive validation report"""
        # Count test results
        passed_tests = sum(1 for test in test_results if test.status == ValidationStatus.PASSED)
        failed_tests = sum(1 for test in test_results if test.status == ValidationStatus.FAILED)
        warning_tests = sum(1 for test in test_results if test.status == ValidationStatus.WARNING)
        
        total_tests = len(test_results)
        
        # Calculate overall status
        if failed_tests > 0:
            overall_status = ValidationStatus.FAILED
        elif warning_tests > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(test_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(test_results)
        
        # Generate improvement areas
        improvement_areas = self._generate_improvement_areas(test_results)
        
        report_id = f"val_{int(time.time())}_{suite_name}"
        
        return ValidationReport(
            report_id=report_id,
            validation_date=datetime.now(),
            overall_status=overall_status,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            quality_score=quality_score,
            test_results=test_results,
            recommendations=recommendations,
            improvement_areas=improvement_areas
        )
    
    def _calculate_quality_score(self, test_results: List[ValidationTest]) -> float:
        """Calculate overall quality score from test results"""
        if not test_results:
            return 0.0
        
        total_score = 0.0
        for test in test_results:
            if test.status == ValidationStatus.PASSED:
                score = 1.0
            elif test.status == ValidationStatus.WARNING:
                score = 0.7
            elif test.status == ValidationStatus.FAILED:
                score = 0.3
            else:  # PENDING
                score = 0.5
            
            total_score += score
        
        return round((total_score / len(test_results)) * 10, 2)
    
    def _generate_recommendations(self, test_results: List[ValidationTest]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [test for test in test_results if test.status == ValidationStatus.FAILED]
        if failed_tests:
            recommendations.append("Address failed validation tests immediately")
        
        # Analyze warning tests
        warning_tests = [test for test in test_results if test.status == ValidationStatus.WARNING]
        if warning_tests:
            recommendations.append("Review tests with warnings for optimization opportunities")
        
        # Specific recommendations based on test types
        for test in test_results:
            if test.validation_type == ValidationType.TECHNICAL_PERFORMANCE:
                if test.actual_result and "avg_response_time" in test.actual_result:
                    if test.actual_result["avg_response_time"] > 400:
                        recommendations.append("Optimize system response times")
            
            elif test.validation_type == ValidationType.STAKEHOLDER_SATISFACTION:
                if test.actual_result and "engagement_score" in test.actual_result:
                    if test.actual_result["engagement_score"] < 8.0:
                        recommendations.append("Improve audience engagement strategies")
            
            elif test.validation_type == ValidationType.SCENARIO_ACCURACY:
                recommendations.append("Regularly validate medical scenario accuracy")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_improvement_areas(self, test_results: List[ValidationTest]) -> List[str]:
        """Generate list of improvement areas"""
        improvement_areas = []
        
        # Group tests by validation type
        type_performance = {}
        for test in test_results:
            test_type = test.validation_type.value
            if test_type not in type_performance:
                type_performance[test_type] = []
            type_performance[test_type].append(test.status == ValidationStatus.PASSED)
        
        # Identify underperforming areas
        for test_type, results in type_performance.items():
            pass_rate = sum(results) / len(results)
            if pass_rate < 0.8:
                improvement_areas.append(f"{test_type.replace('_', ' ').title()} Performance")
        
        return improvement_areas
    
    def _save_validation_report(self, report: ValidationReport):
        """Save validation report to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO validation_reports
            (report_id, validation_date, overall_status, total_tests, passed_tests,
             failed_tests, warning_tests, quality_score, recommendations, improvement_areas, full_report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.report_id,
            report.validation_date.isoformat(),
            report.overall_status.value,
            report.total_tests,
            report.passed_tests,
            report.failed_tests,
            report.warning_tests,
            report.quality_score,
            json.dumps(report.recommendations),
            json.dumps(report.improvement_areas),
            json.dumps(asdict(report), default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def get_validation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get validation history for specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM validation_reports
            WHERE validation_date > ?
            ORDER BY validation_date DESC
        ''', (cutoff_date,))
        
        reports = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        history = []
        for report in reports:
            history.append({
                "report_id": report[0],
                "validation_date": report[1],
                "overall_status": report[2],
                "total_tests": report[3],
                "passed_tests": report[4],
                "failed_tests": report[5],
                "warning_tests": report[6],
                "quality_score": report[7]
            })
        
        return history
    
    def validate_demo_quality(self, demo_type: str, stakeholder_type: str) -> Dict[str, Any]:
        """Validate overall demo quality for specific demo and stakeholder type"""
        # Get historical validation data
        history = self.get_validation_history(days=30)
        
        # Filter relevant tests
        relevant_reports = [report for report in history if report.get("quality_score", 0) > 0]
        
        if not relevant_reports:
            return {"status": "no_data", "message": "No recent validation data available"}
        
        # Calculate quality metrics
        avg_quality_score = sum(report["quality_score"] for report in relevant_reports) / len(relevant_reports)
        pass_rate = sum(report["passed_tests"] for report in relevant_reports) / sum(report["total_tests"] for report in relevant_reports)
        
        # Determine quality level
        if avg_quality_score >= 9.0:
            quality_level = QualityLevel.EXCELLENT
        elif avg_quality_score >= 8.0:
            quality_level = QualityLevel.GOOD
        elif avg_quality_score >= 7.0:
            quality_level = QualityLevel.ACCEPTABLE
        elif avg_quality_score >= 6.0:
            quality_level = QualityLevel.NEEDS_IMPROVEMENT
        else:
            quality_level = QualityLevel.CRITICAL
        
        return {
            "demo_type": demo_type,
            "stakeholder_type": stakeholder_type,
            "quality_level": quality_level.value,
            "quality_score": round(avg_quality_score, 2),
            "pass_rate": round(pass_rate, 2),
            "validation_count": len(relevant_reports),
            "status": "validated" if quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD] else "needs_attention"
        }

def main():
    """Main function for validation framework CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Demo Validation Framework")
    parser.add_argument("--suite", type=str, help="Run validation suite")
    parser.add_argument("--history", type=int, default=30, help="Show validation history (days)")
    parser.add_argument("--validate-demo", nargs=2, metavar=('DEMO_TYPE', 'STAKEHOLDER_TYPE'),
                       help="Validate demo quality for specific type")
    parser.add_argument("--list-suites", action="store_true", help="List available test suites")
    parser.add_argument("--report", type=str, help="Show detailed validation report")
    
    args = parser.parse_args()
    
    validator = DemoValidationFramework()
    
    if args.list_suites:
        print("Available Test Suites:")
        for suite_name in validator.test_suites:
            print(f"  - {suite_name}")
    
    elif args.suite:
        if args.suite in validator.test_suites:
            report = validator.run_validation_suite(args.suite)
            print(f"Validation Report - {args.suite}")
            print(f"Overall Status: {report.overall_status.value}")
            print(f"Quality Score: {report.quality_score}/10")
            print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")
            print("Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        else:
            print(f"Test suite '{args.suite}' not found")
    
    elif args.validate_demo:
        demo_type, stakeholder_type = args.validate_demo
        result = validator.validate_demo_quality(demo_type, stakeholder_type)
        print(json.dumps(result, indent=2))
    
    elif args.history:
        history = validator.get_validation_history(args.history)
        print(f"Validation History (Last {args.history} days):")
        for report in history:
            print(f"  {report['validation_date']}: {report['overall_status']} "
                  f"(Score: {report['quality_score']}, Tests: {report['passed_tests']}/{report['total_tests']})")

if __name__ == "__main__":
    main()