"""
Production Data Quality Monitoring for Healthcare Data
Implements comprehensive data quality checks with medical validation rules
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import defaultdict
import re
import statistics

class QualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"          # 85-94%
    FAIR = "fair"          # 75-84%
    POOR = "poor"          # Below 75%

class ValidationRule(Enum):
    """Types of data validation rules"""
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    MEDICAL_LOGIC = "medical_logic"

@dataclass
class QualityCheck:
    """Individual data quality check result"""
    check_id: str
    table_name: str
    column_name: str
    rule_type: ValidationRule
    check_name: str
    threshold: float
    actual_score: float
    status: QualityLevel
    failed_records: int
    total_records: int
    error_details: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    report_id: str
    generated_at: datetime
    table_name: str
    overall_score: float
    quality_level: QualityLevel
    total_records: int
    checks_performed: List[QualityCheck]
    critical_issues: List[str]
    warnings: List[str]
    improvements: List[str]

class MedicalDataQualityMonitor:
    """Production data quality monitoring system for healthcare data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.validation_rules = self._load_validation_rules()
        self.quality_history = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup quality monitoring logging"""
        logger = logging.getLogger("data_quality")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_validation_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load healthcare-specific validation rules"""
        return {
            "patients": [
                {
                    "rule": "patient_id_format",
                    "type": ValidationRule.VALIDITY,
                    "pattern": r"^[A-Z0-9]{8,12}$",
                    "threshold": 0.99,
                    "description": "Patient ID must be 8-12 alphanumeric characters"
                },
                {
                    "rule": "birth_date_validity",
                    "type": ValidationRule.VALIDITY,
                    "min_date": "1900-01-01",
                    "max_date": datetime.now().date(),
                    "threshold": 1.0,
                    "description": "Birth date must be valid and not in the future"
                },
                {
                    "rule": "age_consistency",
                    "type": ValidationRule.CONSISTENCY,
                    "threshold": 0.95,
                    "description": "Age must be consistent with birth date"
                },
                {
                    "rule": "required_fields_complete",
                    "type": ValidationRule.COMPLETENESS,
                    "required_fields": ["patient_id", "birth_date", "gender"],
                    "threshold": 0.98,
                    "description": "Required patient fields must be complete"
                }
            ],
            "encounters": [
                {
                    "rule": "encounter_id_format",
                    "type": ValidationRule.VALIDITY,
                    "pattern": r"^[A-Z0-9]{10,15}$",
                    "threshold": 0.99,
                    "description": "Encounter ID must be valid format"
                },
                {
                    "rule": "admission_date_logic",
                    "type": ValidationRule.MEDICAL_LOGIC,
                    "threshold": 1.0,
                    "description": "Admission date must be before discharge date"
                },
                {
                    "rule": "length_of_stay_valid",
                    "type": ValidationRule.VALIDITY,
                    "min_value": 0,
                    "max_value": 365,
                    "threshold": 0.95,
                    "description": "Length of stay must be reasonable (0-365 days)"
                },
                {
                    "rule": "diagnosis_codes_valid",
                    "type": ValidationRule.VALIDITY,
                    "pattern": r"^[A-Z][0-9]{2}(\.[0-9]{1,2})?$",
                    "threshold": 0.98,
                    "description": "Diagnosis codes must be valid ICD-10 format"
                }
            ],
            "medications": [
                {
                    "rule": "medication_code_format",
                    "type": ValidationRule.VALIDITY,
                    "pattern": r"^[A-Z0-9]{5,10}$",
                    "threshold": 0.98,
                    "description": "Medication codes must be valid format"
                },
                {
                    "rule": "dosage_validity",
                    "type": ValidationRule.VALIDITY,
                    "min_value": 0,
                    "max_value": 1000,
                    "threshold": 0.95,
                    "description": "Medication dosages must be within reasonable ranges"
                },
                {
                    "rule": "date_logic",
                    "type": ValidationRule.CONSISTENCY,
                    "threshold": 0.99,
                    "description": "Medication dates must be logical"
                }
            ],
            "lab_results": [
                {
                    "rule": "test_codes_valid",
                    "type": ValidationRule.VALIDITY,
                    "pattern": r"^[A-Z0-9]{3,8}$",
                    "threshold": 0.99,
                    "description": "Lab test codes must be valid format"
                },
                {
                    "rule": "reference_range_logic",
                    "type": ValidationRule.MEDICAL_LOGIC,
                    "threshold": 0.95,
                    "description": "Lab values must be within reasonable ranges"
                },
                {
                    "rule": "unit_consistency",
                    "type": ValidationRule.CONSISTENCY,
                    "threshold": 0.98,
                    "description": "Lab units must be consistent with test codes"
                }
            ],
            "vital_signs": [
                {
                    "rule": "vital_ranges",
                    "type": ValidationRule.MEDICAL_LOGIC,
                    "vital_ranges": {
                        "blood_pressure_systolic": (70, 250),
                        "blood_pressure_diastolic": (40, 150),
                        "heart_rate": (30, 200),
                        "temperature": (35.0, 42.0),
                        "oxygen_saturation": (70, 100),
                        "respiratory_rate": (8, 60)
                    },
                    "threshold": 0.95,
                    "description": "Vital signs must be within human ranges"
                }
            ]
        }
    
    async def perform_quality_check(self, table_name: str, df: pd.DataFrame, 
                                   check_config: Optional[Dict[str, Any]] = None) -> QualityReport:
        """Perform comprehensive data quality assessment"""
        try:
            report_id = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get validation rules for table
            rules = self.validation_rules.get(table_name, [])
            if check_config and check_config.get("custom_rules"):
                rules.extend(check_config["custom_rules"])
            
            # Perform all checks
            quality_checks = []
            for rule in rules:
                check = await self._execute_validation_rule(df, rule, table_name)
                quality_checks.append(check)
            
            # Calculate overall scores
            overall_score = self._calculate_overall_score(quality_checks)
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate report
            report = QualityReport(
                report_id=report_id,
                generated_at=datetime.now(),
                table_name=table_name,
                overall_score=overall_score,
                quality_level=quality_level,
                total_records=len(df),
                checks_performed=quality_checks,
                critical_issues=self._identify_critical_issues(quality_checks),
                warnings=self._identify_warnings(quality_checks),
                improvements=self._suggest_improvements(quality_checks)
            )
            
            # Log results
            self._log_quality_report(report)
            
            # Store in history
            self.quality_history.append(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Quality check failed for {table_name}: {str(e)}")
            raise
    
    async def _execute_validation_rule(self, df: pd.DataFrame, rule: Dict[str, Any], 
                                     table_name: str) -> QualityCheck:
        """Execute a specific validation rule"""
        check_id = f"{table_name}_{rule['rule']}"
        
        try:
            rule_type = rule["type"]
            
            if rule_type == ValidationRule.COMPLETENESS:
                return await self._check_completeness(df, rule)
            
            elif rule_type == ValidationRule.VALIDITY:
                return await self._check_validity(df, rule)
            
            elif rule_type == ValidationRule.CONSISTENCY:
                return await self._check_consistency(df, rule)
            
            elif rule_type == ValidationRule.MEDICAL_LOGIC:
                return await self._check_medical_logic(df, rule)
            
            elif rule_type == ValidationRule.ACCURACY:
                return await self._check_accuracy(df, rule)
            
            elif rule_type == ValidationRule.TIMELINESS:
                return await self._check_timeliness(df, rule)
            
            else:
                # Default validity check
                return QualityCheck(
                    check_id=check_id,
                    table_name=table_name,
                    column_name="multiple",
                    rule_type=rule_type,
                    check_name=rule["rule"],
                    threshold=rule["threshold"],
                    actual_score=0.0,
                    status=QualityLevel.POOR,
                    failed_records=len(df),
                    total_records=len(df),
                    error_details=[f"Unknown validation rule type: {rule_type}"]
                )
                
        except Exception as e:
            return QualityCheck(
                check_id=check_id,
                table_name=table_name,
                column_name="multiple",
                rule_type=rule_type,
                check_name=rule["rule"],
                threshold=rule["threshold"],
                actual_score=0.0,
                status=QualityLevel.POOR,
                failed_records=len(df),
                total_records=len(df),
                error_details=[f"Validation rule execution failed: {str(e)}"]
            )
    
    async def _check_completeness(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityCheck:
        """Check data completeness"""
        required_fields = rule.get("required_fields", [])
        threshold = rule["threshold"]
        check_name = rule["rule"]
        
        total_records = len(df)
        failed_records = 0
        error_details = []
        
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                null_percentage = null_count / total_records * 100
                
                if null_percentage > (1 - threshold) * 100:
                    failed_records += null_count
                    error_details.append(f"Field '{field}' has {null_percentage:.1f}% null values")
            else:
                error_details.append(f"Required field '{field}' not found in dataset")
        
        actual_score = 1.0 - (failed_records / (total_records * len(required_fields))) if required_fields else 1.0
        status = self._determine_quality_level(actual_score * 100)
        
        return QualityCheck(
            check_id=f"{check_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            table_name="",
            column_name=", ".join(required_fields),
            rule_type=ValidationRule.COMPLETENESS,
            check_name=check_name,
            threshold=threshold,
            actual_score=actual_score * 100,
            status=status,
            failed_records=failed_records,
            total_records=total_records,
            error_details=error_details
        )
    
    async def _check_validity(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityCheck:
        """Check data validity"""
        threshold = rule["threshold"]
        check_name = rule["rule"]
        pattern = rule.get("pattern")
        
        total_records = len(df)
        failed_records = 0
        error_details = []
        valid_records = 0
        
        if pattern:
            # Regex pattern validation
            pattern_regex = re.compile(pattern)
            
            for column in df.columns:
                if df[column].dtype == 'object':
                    valid_count = df[column].str.match(pattern_regex, na=False).sum()
                    invalid_count = df[column].notna().sum() - valid_count
                    
                    if invalid_count > 0:
                        error_details.append(f"Column '{column}': {invalid_count} invalid formats")
                        failed_records += invalid_count
                    
                    valid_records += valid_count
            
            actual_score = valid_records / total_records if total_records > 0 else 0.0
        else:
            # Value range validation
            min_val = rule.get("min_value")
            max_val = rule.get("max_value")
            min_date = rule.get("min_date")
            max_date = rule.get("max_date")
            
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    if min_val is not None and max_val is not None:
                        valid_count = ((df[column] >= min_val) & (df[column] <= max_val)).sum()
                        invalid_count = df[column].notna().sum() - valid_count
                        
                        if invalid_count > 0:
                            error_details.append(f"Column '{column}': {invalid_count} values outside range [{min_val}, {max_val}]")
                            failed_records += invalid_count
                
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    if min_date and max_date:
                        valid_count = ((pd.to_datetime(df[column]) >= min_date) & 
                                     (pd.to_datetime(df[column]) <= max_date)).sum()
                        invalid_count = df[column].notna().sum() - valid_count
                        
                        if invalid_count > 0:
                            error_details.append(f"Column '{column}': {invalid_count} dates outside valid range")
                            failed_records += invalid_count
            
            actual_score = (total_records - failed_records) / total_records if total_records > 0 else 0.0
        
        status = self._determine_quality_level(actual_score * 100)
        
        return QualityCheck(
            check_id=f"{check_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            table_name="",
            column_name="",
            rule_type=ValidationRule.VALIDITY,
            check_name=check_name,
            threshold=threshold * 100,
            actual_score=actual_score * 100,
            status=status,
            failed_records=failed_records,
            total_records=total_records,
            error_details=error_details
        )
    
    async def _check_consistency(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityCheck:
        """Check data consistency"""
        threshold = rule["threshold"]
        check_name = rule["rule"]
        
        total_records = len(df)
        consistency_issues = 0
        error_details = []
        
        # Age consistency check
        if "birth_date" in df.columns and "age" in df.columns:
            try:
                calculated_ages = (datetime.now().date() - pd.to_datetime(df["birth_date"]).dt.date).dt.days // 365
                age_differences = abs(df["age"] - calculated_ages.fillna(0))
                inconsistent_ages = (age_differences > 1).sum()  # Allow 1 year difference
                
                if inconsistent_ages > 0:
                    error_details.append(f"{inconsistent_ages} records have inconsistent ages")
                    consistency_issues += inconsistent_ages
            except Exception as e:
                error_details.append(f"Age consistency check failed: {str(e)}")
        
        # Date logic consistency
        if "admission_date" in df.columns and "discharge_date" in df.columns:
            try:
                admission_dates = pd.to_datetime(df["admission_date"])
                discharge_dates = pd.to_datetime(df["discharge_date"])
                
                # Check if discharge is after admission
                invalid_dates = (discharge_dates < admission_dates).sum()
                if invalid_dates > 0:
                    error_details.append(f"{invalid_dates} records have discharge dates before admission")
                    consistency_issues += invalid_dates
            except Exception as e:
                error_details.append(f"Date consistency check failed: {str(e)}")
        
        actual_score = (total_records - consistency_issues) / total_records if total_records > 0 else 0.0
        status = self._determine_quality_level(actual_score * 100)
        
        return QualityCheck(
            check_id=f"{check_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            table_name="",
            column_name="",
            rule_type=ValidationRule.CONSISTENCY,
            check_name=check_name,
            threshold=threshold * 100,
            actual_score=actual_score * 100,
            status=status,
            failed_records=consistency_issues,
            total_records=total_records,
            error_details=error_details
        )
    
    async def _check_medical_logic(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityCheck:
        """Check medical logic validation"""
        threshold = rule["threshold"]
        check_name = rule["rule"]
        vital_ranges = rule.get("vital_ranges", {})
        
        total_records = len(df)
        logic_violations = 0
        error_details = []
        
        # Vital signs range validation
        for vital_sign, (min_val, max_val) in vital_ranges.items():
            if vital_sign in df.columns:
                values = pd.to_numeric(df[vital_sign], errors='coerce')
                out_of_range = ((values < min_val) | (values > max_val)).sum()
                
                if out_of_range > 0:
                    error_details.append(f"{out_of_range} {vital_sign} values outside normal range [{min_val}, {max_val}]")
                    logic_violations += out_of_range
        
        actual_score = (total_records - logic_violations) / total_records if total_records > 0 else 0.0
        status = self._determine_quality_level(actual_score * 100)
        
        return QualityCheck(
            check_id=f"{check_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            table_name="",
            column_name=", ".join(vital_ranges.keys()),
            rule_type=ValidationRule.MEDICAL_LOGIC,
            check_name=check_name,
            threshold=threshold * 100,
            actual_score=actual_score * 100,
            status=status,
            failed_records=logic_violations,
            total_records=total_records,
            error_details=error_details
        )
    
    async def _check_accuracy(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityCheck:
        """Check data accuracy (placeholder for more sophisticated checks)"""
        threshold = rule["threshold"]
        check_name = rule["rule"]
        
        # Basic accuracy checks (can be extended with reference data)
        total_records = len(df)
        accuracy_score = 0.98  # Default high accuracy
        
        status = self._determine_quality_level(accuracy_score * 100)
        
        return QualityCheck(
            check_id=f"{check_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            table_name="",
            column_name="",
            rule_type=ValidationRule.ACCURACY,
            check_name=check_name,
            threshold=threshold * 100,
            actual_score=accuracy_score * 100,
            status=status,
            failed_records=int(total_records * (1 - accuracy_score)),
            total_records=total_records,
            error_details=[]
        )
    
    async def _check_timeliness(self, df: pd.DataFrame, rule: Dict[str, Any]) -> QualityCheck:
        """Check data timeliness"""
        threshold = rule["threshold"]
        check_name = rule["rule"]
        
        total_records = len(df)
        
        # Check for stale data based on timestamp columns
        timestamp_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                timestamp_columns.append(col)
        
        timeliness_score = 1.0
        error_details = []
        
        for col in timestamp_columns:
            try:
                timestamps = pd.to_datetime(df[col], errors='coerce')
                max_age = datetime.now() - timedelta(days=30)  # Example: 30 days
                stale_records = (timestamps < max_age).sum()
                
                if stale_records > total_records * 0.1:  # More than 10% stale
                    timeliness_score -= 0.2
                    error_details.append(f"{stale_records} records in {col} are more than 30 days old")
            except Exception:
                continue
        
        status = self._determine_quality_level(timeliness_score * 100)
        
        return QualityCheck(
            check_id=f"{check_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            table_name="",
            column_name=", ".join(timestamp_columns),
            rule_type=ValidationRule.TIMELINESS,
            check_name=check_name,
            threshold=threshold * 100,
            actual_score=timeliness_score * 100,
            status=status,
            failed_records=int(total_records * (1 - timeliness_score)),
            total_records=total_records,
            error_details=error_details
        )
    
    def _calculate_overall_score(self, checks: List[QualityCheck]) -> float:
        """Calculate overall data quality score"""
        if not checks:
            return 0.0
        
        # Weight different types of checks
        weights = {
            ValidationRule.COMPLETENESS: 0.25,
            ValidationRule.VALIDITY: 0.25,
            ValidationRule.CONSISTENCY: 0.20,
            ValidationRule.MEDICAL_LOGIC: 0.15,
            ValidationRule.ACCURACY: 0.10,
            ValidationRule.TIMELINESS: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for check in checks:
            weight = weights.get(check.rule_type, 0.1)
            weighted_sum += check.actual_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score"""
        if score >= 95:
            return QualityLevel.EXCELLENT
        elif score >= 85:
            return QualityLevel.GOOD
        elif score >= 75:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def _identify_critical_issues(self, checks: List[QualityCheck]) -> List[str]:
        """Identify critical quality issues"""
        critical_issues = []
        
        for check in checks:
            if check.actual_score < check.threshold * 0.8:  # More than 20% below threshold
                critical_issues.append(f"{check.check_name}: Score {check.actual_score:.1f}% below threshold {check.threshold:.1f}%")
        
        return critical_issues
    
    def _identify_warnings(self, checks: List[QualityCheck]) -> List[str]:
        """Identify quality warnings"""
        warnings = []
        
        for check in checks:
            if check.actual_score < check.threshold and check.actual_score >= check.threshold * 0.8:
                warnings.append(f"{check.check_name}: Score {check.actual_score:.1f}% slightly below threshold {check.threshold:.1f}%")
        
        return warnings
    
    def _suggest_improvements(self, checks: List[QualityCheck]) -> List[str]:
        """Suggest improvements based on quality checks"""
        improvements = []
        
        for check in checks:
            if check.actual_score < check.threshold:
                if check.rule_type == ValidationRule.COMPLETENESS:
                    improvements.append(f"Implement validation to ensure required fields are filled for {check.column_name}")
                elif check.rule_type == ValidationRule.VALIDITY:
                    improvements.append(f"Add format validation for {check.column_name} to improve data validity")
                elif check.rule_type == ValidationRule.CONSISTENCY:
                    improvements.append(f"Implement cross-field validation for {check.column_name}")
                elif check.rule_type == ValidationRule.MEDICAL_LOGIC:
                    improvements.append(f"Add clinical validation rules for {check.column_name}")
        
        return improvements
    
    def _log_quality_report(self, report: QualityReport) -> None:
        """Log quality report summary"""
        self.logger.info(f"Quality Report {report.report_id}")
        self.logger.info(f"Table: {report.table_name}")
        self.logger.info(f"Overall Score: {report.overall_score:.1f}% ({report.quality_level.value})")
        self.logger.info(f"Records Analyzed: {report.total_records}")
        
        if report.critical_issues:
            self.logger.warning(f"Critical Issues: {len(report.critical_issues)}")
            for issue in report.critical_issues[:5]:  # Log first 5 issues
                self.logger.warning(f"  - {issue}")
        
        if report.warnings:
            self.logger.info(f"Warnings: {len(report.warnings)}")
    
    async def monitor_data_quality_continuously(self, monitoring_config: Dict[str, Any]) -> None:
        """Continuously monitor data quality across all tables"""
        while True:
            try:
                for table_config in monitoring_config["tables"]:
                    table_name = table_config["name"]
                    check_config = table_config.get("quality_checks", {})
                    
                    # Load data for quality check
                    df = await self._load_table_data(table_name)
                    
                    # Perform quality check
                    report = await self.perform_quality_check(table_name, df, check_config)
                    
                    # Send alerts if needed
                    await self._check_alert_conditions(report)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(monitoring_config.get("interval_minutes", 60) * 60)
                
            except Exception as e:
                self.logger.error(f"Continuous monitoring error: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _load_table_data(self, table_name: str) -> pd.DataFrame:
        """Load data from specified table (placeholder)"""
        # This would connect to actual data sources
        # For now, return sample data
        return pd.DataFrame({
            "patient_id": ["PAT001", "PAT002", "PAT003"],
            "birth_date": ["1980-01-15", "1975-05-20", "1990-12-10"],
            "age": [44, 49, 34],
            "gender": ["M", "F", "F"]
        })
    
    async def _check_alert_conditions(self, report: QualityReport) -> None:
        """Check if quality report triggers any alerts"""
        # Alert for critical quality issues
        if report.overall_score < 75 or len(report.critical_issues) > 0:
            self.logger.error(f"ALERT: Critical quality issues in {report.table_name}")
            
            # Send alert (email, webhook, etc.)
            await self._send_quality_alert(report)
    
    async def _send_quality_alert(self, report: QualityReport) -> None:
        """Send quality alert notification"""
        alert_message = {
            "table_name": report.table_name,
            "overall_score": report.overall_score,
            "quality_level": report.quality_level.value,
            "critical_issues": report.critical_issues,
            "generated_at": report.generated_at.isoformat()
        }
        
        # Log alert (in production, this would send to monitoring system)
        self.logger.error(f"Quality Alert: {json.dumps(alert_message)}")
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get quality trends over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [r for r in self.quality_history if r.generated_at >= cutoff_date]
        
        if not recent_reports:
            return {"message": "No quality data available for specified period"}
        
        # Calculate trends
        trends = {}
        for table in set(r.table_name for r in recent_reports):
            table_reports = [r for r in recent_reports if r.table_name == table]
            scores = [r.overall_score for r in sorted(table_reports, key=lambda x: x.generated_at)]
            
            trends[table] = {
                "current_score": scores[-1],
                "average_score": statistics.mean(scores),
                "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining",
                "check_count": len(table_reports)
            }
        
        return trends

def create_quality_monitor(config: Dict[str, Any] = None) -> MedicalDataQualityMonitor:
    """Factory function to create quality monitor instance"""
    if config is None:
        config = {
            "alert_thresholds": {
                "critical_score": 75,
                "warning_score": 85
            }
        }
    
    return MedicalDataQualityMonitor(config)

# Example usage
if __name__ == "__main__":
    async def main():
        monitor = create_quality_monitor()
        
        # Create sample data for testing
        sample_data = pd.DataFrame({
            "patient_id": ["PAT001", "PAT002", "", "PAT004"],
            "birth_date": ["1980-01-15", "1975-13-20", "1990-12-10", "1985-06-30"],
            "age": [44, 49, 34, 39],
            "gender": ["M", "X", "F", "M"],
            "blood_pressure_systolic": [120, 180, 90, 150],
            "blood_pressure_diastolic": [80, 120, 60, 95]
        })
        
        # Perform quality check
        report = await monitor.perform_quality_check("patients", sample_data)
        
        print(f"Overall Quality Score: {report.overall_score:.1f}%")
        print(f"Quality Level: {report.quality_level.value}")
        print(f"Critical Issues: {len(report.critical_issues)}")
        print(f"Warnings: {len(report.warnings)}")
    
    asyncio.run(main())
