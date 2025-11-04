"""
Continuous integration and deploy testing with medical compliance checks.

This module provides CI/CD testing infrastructure for medical AI systems including:
- Automated testing pipeline validation
- Deployment validation with medical compliance checks
- Production readiness assessment
- Medical regulatory compliance verification
- Performance regression testing
- Security compliance validation
"""

import pytest
import json
import subprocess
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import yaml

from fastapi.testclient import TestClient

# Import serving components
from api.main import app


class CIPipelineValidator:
    """Validate CI/CD pipeline compliance for medical systems."""
    
    def __init__(self):
        self.medical_compliance_requirements = {
            "security": {
                "required_checks": [
                    "vulnerability_scanning",
                    "static_code_analysis",
                    "dependency_scanning",
                    "secrets_detection"
                ],
                "compliance_level": "strict"
            },
            "testing": {
                "required_coverage": 0.85,
                "required_tests": [
                    "unit_tests",
                    "integration_tests",
                    "security_tests",
                    "compliance_tests"
                ],
                "medical_specific_tests": [
                    "phi_protection_test",
                    "clinical_accuracy_test",
                    "medical_workflow_test"
                ]
            },
            "documentation": {
                "required_docs": [
                    "api_documentation",
                    "clinical_guidelines",
                    "security_ocumentation",
                    "compliance_matrix"
                ]
            },
            "validation": {
                "pre_deployment_checks": [
                    "model_validation",
                    "clinical_validation",
                    "performance_benchmarks",
                    "security_scan"
                ]
            }
        }
        
        self.deployment_environments = {
            "development": {
                "security_level": "standard",
                "compliance_check": False,
                "phi_allowed": False
            },
            "staging": {
                "security_level": "elevated",
                "compliance_check": True,
                "phi_allowed": False
            },
            "production": {
                "security_level": "strict",
                "compliance_check": True,
                "phi_allowed": True,
                "audit_required": True
            }
        }
    
    def validate_pipeline_config(self, pipeline_config: Dict) -> Dict[str, Any]:
        """Validate CI pipeline configuration for medical compliance."""
        
        validation_results = {
            "valid": True,
            "compliance_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Check security requirements
        security_score = self._validate_security_requirements(pipeline_config.get("security", {}))
        validation_results["security_score"] = security_score
        
        # Check testing requirements
        testing_score = self._validate_testing_requirements(pipeline_config.get("testing", {}))
        validation_results["testing_score"] = testing_score
        
        # Check documentation requirements
        documentation_score = self._validate_documentation_requirements(pipeline_config.get("documentation", {}))
        validation_results["documentation_score"] = documentation_score
        
        # Calculate overall compliance score
        scores = [security_score, testing_score, documentation_score]
        validation_results["compliance_score"] = sum(scores) / len(scores)
        
        # Determine if pipeline is valid for medical deployment
        validation_results["valid"] = (
            validation_results["compliance_score"] >= 0.80 and
            security_score >= 0.90 and
            testing_score >= 0.85
        )
        
        return validation_results
    
    def _validate_security_requirements(self, security_config: Dict) -> float:
        """Validate security configuration."""
        
        required_checks = self.medical_compliance_requirements["security"]["required_checks"]
        implemented_checks = 0
        
        for check in required_checks:
            if check in security_config:
                if security_config[check].get("enabled", False):
                    implemented_checks += 1
            else:
                # Check if implemented in standard CI tools
                if check == "vulnerability_scanning" and security_config.get("sast_enabled"):
                    implemented_checks += 1
                elif check == "dependency_scanning" and security_config.get("dependency_check_enabled"):
                    implemented_checks += 1
        
        return implemented_checks / len(required_checks)
    
    def _validate_testing_requirements(self, testing_config: Dict) -> float:
        """Validate testing configuration."""
        
        required_tests = self.medical_compliance_requirements["testing"]["required_tests"]
        implemented_tests = 0
        
        for test in required_tests:
            if test in testing_config:
                if testing_config[test].get("enabled", False):
                    implemented_tests += 1
        
        # Check for medical-specific tests
        medical_tests = self.medical_compliance_requirements["testing"]["medical_specific_tests"]
        medical_implemented = 0
        
        for test in medical_tests:
            if test in testing_config:
                if testing_config[test].get("enabled", False):
                    medical_implemented += 1
        
        # Weight medical tests more heavily
        regular_score = implemented_tests / len(required_tests)
        medical_score = medical_implemented / len(medical_tests)
        
        # Combine scores: 60% regular tests, 40% medical tests
        total_score = (regular_score * 0.6) + (medical_score * 0.4)
        
        return total_score
    
    def _validate_documentation_requirements(self, documentation_config: Dict) -> float:
        """Validate documentation configuration."""
        
        required_docs = self.medical_compliance_requirements["documentation"]["required_docs"]
        implemented_docs = 0
        
        for doc in required_docs:
            if doc in documentation_config:
                if documentation_config[doc].get("required", False):
                    implemented_docs += 1
        
        return implemented_docs / len(required_docs)


class DeploymentValidator:
    """Validate deployment configurations for medical systems."""
    
    def __init__(self):
        self.deployment_checks = {
            "environment_specific": [
                "environment_variables",
                "security_configuration",
                "monitoring_setup",
                "backup_procedures"
            ],
            "medical_compliance": [
                "phi_handling",
                "audit_logging",
                "access_control",
                "data_encryption"
            ],
            "operational": [
                "health_checks",
                "auto_scaling",
                "load_balancing",
                "disaster_recovery"
            ]
        }
    
    def validate_deployment_config(self, deployment_config: Dict, environment: str) -> Dict[str, Any]:
        """Validate deployment configuration for specific environment."""
        
        if environment not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {environment}")
        
        validation_results = {
            "environment": environment,
            "valid": True,
            "compliance_score": 0.0,
            "checks": {},
            "critical_issues": [],
            "warnings": []
        }
        
        # Run environment-specific validations
        env_score = self._validate_environment_specific(deployment_config, environment)
        validation_results["environment_score"] = env_score
        
        # Run medical compliance validations
        compliance_score = self._validate_medical_compliance(deployment_config, environment)
        validation_results["compliance_score"] = compliance_score
        
        # Run operational validations
        operational_score = self._validate_operational_requirements(deployment_config, environment)
        validation_results["operational_score"] = operational_score
        
        # Calculate overall score
        validation_results["overall_score"] = (
            env_score * 0.3 + compliance_score * 0.5 + operational_score * 0.2
        )
        
        # Determine if deployment is valid
        validation_results["valid"] = (
            validation_results["overall_score"] >= 0.80 and
            len(validation_results["critical_issues"]) == 0
        )
        
        return validation_results
    
    def _validate_environment_specific(self, config: Dict, environment: str) -> float:
        """Validate environment-specific configuration."""
        
        environment_checks = self.deployment_checks["environment_specific"]
        passed_checks = 0
        
        for check in environment_checks:
            if check in config:
                passed_checks += 1
        
        return passed_checks / len(environment_checks)
    
    def _validate_medical_compliance(self, config: Dict, environment: str) -> float:
        """Validate medical compliance configuration."""
        
        medical_checks = self.deployment_checks["medical_compliance"]
        passed_checks = 0
        
        # Production requires all medical compliance checks
        # Staging requires most checks
        # Development requires basic checks
        
        required_level = {
            "development": 0.50,  # 50% of checks
            "staging": 0.75,     # 75% of checks
            "production": 1.0     # 100% of checks
        }
        
        required_ratio = required_level[environment]
        required_checks = int(len(medical_checks) * required_ratio)
        
        for check in medical_checks:
            if check in config and config[check].get("enabled", False):
                passed_checks += 1
        
        return min(passed_checks / len(medical_checks), required_ratio)
    
    def _validate_operational_requirements(self, config: Dict, environment: str) -> float:
        """Validate operational requirements."""
        
        operational_checks = self.deployment_checks["operational"]
        passed_checks = 0
        
        for check in operational_checks:
            if check in config:
                passed_checks += 1
        
        return passed_checks / len(operational_checks)


class MedicalComplianceChecker:
    """Check medical regulatory compliance for deployments."""
    
    def __init__(self):
        self.compliance_frameworks = {
            "hipaa": {
                "required_safeguards": [
                    "administrative",
                    "physical",
                    "technical"
                ],
                "required_procedures": [
                    "access_control",
                    "audit_controls",
                    "integrity",
                    "transmission_security"
                ]
            },
            "fda": {
                "requirements": [
                    "clinical_validation",
                    "risk_management",
                    "documentation",
                    "post_market_surveillance"
                ]
            },
            "iso13485": {
                "requirements": [
                    "quality_management",
                    "risk_management",
                    "regulatory_compliance",
                    "post_market_surveillance"
                ]
            }
        }
    
    def check_compliance(self, deployment_info: Dict, framework: str) -> Dict[str, Any]:
        """Check compliance with specified framework."""
        
        if framework not in self.compliance_frameworks:
            raise ValueError(f"Unknown compliance framework: {framework}")
        
        framework_config = self.compliance_frameworks[framework]
        compliance_results = {
            "framework": framework,
            "compliant": False,
            "score": 0.0,
            "findings": [],
            "recommendations": []
        }
        
        if framework == "hipaa":
            compliance_results = self._check_hipaa_compliance(deployment_info)
        elif framework == "fda":
            compliance_results = self._check_fda_compliance(deployment_info)
        elif framework == "iso13485":
            compliance_results = self._check_iso13485_compliance(deployment_info)
        
        return compliance_results
    
    def _check_hipaa_compliance(self, deployment_info: Dict) -> Dict[str, Any]:
        """Check HIPAA compliance."""
        
        findings = []
        score = 0.0
        max_score = 4  # 4 main safeguard categories
        
        # Check administrative safeguards
        admin_safeguards = deployment_info.get("security", {}).get("administrative_safeguards", {})
        if admin_safeguards.get("security_officer", False):
            score += 1
            findings.append("Administrative safeguards: Security officer assigned")
        else:
            findings.append("Administrative safeguards: Security officer not assigned")
        
        # Check physical safeguards
        physical_safeguards = deployment_info.get("security", {}).get("physical_safeguards", {})
        if physical_safeguards.get("facility_access_controls", False):
            score += 1
            findings.append("Physical safeguards: Facility access controls implemented")
        else:
            findings.append("Physical safeguards: Facility access controls missing")
        
        # Check technical safeguards
        technical_safeguards = deployment_info.get("security", {}).get("technical_safeguards", {})
        if technical_safeguards.get("access_control", False) and technical_safeguards.get("audit_logs", False):
            score += 1
            findings.append("Technical safeguards: Access control and audit logs implemented")
        else:
            findings.append("Technical safeguards: Missing access control or audit logs")
        
        # Check policy and procedures
        policies = deployment_info.get("compliance", {}).get("policies", {})
        if policies.get("incident_response", False) and policies.get("business_associate", False):
            score += 1
            findings.append("Policies: Incident response and BAA procedures implemented")
        else:
            findings.append("Policies: Missing incident response or BAA procedures")
        
        compliance_percentage = score / max_score
        
        return {
            "framework": "hipaa",
            "compliant": compliance_percentage >= 0.75,
            "score": compliance_percentage,
            "findings": findings,
            "recommendations": self._generate_hipaa_recommendations(deployment_info)
        }
    
    def _check_fda_compliance(self, deployment_info: Dict) -> Dict[str, Any]:
        """Check FDA compliance for medical software."""
        
        findings = []
        score = 0.0
        max_score = 4
        
        # Clinical validation
        if deployment_info.get("validation", {}).get("clinical_validation", False):
            score += 1
            findings.append("Clinical validation: Completed")
        else:
            findings.append("Clinical validation: Not completed")
        
        # Risk management
        if deployment_info.get("validation", {}).get("risk_management", False):
            score += 1
            findings.append("Risk management: Implemented")
        else:
            findings.append("Risk management: Not implemented")
        
        # Documentation
        if deployment_info.get("documentation", {}).get("technical_file", False):
            score += 1
            findings.append("Documentation: Technical file complete")
        else:
            findings.append("Documentation: Technical file incomplete")
        
        # Post-market surveillance
        if deployment_info.get("monitoring", {}).get("post_market_surveillance", False):
            score += 1
            findings.append("Post-market surveillance: Implemented")
        else:
            findings.append("Post-market surveillance: Not implemented")
        
        compliance_percentage = score / max_score
        
        return {
            "framework": "fda",
            "compliant": compliance_percentage >= 0.75,
            "score": compliance_percentage,
            "findings": findings,
            "recommendations": self._generate_fda_recommendations(deployment_info)
        }
    
    def _check_iso13485_compliance(self, deployment_info: Dict) -> Dict[str, Any]:
        """Check ISO 13485 compliance."""
        
        findings = []
        score = 0.0
        max_score = 4
        
        # Quality management system
        if deployment_info.get("quality", {}).get("qms_implemented", False):
            score += 1
            findings.append("QMS: Implemented")
        else:
            findings.append("QMS: Not implemented")
        
        # Risk management
        if deployment_info.get("quality", {}).get("risk_management", False):
            score += 1
            findings.append("Risk management: Implemented")
        else:
            findings.append("Risk management: Not implemented")
        
        # Regulatory compliance
        if deployment_info.get("compliance", {}).get("regulatory_oversight", False):
            score += 1
            findings.append("Regulatory compliance: Oversight implemented")
        else:
            findings.append("Regulatory compliance: Missing oversight")
        
        # Post-market surveillance
        if deployment_info.get("monitoring", {}).get("post_market_surveillance", False):
            score += 1
            findings.append("Post-market surveillance: Implemented")
        else:
            findings.append("Post-market surveillance: Not implemented")
        
        compliance_percentage = score / max_score
        
        return {
            "framework": "iso13485",
            "compliant": compliance_percentage >= 0.75,
            "score": compliance_percentage,
            "findings": findings,
            "recommendations": self._generate_iso13485_recommendations(deployment_info)
        }
    
    def _generate_hipaa_recommendations(self, deployment_info: Dict) -> List[str]:
        """Generate HIPAA-specific recommendations."""
        recommendations = []
        
        if not deployment_info.get("security", {}).get("administrative_safeguards", {}).get("security_officer"):
            recommendations.append("Assign a HIPAA Security Officer to oversee compliance")
        
        if not deployment_info.get("security", {}).get("technical_safeguards", {}).get("audit_logs"):
            recommendations.append("Implement comprehensive audit logging for all PHI access")
        
        if not deployment_info.get("compliance", {}).get("policies", {}).get("incident_response"):
            recommendations.append("Develop and implement incident response procedures")
        
        return recommendations
    
    def _generate_fda_recommendations(self, deployment_info: Dict) -> List[str]:
        """Generate FDA-specific recommendations."""
        recommendations = []
        
        if not deployment_info.get("validation", {}).get("clinical_validation"):
            recommendations.append("Conduct clinical validation studies for the AI system")
        
        if not deployment_info.get("validation", {}).get("risk_management"):
            recommendations.append("Implement comprehensive risk management procedures")
        
        return recommendations
    
    def _generate_iso13485_recommendations(self, deployment_info: Dict) -> List[str]:
        """Generate ISO 13485-specific recommendations."""
        recommendations = []
        
        if not deployment_info.get("quality", {}).get("qms_implemented"):
            recommendations.append("Implement a Quality Management System (QMS)")
        
        return recommendations


class TestCIPipelineValidation:
    """Test CI pipeline validation for medical systems."""
    
    @pytest.fixture
    def pipeline_validator(self):
        """Create pipeline validator."""
        return CIPipelineValidator()
    
    @pytest.mark.compliance
    @pytest.mark.ci
    def test_pipeline_security_requirements(self, pipeline_validator):
        """Test CI pipeline meets security requirements."""
        
        # Mock CI pipeline configuration
        pipeline_config = {
            "security": {
                "vulnerability_scanning": {"enabled": True, "threshold": "high"},
                "static_code_analysis": {"enabled": True},
                "dependency_scanning": {"enabled": True},
                "secrets_detection": {"enabled": True}
            },
            "testing": {
                "unit_tests": {"enabled": True, "coverage_threshold": 85},
                "integration_tests": {"enabled": True},
                "security_tests": {"enabled": True},
                "compliance_tests": {"enabled": True},
                "phi_protection_test": {"enabled": True},
                "clinical_accuracy_test": {"enabled": True},
                "medical_workflow_test": {"enabled": True}
            },
            "documentation": {
                "api_documentation": {"required": True},
                "clinical_guidelines": {"required": True},
                "security_documentation": {"required": True},
                "compliance_matrix": {"required": True}
            }
        }
        
        validation_result = pipeline_validator.validate_pipeline_config(pipeline_config)
        
        assert validation_result["valid"], "Pipeline does not meet medical compliance requirements"
        assert validation_result["security_score"] >= 0.90, "Security score below threshold"
        assert validation_result["testing_score"] >= 0.85, "Testing score below threshold"
        
        print(f"Pipeline Compliance Score: {validation_result['compliance_score']:.3f}")
    
    @pytest.mark.compliance
    @pytest.mark.ci
    def test_medical_specific_testing(self, pipeline_validator):
        """Test medical-specific testing requirements in CI."""
        
        # Test with missing medical-specific tests
        incomplete_config = {
            "security": {
                "vulnerability_scanning": {"enabled": True},
                "static_code_analysis": {"enabled": True}
            },
            "testing": {
                "unit_tests": {"enabled": True},
                "integration_tests": {"enabled": True},
                # Missing medical-specific tests
            },
            "documentation": {}
        }
        
        validation_result = pipeline_validator.validate_pipeline_config(incomplete_config)
        
        assert not validation_result["valid"], "Pipeline should fail with missing medical tests"
        assert validation_result["testing_score"] < 0.85, "Testing score should be low"
    
    @pytest.mark.compliance
    @pytest.mark.ci
    def test_production_readiness_checklist(self, pipeline_validator):
        """Test production readiness checklist."""
        
        production_readiness_config = {
            "validation": {
                "model_validation": {"enabled": True},
                "clinical_validation": {"enabled": True},
                "performance_benchmarks": {"enabled": True},
                "security_scan": {"enabled": True}
            },
            "compliance": {
                "hipaa_compliance": {"enabled": True},
                "fda_compliance": {"enabled": True},
                "audit_trail": {"enabled": True}
            },
            "monitoring": {
                "clinical_monitoring": {"enabled": True},
                "security_monitoring": {"enabled": True},
                "performance_monitoring": {"enabled": True}
            },
            "documentation": {
                "clinical_guidelines": {"required": True},
                "user_manual": {"required": True},
                "compliance_certificate": {"required": True}
            }
        }
        
        validation_result = pipeline_validator.validate_pipeline_config(production_readiness_config)
        
        # Production pipeline should have high scores
        assert validation_result["security_score"] >= 0.90
        assert validation_result["testing_score"] >= 0.90
        assert validation_result["documentation_score"] >= 0.75
        assert validation_result["valid"] is True


class TestDeploymentValidation:
    """Test deployment validation for medical systems."""
    
    @pytest.fixture
    def deployment_validator(self):
        """Create deployment validator."""
        return DeploymentValidator()
    
    @pytest.mark.compliance
    @pytest.mark.deployment
    def test_production_deployment_validation(self, deployment_validator):
        """Test production deployment validation."""
        
        production_config = {
            "environment": "production",
            "security": {
                "phi_handling": {"enabled": True},
                "audit_logging": {"enabled": True},
                "access_control": {"enabled": True, "method": "rbac"},
                "data_encryption": {"enabled": True, "algorithm": "AES-256"}
            },
            "monitoring": {
                "health_checks": {"enabled": True},
                "clinical_monitoring": {"enabled": True},
                "security_alerts": {"enabled": True}
            },
            "backup": {
                "automated_backups": {"enabled": True},
                "disaster_recovery": {"enabled": True}
            },
            "scaling": {
                "auto_scaling": {"enabled": True},
                "load_balancing": {"enabled": True}
            }
        }
        
        validation_result = deployment_validator.validate_deployment_config(
            production_config, "production"
        )
        
        assert validation_result["valid"], "Production deployment not ready"
        assert validation_result["compliance_score"] >= 0.80
        assert len(validation_result["critical_issues"]) == 0
        
        print(f"Production Deployment Score: {validation_result['overall_score']:.3f}")
    
    @pytest.mark.compliance
    @pytest.mark.deployment
    def test_staging_deployment_validation(self, deployment_validator):
        """Test staging deployment validation."""
        
        staging_config = {
            "environment": "staging",
            "security": {
                "phi_handling": {"enabled": False},  # No PHI in staging
                "audit_logging": {"enabled": True},
                "access_control": {"enabled": True}
            },
            "monitoring": {
                "health_checks": {"enabled": True},
                "security_alerts": {"enabled": True}
            },
            "backup": {
                "automated_backups": {"enabled": True}
            }
        }
        
        validation_result = deployment_validator.validate_deployment_config(
            staging_config, "staging"
        )
        
        assert validation_result["valid"], "Staging deployment not ready"
        assert validation_result["overall_score"] >= 0.75
    
    @pytest.mark.compliance
    @pytest.mark.deployment
    def test_deployment_environment_differences(self, deployment_validator):
        """Test that different environments have appropriate security levels."""
        
        base_config = {
            "security": {
                "audit_logging": {"enabled": True},
                "access_control": {"enabled": True}
            },
            "monitoring": {
                "health_checks": {"enabled": True}
            }
        }
        
        dev_config = base_config.copy()
        dev_config["security"]["phi_handling"] = {"enabled": False}
        
        prod_config = base_config.copy()
        prod_config["security"]["phi_handling"] = {"enabled": True}
        prod_config["security"]["data_encryption"] = {"enabled": True}
        prod_config["monitoring"]["security_alerts"] = {"enabled": True}
        
        dev_validation = deployment_validator.validate_deployment_config(dev_config, "development")
        prod_validation = deployment_validator.validate_deployment_config(prod_config, "production")
        
        # Production should have higher compliance requirements
        assert prod_validation["compliance_score"] > dev_validation["compliance_score"]
        assert prod_validation["overall_score"] > dev_validation["overall_score"]


class TestMedicalCompliance:
    """Test medical regulatory compliance validation."""
    
    @pytest.fixture
    def compliance_checker(self):
        """Create compliance checker."""
        return MedicalComplianceChecker()
    
    @pytest.mark.compliance
    @pytest.mark.regulation
    def test_hipaa_compliance_check(self, compliance_checker):
        """Test HIPAA compliance validation."""
        
        hipaa_compliant_deployment = {
            "security": {
                "administrative_safeguards": {
                    "security_officer": True,
                    "workforce_training": True,
                    "information_access_management": True
                },
                "physical_safeguards": {
                    "facility_access_controls": True,
                    "workstation_use": True,
                    "device_and_media_controls": True
                },
                "technical_safeguards": {
                    "access_control": True,
                    "audit_logs": True,
                    "integrity": True,
                    "transmission_security": True
                }
            },
            "compliance": {
                "policies": {
                    "incident_response": True,
                    "business_associate": True,
                    "sanctions": True
                }
            }
        }
        
        compliance_result = compliance_checker.check_compliance(
            hipaa_compliant_deployment, "hipaa"
        )
        
        assert compliance_result["framework"] == "hipaa"
        assert compliance_result["compliant"] is True
        assert compliance_result["score"] >= 0.75
        
        print(f"HIPAA Compliance Score: {compliance_result['score']:.3f}")
        for finding in compliance_result["findings"]:
            print(f"  {finding}")
    
    @pytest.mark.compliance
    @pytest.mark.regulation
    def test_fda_compliance_check(self, compliance_checker):
        """Test FDA compliance for medical AI."""
        
        fda_compliant_deployment = {
            "validation": {
                "clinical_validation": {"completed": True, "studies": 3},
                "risk_management": {"implemented": True, "iso_14971": True},
                "software_lifecycle": {"controlled": True, "documentation": True}
            },
            "documentation": {
                "technical_file": {"complete": True, "revised": True},
                "clinical_evaluation": {"available": True},
                "risk_management_file": {"maintained": True}
            },
            "monitoring": {
                "post_market_surveillance": {"implemented": True},
                "adverse_event_reporting": {"enabled": True}
            }
        }
        
        compliance_result = compliance_checker.check_compliance(
            fda_compliant_deployment, "fda"
        )
        
        assert compliance_result["framework"] == "fda"
        assert compliance_result["compliant"] is True
        assert compliance_result["score"] >= 0.75
    
    @pytest.mark.compliance
    @pytest.mark.regulation
    def test_iso13485_compliance_check(self, compliance_checker):
        """Test ISO 13485 quality management compliance."""
        
        iso_compliant_deployment = {
            "quality": {
                "qms_implemented": True,
                "risk_management": {"iso_14971": True},
                "regulatory_requirements": {"tracked": True}
            },
            "compliance": {
                "regulatory_oversight": {"implemented": True},
                "quality_objectives": {"defined": True}
            },
            "monitoring": {
                "post_market_surveillance": {"systematic": True},
                "customer_feedback": {"monitored": True}
            }
        }
        
        compliance_result = compliance_checker.check_compliance(
            iso_compliant_deployment, "iso13485"
        )
        
        assert compliance_result["framework"] == "iso13485"
        assert compliance_result["compliant"] is True
        assert compliance_result["score"] >= 0.75
    
    @pytest.mark.compliance
    @pytest.mark.regulation
    def test_compliance_recommendations(self, compliance_checker):
        """Test compliance recommendations generation."""
        
        non_compliant_deployment = {
            "security": {
                "administrative_safeguards": {"security_officer": False},
                "technical_safeguards": {"audit_logs": False}
            },
            "compliance": {
                "policies": {"incident_response": False}
            }
        }
        
        compliance_result = compliance_checker.check_compliance(
            non_compliant_deployment, "hipaa"
        )
        
        assert compliance_result["compliant"] is False
        assert len(compliance_result["recommendations"]) > 0
        
        # Should have specific recommendations
        recommendations = compliance_result["recommendations"]
        assert any("Security Officer" in rec for rec in recommendations)
        assert any("audit logging" in rec.lower() for rec in recommendations)


class TestPreDeploymentChecks:
    """Test pre-deployment validation checks."""
    
    @pytest.mark.compliance
    @pytest.mark.deployment
    def test_model_validation_requirements(self, client):
        """Test model validation before deployment."""
        
        validation_checks = [
            "model_accuracy_validation",
            "clinical_validation",
            "bias_detection",
            "performance_benchmarks",
            "security_scan"
        ]
        
        for check in validation_checks:
            response = client.post("/api/v1/validation/pre-deployment-check", json={
                "check_type": check,
                "model_version": "v1.0",
                "environment": "production"
            })
            
            assert response.status_code == 200
            result = response.json()
            
            assert "check_passed" in result
            assert "validation_details" in result
            assert "timestamp" in result
    
    @pytest.mark.compliance
    @pytest.mark.deployment
    def test_performance_regression_detection(self, client):
        """Test performance regression detection."""
        
        performance_baseline = {
            "response_time_ms": 1500,
            "throughput_rps": 50,
            "accuracy_score": 0.85,
            "error_rate": 0.02
        }
        
        current_metrics = {
            "response_time_ms": 1600,
            "throughput_rps": 48,
            "accuracy_score": 0.84,
            "error_rate": 0.03
        }
        
        response = client.post("/api/v1/validation/performance-check", json={
            "baseline": performance_baseline,
            "current": current_metrics,
            "tolerance": {
                "response_time_degradation": 0.10,  # 10% tolerance
                "throughput_degradation": 0.05,     # 5% tolerance
                "accuracy_degradation": 0.02,       # 2% tolerance
                "error_rate_increase": 0.01         # 1% tolerance
            }
        })
        
        assert response.status_code == 200
        result = response.json()
        
        assert "regression_detected" in result
        assert "performance_score" in result
        assert "deployment_recommended" in result
        
        # Current metrics should be within acceptable ranges
        assert result["deployment_recommended"] is True
    
    @pytest.mark.compliance
    @pytest.mark.deployment
    def test_security_compliance_verification(self, client):
        """Test security compliance verification."""
        
        response = client.post("/api/v1/validation/security-check", json={
            "deployment_type": "production",
            "compliance_frameworks": ["hipaa", "soc2", "iso27001"],
            "security_scan_required": True
        })
        
        assert response.status_code == 200
        result = response.json()
        
        assert "security_score" in result
        assert "compliance_frameworks" in result
        assert "deployment_approved" in result
        
        # Security score should be high for production
        assert result["security_score"] >= 0.80
        assert result["deployment_approved"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "compliance"])