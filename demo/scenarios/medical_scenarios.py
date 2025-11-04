"""
Medical Scenario Templates for Demo Environment
Provides structured templates for diabetes management, hypertension monitoring, and chest pain assessment
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class VitalSignsData:
    """Structured vital signs data"""
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    heart_rate: int
    temperature: float
    respiratory_rate: int
    oxygen_saturation: int
    glucose_level: float
    timestamp: str
    
@dataclass
class LabResult:
    """Structured lab result data"""
    test_name: str
    value: float
    unit: str
    status: str  # 'normal', 'abnormal', 'critical'
    reference_range: str
    
@dataclass
class Medication:
    """Structured medication data"""
    name: str
    dosage: str
    frequency: str
    adherence: float  # 0.0 to 1.0
    last_taken: str

@dataclass
class AISuggestion:
    """AI-generated suggestion"""
    type: str
    message: str
    confidence: float
    priority: str  # 'low', 'medium', 'high'
    action_required: bool

class DiabetesManagementScenario:
    """Diabetes Management Scenario Template"""
    
    def __init__(self, patient_id: int = 1):
        self.patient_id = patient_id
        self.scenario_id = str(uuid.uuid4())
        self.patient_name = "John Smith"
        self.patient_age = 45
        self.diabetes_type = "Type 2"
        self.diagnosis_date = "2022-03-15"
        
    def get_vital_signs_series(self) -> List[VitalSignsData]:
        """Generate realistic glucose monitoring data"""
        vitals = []
        base_glucose = 140
        
        for day in range(7):  # Last 7 days
            date = datetime.now() - timedelta(days=day)
            
            # Simulate glucose fluctuations
            morning_glucose = base_glucose + (day * 2) - 10
            pre_lunch_glucose = morning_glucose + 20
            pre_dinner_glucose = morning_glucose + 15
            bedtime_glucose = morning_glucose - 5
            
            for time_point, glucose in [
                ("07:00", morning_glucose),
                ("12:00", pre_lunch_glucose), 
                ("18:00", pre_dinner_glucose),
                ("22:00", bedtime_glucose)
            ]:
                vitals.append(VitalSignsData(
                    blood_pressure_systolic=125 + (day * 2),
                    blood_pressure_diastolic=80 + (day * 1),
                    heart_rate=72 + (day * 2),
                    temperature=36.8,
                    respiratory_rate=16,
                    oxygen_saturation=98,
                    glucose_level=max(glucose, 80),  # Minimum safe glucose
                    timestamp=f"{date.strftime('%Y-%m-%d')} {time_point}"
                ))
                
        return vitals
        
    def get_lab_results(self) -> List[LabResult]:
        """Generate recent lab results"""
        return [
            LabResult("Hemoglobin A1c", 7.8, "%", "abnormal", "4.0-7.0%"),
            LabResult("Glucose, Fasting", 145, "mg/dL", "abnormal", "70-100 mg/dL"),
            LabResult("Creatinine", 1.1, "mg/dL", "normal", "0.7-1.3 mg/dL"),
            LabResult("eGFR", 85, "mL/min/1.73m²", "normal", ">60 mL/min/1.73m²"),
            LabResult("Cholesterol, Total", 195, "mg/dL", "normal", "<200 mg/dL")
        ]
        
    def get_medications(self) -> List[Medication]:
        """Get current medications"""
        return [
            Medication("Metformin", "500mg", "twice daily", 0.85, "2024-11-03 18:00"),
            Medication("Lisinopril", "10mg", "once daily", 0.92, "2024-11-03 08:00"),
            Medication("Atorvastatin", "20mg", "once daily", 0.78, "2024-11-03 08:00")
        ]
        
    def get_ai_recommendations(self) -> List[AISuggestion]:
        """Generate AI recommendations based on data"""
        return [
            AISuggestion(
                "glucose_management",
                "Consider increasing Metformin dose based on HbA1c levels",
                0.85,
                "high",
                True
            ),
            AISuggestion(
                "lifestyle",
                "Recommend 30 minutes of moderate exercise 5 days per week",
                0.75,
                "medium",
                False
            ),
            AISuggestion(
                "monitoring",
                "Increase glucose monitoring frequency to 6 times daily",
                0.90,
                "high", 
                True
            ),
            AISuggestion(
                "dietary",
                "Consider carbohydrate counting education session",
                0.80,
                "medium",
                False
            )
        ]
        
    def get_workflow_steps(self) -> List[str]:
        """Get demo workflow steps"""
        return [
            "Review current glucose monitoring trends",
            "Analyze HbA1c progression over time",
            "Assess medication adherence patterns",
            "Evaluate dietary compliance",
            "Generate AI-powered recommendations",
            "Create personalized management plan",
            "Schedule follow-up monitoring"
        ]

class HypertensionMonitoringScenario:
    """Hypertension Monitoring Scenario Template"""
    
    def __init__(self, patient_id: int = 2):
        self.patient_id = patient_id
        self.scenario_id = str(uuid.uuid4())
        self.patient_name = "Emily Davis"
        self.patient_age = 58
        self.diagnosis_date = "2021-08-20"
        
    def get_bp_trends(self) -> List[Dict]:
        """Generate blood pressure trend data"""
        trends = []
        
        for day in range(30):  # Last 30 days
            date = datetime.now() - timedelta(days=day)
            
            # Simulate BP variations
            systolic = 135 + (day * 0.5) - 10 + (5 if day % 7 == 0 else 0)
            diastolic = 85 + (day * 0.3) - 5 + (3 if day % 7 == 0 else 0)
            
            trends.append({
                "date": date.strftime('%Y-%m-%d'),
                "systolic": max(110, min(160, int(systolic))),
                "diastolic": max(70, min(100, int(diastolic))),
                "pulse_pressure": int(systolic) - int(diastolic),
                "map": int(systolic) + (2 * int(diastolic)) / 3  # Mean Arterial Pressure
            })
            
        return trends
        
    def get_medications(self) -> List[Medication]:
        """Get hypertension medications"""
        return [
            Medication("Amlodipine", "5mg", "once daily", 0.95, "2024-11-03 08:00"),
            Medication("Lisinopril", "10mg", "once daily", 0.88, "2024-11-03 08:00"),
            Medication("Hydrochlorothiazide", "25mg", "once daily", 0.82, "2024-11-03 08:00")
        ]
        
    def get_cv_risk_score(self) -> Dict:
        """Calculate cardiovascular risk score"""
        return {
            "ascvd_10_year_risk": 12.5,
            "risk_category": "moderate",
            "factors": {
                "age": 58,
                "gender": "female", 
                "total_cholesterol": 210,
                "hdl_cholesterol": 55,
                "systolic_bp": 138,
                "smoking": False,
                "diabetes": False
            },
            "recommendations": [
                "Continue current BP management",
                "Consider lifestyle modifications",
                "Monitor cholesterol levels",
                "Annual cardiovascular assessment"
            ]
        }
        
    def get_ai_recommendations(self) -> List[AISuggestion]:
        """Generate hypertension-specific recommendations"""
        return [
            AISuggestion(
                "medication_adjustment",
                "Consider reducing Hydrochlorothiazide dose due to improved BP control",
                0.78,
                "medium",
                True
            ),
            AISuggestion(
                "lifestyle",
                "Increase aerobic exercise to 150 minutes per week",
                0.85,
                "medium",
                False
            ),
            AISuggestion(
                "monitoring",
                "Implement home BP monitoring with weekly documentation",
                0.92,
                "high",
                True
            ),
            AISuggestion(
                "dietary",
                "Consider DASH diet consultation",
                0.70,
                "low",
                False
            )
        ]

class ChestPainAssessmentScenario:
    """Chest Pain Assessment Scenario Template"""
    
    def __init__(self, patient_id: int = 3):
        self.patient_id = patient_id
        self.scenario_id = str(uuid.uuid4())
        self.patient_name = "Michael Johnson"
        self.patient_age = 62
        self.presentation_time = datetime.now()
        
    def get_presentation_symptoms(self) -> Dict:
        """Get chest pain presentation details"""
        return {
            "chief_complaint": "Crushing chest pain for 2 hours",
            "pain_characteristics": {
                "onset": "Gradual",
                "location": "Substernal, radiating to left arm",
                "quality": "Crushing, pressure-like",
                "severity": "8/10",
                "duration": "2 hours",
                "aggravating_factors": ["Physical exertion", "Emotional stress"],
                "relieving_factors": ["Rest", "Nitroglycerin (partial)"]
            },
            "associated_symptoms": [
                "Shortness of breath",
                "Diaphoresis", 
                "Nausea",
                "Lightheadedness"
            ],
            "risk_factors": {
                "age": 62,
                "gender": "male",
                "smoking": True,
                "diabetes": True,
                "hypertension": True,
                "family_history": True,
                "hyperlipidemia": True
            },
            "vital_signs": {
                "blood_pressure": "165/95",
                "heart_rate": 95,
                "respiratory_rate": 22,
                "temperature": "37.2°C",
                "oxygen_saturation": "94% on room air"
            }
        }
        
    def get_ecg_findings(self) -> Dict:
        """Simulate ECG findings"""
        return {
            "rhythm": "Normal sinus rhythm",
            "rate": 95,
            "intervals": {
                "PR": "160ms (normal)",
                "QRS": "90ms (normal)",
                "QT": "420ms (normal)"
            },
            "st_segments": "2mm ST elevation in leads V1-V4",
            "interpretation": "Anterior STEMI",
            "priority": "urgent"
        }
        
    def get_risk_stratification(self) -> Dict:
        """HEART score calculation"""
        return {
            "heart_score": {
                "history": 2,  # Suspicious history
                "ecg": 2,      # Significant ST deviation
                "age": 1,      # >65 years
                "risk_factors": 2,  # Multiple risk factors
                "troponin": 0, # Not available yet
                "total": 7
            },
            "risk_level": "high",
            "recommendation": "Immediate cardiology consultation, consider cath lab activation",
            "probability_mi": "85-90%"
        }
        
    def get_emergency_protocol(self) -> Dict:
        """Emergency response protocol"""
        return {
            "immediate_actions": [
                "Activate chest pain protocol",
                "Call cardiology consultation", 
                "Prepare for potential cath lab",
                "Administer aspirin 325mg",
                "Establish IV access",
                "Continuous cardiac monitoring",
                "Serial ECGs",
                "Draw cardiac biomarkers"
            ],
            "medications": [
                {"name": "Aspirin", "dose": "325mg", "route": "PO", "status": "administered"},
                {"name": "Nitroglycerin", "dose": "0.4mg", "route": "SL", "status": "ordered"},
                {"name": "Morphine", "dose": "2-4mg", "route": "IV", "status": "as needed"},
                {"name": "Clopidogrel", "dose": "600mg", "route": "PO", "status": "pending"}
            ],
            "consultations": [
                {"service": "Cardiology", "urgency": "immediate", "status": "consulted"},
                {"service": "Cath Lab", "urgency": "immediate", "status": "on standby"}
            ],
            "monitoring": {
                "continuous_cardiac": True,
                "blood_pressure": "q5min",
                "oxygen_saturation": "continuous",
                "neurological": "q15min"
            }
        }

class ScenarioManager:
    """Manages all demo scenarios"""
    
    def __init__(self):
        self.scenarios = {}
        
    def create_scenario(self, scenario_type: str, patient_id: int = 1) -> Any:
        """Create a scenario instance"""
        if scenario_type == "diabetes":
            self.scenarios[scenario_type] = DiabetesManagementScenario(patient_id)
        elif scenario_type == "hypertension":
            self.scenarios[scenario_type] = HypertensionMonitoringScenario(patient_id)  
        elif scenario_type == "chest_pain":
            self.scenarios[scenario_type] = ChestPainAssessmentScenario(patient_id)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        return self.scenarios[scenario_type]
        
    def get_scenario_data(self, scenario_type: str) -> Dict:
        """Get complete scenario data"""
        if scenario_type not in self.scenarios:
            self.create_scenario(scenario_type)
            
        scenario = self.scenarios[scenario_type]
        
        if isinstance(scenario, DiabetesManagementScenario):
            return {
                "patient_info": {
                    "name": scenario.patient_name,
                    "age": scenario.patient_age,
                    "diagnosis": scenario.diabetes_type,
                    "diagnosis_date": scenario.diagnosis_date
                },
                "vital_signs": [asdict(vs) for vs in scenario.get_vital_signs_series()],
                "lab_results": [asdict(lr) for lr in scenario.get_lab_results()],
                "medications": [asdict(med) for med in scenario.get_medications()],
                "ai_recommendations": [asdict(rec) for rec in scenario.get_ai_recommendations()],
                "workflow_steps": scenario.get_workflow_steps()
            }
            
        elif isinstance(scenario, HypertensionMonitoringScenario):
            return {
                "patient_info": {
                    "name": scenario.patient_name,
                    "age": scenario.patient_age,
                    "diagnosis_date": scenario.diagnosis_date
                },
                "bp_trends": scenario.get_bp_trends(),
                "medications": [asdict(med) for med in scenario.get_medications()],
                "cv_risk_score": scenario.get_cv_risk_score(),
                "ai_recommendations": [asdict(rec) for rec in scenario.get_ai_recommendations()]
            }
            
        elif isinstance(scenario, ChestPainAssessmentScenario):
            return {
                "patient_info": {
                    "name": scenario.patient_name,
                    "age": scenario.patient_age,
                    "presentation_time": scenario.presentation_time.isoformat()
                },
                "symptoms": scenario.get_presentation_symptoms(),
                "ecg_findings": scenario.get_ecg_findings(),
                "risk_stratification": scenario.get_risk_stratification(),
                "emergency_protocol": scenario.get_emergency_protocol()
            }
            
        return {}

# Demo scenario templates for easy access
DIABETES_TEMPLATE = {
    "name": "Diabetes Management",
    "description": "Comprehensive diabetes monitoring and management",
    "duration": 15,
    "difficulty": "intermediate",
    "learning_objectives": [
        "Understand glucose monitoring workflows",
        "Practice medication adjustment decisions", 
        "Generate evidence-based recommendations",
        "Create patient education plans"
    ]
}

HYPERTENSION_TEMPLATE = {
    "name": "Hypertension Monitoring", 
    "description": "Blood pressure management and cardiovascular risk assessment",
    "duration": 12,
    "difficulty": "beginner",
    "learning_objectives": [
        "Analyze BP trends over time",
        "Calculate cardiovascular risk scores",
        "Implement lifestyle interventions",
        "Monitor medication effectiveness"
    ]
}

CHEST_PAIN_TEMPLATE = {
    "name": "Chest Pain Assessment",
    "description": "Emergency triage and cardiovascular risk evaluation", 
    "duration": 10,
    "difficulty": "advanced",
    "learning_objectives": [
        "Perform systematic chest pain assessment",
        "Calculate HEART scores",
        "Recognize STEMI patterns",
        "Execute emergency protocols"
    ]
}

if __name__ == "__main__":
    # Test scenario creation
    manager = ScenarioManager()
    
    # Create diabetes scenario
    diabetes = manager.create_scenario("diabetes", 1)
    diabetes_data = manager.get_scenario_data("diabetes")
    
    print("Diabetes Scenario Created:")
    print(f"Patient: {diabetes_data['patient_info']['name']}")
    print(f"Vital Signs Records: {len(diabetes_data['vital_signs'])}")
    print(f"AI Recommendations: {len(diabetes_data['ai_recommendations'])}")