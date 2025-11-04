#!/usr/bin/env python3
"""
Demo Scenarios - Compelling medical scenarios for stakeholder demonstrations.

This module provides comprehensive medical scenarios designed for different stakeholder types
and medical specialties, showcasing the Medical AI Assistant's capabilities in realistic,
engaging clinical situations.
"""

import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class PatientProfile:
    """Patient profile for demo scenarios"""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: List[str]
    medications: List[str]
    allergies: List[str]
    vital_signs: Dict[str, Any]
    lab_values: Dict[str, Any]
    current_symptoms: List[str]
    
@dataclass
class ScenarioStep:
    """Individual step in a demo scenario"""
    step_id: str
    title: str
    description: str
    duration_minutes: int
    presenter_notes: str
    expected_ai_responses: List[str]
    visual_elements: List[str]
    key_learning_points: List[str]
    stakeholder_focus: List[str]
    
@dataclass
class MedicalScenario:
    """Complete medical scenario definition"""
    scenario_id: str
    title: str
    specialty: str
    complexity_level: str
    duration_minutes: int
    patient_profile: PatientProfile
    scenario_steps: List[ScenarioStep]
    learning_objectives: List[str]
    stakeholder_outcomes: Dict[str, str]
    clinical_guidelines: List[str]

class DemoScenarioManager:
    """Manager for medical demo scenarios"""
    
    def __init__(self):
        self.scenarios_db = self._initialize_scenarios_db()
        
    def _initialize_scenarios_db(self) -> Dict[str, MedicalScenario]:
        """Initialize comprehensive scenarios database"""
        scenarios = {}
        
        # Cardiology Scenarios
        scenarios["stemi_pci"] = self._create_stemi_scenario()
        scenarios["heart_failure"] = self._create_heart_failure_scenario()
        
        # Oncology Scenarios
        scenarios["breast_cancer"] = self._create_breast_cancer_scenario()
        
        # Emergency Medicine Scenarios
        scenarios["stroke_tpa"] = self._create_stroke_tpa_scenario()
        
        # Chronic Disease Scenarios
        scenarios["diabetes_management"] = self._create_diabetes_scenario()
        
        return scenarios
    
    def _create_patient_profile(self, patient_id: str, age: int, gender: str, 
                              chief_complaint: str, history: str, past_med: List[str],
                              meds: List[str], allergies: List[str], vitals: Dict,
                              labs: Dict, symptoms: List[str]) -> PatientProfile:
        """Create standardized patient profile"""
        return PatientProfile(
            patient_id=patient_id,
            age=age,
            gender=gender,
            chief_complaint=chief_complaint,
            history_of_present_illness=history,
            past_medical_history=past_med,
            medications=meds,
            allergies=allergies,
            vital_signs=vitals,
            lab_values=labs,
            current_symptoms=symptoms
        )
    
    def _create_stemi_scenario(self) -> MedicalScenario:
        """Create STEMI with PCI scenario"""
        patient = self._create_patient_profile(
            patient_id="PT001",
            age=62,
            gender="Male",
            chief_complaint="Severe chest pain for 2 hours",
            history="Patient presents with crushing substernal chest pain radiating to left arm, associated with diaphoresis and nausea. Pain started 2 hours ago while watching TV.",
            past_med=["Hypertension", "Hyperlipidemia", "Type 2 Diabetes"],
            meds=["Lisinopril 10mg daily", "Atorvastatin 40mg daily", "Metformin 500mg twice daily"],
            allergies=["Penicillin - rash"],
            vitals={"BP": "158/94", "HR": "98", "RR": "22", "O2_Sat": "94% RA", "Temp": "98.6F"},
            labs={"Troponin": "0.8 ng/mL (elevated)", "BNP": "450 pg/mL", "CK-MB": "25 U/L"},
            symptoms=["Crushing chest pain", "Diaphoresis", "Nausea", "Shortness of breath"]
        )
        
        steps = [
            ScenarioStep(
                step_id="triage",
                title="Emergency Department Triage",
                description="Patient triaged as Level 1 (highest priority) with chest pain protocol activation",
                duration_minutes=2,
                presenter_notes="Emphasize time-critical nature of chest pain evaluation",
                expected_ai_responses=[
                    "Activating chest pain protocol immediately",
                    "ECG ordered stat",
                    "Cardiology consultation requested",
                    "Aspirin 325mg administered"
                ],
                visual_elements=["ECG monitor showing ST elevation", "Triage assessment screen"],
                key_learning_points=["Time is muscle in STEMI", "Door-to-balloon time targets"],
                stakeholder_focus=["C-Suite: Cost of delays", "Clinical: Protocol compliance", "Regulatory: Quality metrics"]
            ),
            ScenarioStep(
                step_id="diagnosis",
                title="STEMI Diagnosis Confirmation",
                description="ECG shows 3mm ST elevation in leads V2-V4 consistent with anterior STEMI",
                duration_minutes=3,
                presenter_notes="Highlight AI-assisted ECG interpretation accuracy",
                expected_ai_responses=[
                    "ECG interpretation: Anterior STEMI confirmed",
                    "STEMI alert activated",
                    "Cardiac catheterization lab notified",
                    "Door-to-balloon time: 45 minutes (target <90 minutes)"
                ],
                visual_elements=["12-lead ECG with ST elevation", "STEMI alert system"],
                key_learning_points=["Automated ECG interpretation", "Protocol automation"],
                stakeholder_focus=["Technical: AI accuracy", "Clinical: Reduced diagnostic time"]
            ),
            ScenarioStep(
                step_id="treatment",
                title="PCI Treatment and Post-Procedure Care",
                description="Emergency PCI performed with drug-eluting stent placement in LAD",
                duration_minutes=4,
                presenter_notes="Show comprehensive post-PCI care recommendations",
                expected_ai_responses=[
                    "PCI completed successfully - DES placed in LAD",
                    "Dual antiplatelet therapy recommended",
                    "Cardiac rehabilitation referral",
                    "Lifestyle modification counseling initiated",
                    "Follow-up echo scheduled in 6 weeks"
                ],
                visual_elements=["Angiography images", "Stent placement visualization"],
                key_learning_points=["Evidence-based post-PCI care", "Patient education automation"],
                stakeholder_focus=["Clinical: Improved outcomes", "Patient: Better recovery"]
            ),
            ScenarioStep(
                step_id="outcome",
                title="Patient Outcome and System Benefits",
                description="Patient discharged day 3 with excellent recovery and care coordination",
                duration_minutes=3,
                presenter_notes="Highlight system-wide improvements and ROI",
                expected_ai_responses=[
                    "Patient discharged day 3 with stable condition",
                    "30-day readmission risk: Low (8%)",
                    "Care coordination: Seamless follow-up scheduled",
                    "Cost savings: $15,000 vs. standard care pathway"
                ],
                visual_elements=["Patient recovery timeline", "Cost-benefit analysis"],
                key_learning_points=["System efficiency gains", "Quality improvement metrics"],
                stakeholder_focus=["C-Suite: ROI demonstration", "Regulatory: Quality metrics"]
            )
        ]
        
        return MedicalScenario(
            scenario_id="stemi_pci",
            title="STEMI Management with Primary PCI",
            specialty="Cardiology",
            complexity_level="Advanced",
            duration_minutes=12,
            patient_profile=patient,
            scenario_steps=steps,
            learning_objectives=[
                "Demonstrate time-critical decision making in STEMI",
                "Show AI-assisted ECG interpretation accuracy",
                "Highlight evidence-based treatment protocols",
                "Illustrate care coordination and patient outcomes"
            ],
            stakeholder_outcomes={
                "C-Suite": "$15,000 cost savings per case, reduced length of stay",
                "Clinical": "Faster diagnosis, improved outcomes, protocol compliance",
                "Regulatory": "Quality metrics compliance, door-to-balloon time targets",
                "Investor": "Proven ROI, scalability potential"
            },
            clinical_guidelines=[
                "ACC/AHA STEMI Guidelines 2023",
                "Door-to-Balloon Time <90 minutes",
                "Dual antiplatelet therapy protocols"
            ]
        )
    
    def _create_heart_failure_scenario(self) -> MedicalScenario:
        """Create heart failure management scenario"""
        patient = self._create_patient_profile(
            patient_id="PT002",
            age=68,
            gender="Female",
            chief_complaint="Progressive shortness of breath and leg swelling",
            history="Patient reports increasing dyspnea on exertion over past 2 weeks, orthopnea, and bilateral leg swelling. Unable to climb stairs without stopping.",
            past_med=["Hypertension", "Diabetes mellitus", "Coronary artery disease"],
            meds=["Lisinopril 20mg daily", "Metoprolol 50mg twice daily", "Furosemide 40mg daily"],
            allergies=["None known"],
            vitals={"BP": "142/88", "HR": "88", "RR": "24", "O2_Sat": "92% RA", "Temp": "98.4F"},
            labs={"BNP": "1200 pg/mL", "Creatinine": "1.2 mg/dL", "Sodium": "132 mEq/L"},
            symptoms=["Dyspnea on exertion", "Orthopnea", "Bilateral leg swelling", "Fatigue"]
        )
        
        steps = [
            ScenarioStep(
                step_id="assessment",
                title="Heart Failure Assessment",
                description="Comprehensive evaluation including echo and BNP levels showing reduced EF",
                duration_minutes=3,
                presenter_notes="Emphasize comprehensive assessment protocols",
                expected_ai_responses=[
                    "BNP elevated to 1200 pg/mL - consistent with heart failure",
                    "Echo ordered: EF 35% (reduced ejection fraction)",
                    "NYHA Class III heart failure confirmed",
                    "Guideline-directed medical therapy optimization recommended"
                ],
                visual_elements=["BNP trending", "Echo results", "NYHA classification"],
                key_learning_points=["Diagnostic criteria for HF", "BNP utility in diagnosis"],
                stakeholder_focus=["Clinical: Evidence-based diagnostics", "Regulatory: Quality measures"]
            ),
            ScenarioStep(
                step_id="optimization",
                title="GDMT Optimization",
                description="AI-assisted medication optimization using latest guidelines",
                duration_minutes=4,
                presenter_notes="Show AI capability in complex medication management",
                expected_ai_responses=[
                    "GDMT optimization recommended: Increase ACE inhibitor, add ARNI",
                    "SGLT2 inhibitor addition for additional benefits",
                    "Mineralocorticoid receptor antagonist consideration",
                    "Ivabradine for heart rate control if needed"
                ],
                visual_elements=["Medication algorithm", "Guideline adherence dashboard"],
                key_learning_points=["Complex medication optimization", "AI-guided therapy"],
                stakeholder_focus=["Clinical: Improved outcomes", "Technical: Complex algorithms"]
            ),
            ScenarioStep(
                step_id="monitoring",
                title="Remote Monitoring Setup",
                description="Implementation of remote patient monitoring for early intervention",
                duration_minutes=3,
                presenter_notes="Highlight technology integration and prevention",
                expected_ai_responses=[
                    "Remote monitoring: Daily weight and symptom tracking",
                    "Alert thresholds: Weight gain >2 lbs in 24 hours",
                    "Proactive intervention for early decompensation signs",
                    "Reduced readmission risk by 35%"
                ],
                visual_elements=["Remote monitoring dashboard", "Alert system"],
                key_learning_points=["Remote monitoring benefits", "Predictive analytics"],
                stakeholder_focus=["C-Suite: Cost reduction", "Patient: Improved quality of life"]
            ),
            ScenarioStep(
                step_id="outcomes",
                title="Long-term Management Success",
                description="6-month follow-up showing significant improvement in quality of life",
                duration_minutes=2,
                presenter_notes="Emphasize long-term benefits and system value",
                expected_ai_responses=[
                    "6-month follow-up: NYHA Class II (improved from Class III)",
                    "EF improved to 42% with optimal medical therapy",
                    "No hospital readmissions in 6 months",
                    "Quality of life scores significantly improved"
                ],
                visual_elements=["Quality of life metrics", "Outcome trending"],
                key_learning_points=["Long-term benefits of GDMT", "Quality of life improvements"],
                stakeholder_focus=["All stakeholders: Comprehensive value demonstration"]
            )
        ]
        
        return MedicalScenario(
            scenario_id="heart_failure",
            title="Heart Failure with Reduced Ejection Fraction",
            specialty="Cardiology",
            complexity_level="Intermediate",
            duration_minutes=12,
            patient_profile=patient,
            scenario_steps=steps,
            learning_objectives=[
                "Demonstrate evidence-based heart failure management",
                "Show AI-assisted medication optimization",
                "Highlight remote monitoring benefits",
                "Illustrate quality of life improvements"
            ],
            stakeholder_outcomes={
                "C-Suite": "35% reduction in readmissions, improved quality metrics",
                "Clinical": "Evidence-based care, improved outcomes, reduced complexity",
                "Patient": "Better quality of life, reduced hospitalizations",
                "Investor": "Scalable technology solution with proven outcomes"
            },
            clinical_guidelines=[
                "ACC/AHA Heart Failure Guidelines 2022",
                "ESC Heart Failure Guidelines 2023",
                "GDMT optimization protocols"
            ]
        )
    
    def _create_breast_cancer_scenario(self) -> MedicalScenario:
        """Create breast cancer multidisciplinary care scenario"""
        patient = self._create_patient_profile(
            patient_id="PT003",
            age=45,
            gender="Female",
            chief_complaint="Palpable lump in right breast discovered during self-examination",
            history="Patient noticed a 2cm firm, mobile mass in upper outer quadrant of right breast during self-examination. No nipple discharge or skin changes. Family history of mother with breast cancer at age 52.",
            past_med=["No significant medical history"],
            meds=["Multivitamin daily"],
            allergies=["None known"],
            vitals={"BP": "118/72", "HR": "72", "RR": "16", "O2_Sat": "98% RA", "Temp": "98.6F"},
            labs={"CBC": "Normal", "CA 15-3": "25 U/mL", "CEA": "2.1 ng/mL"},
            symptoms=["Palpable breast mass", "Anxiety about diagnosis"]
        )
        
        steps = [
            ScenarioStep(
                step_id="diagnosis",
                title="Multidisciplinary Cancer Diagnosis",
                description="Coordinated diagnostic workup involving imaging, pathology, and genetics",
                duration_minutes=4,
                presenter_notes="Show integrated care coordination capabilities",
                expected_ai_responses=[
                    "Mammography and ultrasound ordered: BIRADS 5 - highly suspicious",
                    "Core needle biopsy scheduled for same day",
                    "Genetic counseling referral for family history",
                    "Multidisciplinary tumor board consultation initiated"
                ],
                visual_elements=["Imaging results", "BIRADS classification", "Care coordination timeline"],
                key_learning_points=["Rapid diagnostic pathways", "Multidisciplinary approach"],
                stakeholder_focus=["Clinical: Coordinated care", "Patient: Streamlined experience"]
            ),
            ScenarioStep(
                step_id="treatment_planning",
                title="Personalized Treatment Planning",
                description="AI-assisted treatment planning based on tumor characteristics and patient factors",
                duration_minutes=4,
                presenter_notes="Highlight precision medicine and personalized care",
                expected_ai_responses=[
                    "Biopsy results: ER+/PR+/HER2- invasive ductal carcinoma",
                    "Treatment plan: Neoadjuvant chemotherapy → surgery → radiation",
                    "Oncotype DX testing for chemotherapy benefit prediction",
                    "Fertility preservation consultation before treatment",
                    "Patient preference and shared decision-making integrated"
                ],
                visual_elements=["Treatment pathway", "Genomic testing results", "Decision aids"],
                key_learning_points=["Personalized medicine", "Shared decision making"],
                stakeholder_focus=["Clinical: Evidence-based planning", "Patient: Personal control"]
            ),
            ScenarioStep(
                step_id="treatment_delivery",
                title="Coordinated Treatment Delivery",
                description="Seamless coordination across multiple specialties and treatment modalities",
                duration_minutes=3,
                presenter_notes="Show system integration and patient journey management",
                expected_ai_responses=[
                    "Chemotherapy regimen initiated: AC-T (dose-dense)",
                    "Fertility preservation completed before treatment start",
                    "Side effect management protocol activated",
                    "Social work and nutrition support integrated",
                    "Treatment timeline optimized for patient convenience"
                ],
                visual_elements=["Treatment schedule", "Support services integration", "Timeline optimization"],
                key_learning_points=["Integrated care delivery", "Patient-centered approach"],
                stakeholder_focus=["C-Suite: Operational efficiency", "Patient: Comprehensive support"]
            ),
            ScenarioStep(
                step_id="survivorship",
                title="Cancer Survivorship and Long-term Care",
                description="Transition to survivorship care with ongoing monitoring and support",
                duration_minutes=3,
                presenter_notes="Emphasize long-term value and patient outcomes",
                expected_ai_responses=[
                    "2-year post-treatment: No evidence of disease",
                    "Survivorship care plan: Annual mammography, oncology follow-up",
                    "Late effects monitoring: Cardiac function, secondary malignancies",
                    "Patient advocacy and support group integration",
                    "Quality of life restoration to pre-diagnosis levels"
                ],
                visual_elements=["Survivorship care plan", "Quality of life metrics", "Support resources"],
                key_learning_points=["Long-term survivorship care", "Quality of life focus"],
                stakeholder_focus=["All stakeholders: Complete care cycle value"]
            )
        ]
        
        return MedicalScenario(
            scenario_id="breast_cancer",
            title="Early-Stage Breast Cancer - Multidisciplinary Care",
            specialty="Oncology",
            complexity_level="Advanced",
            duration_minutes=14,
            patient_profile=patient,
            scenario_steps=steps,
            learning_objectives=[
                "Demonstrate multidisciplinary cancer care coordination",
                "Show precision medicine and personalized treatment",
                "Highlight patient-centered care delivery",
                "Illustrate survivorship and long-term outcomes"
            ],
            stakeholder_outcomes={
                "C-Suite": "Integrated care model, improved patient satisfaction, reduced fragmentation",
                "Clinical": "Evidence-based multidisciplinary care, improved outcomes",
                "Patient": "Comprehensive support, shared decision making, quality of life",
                "Investor": "Scalable cancer care platform, market expansion opportunity"
            },
            clinical_guidelines=[
                "NCCN Breast Cancer Guidelines 2024",
                "ASCO Clinical Practice Guidelines",
                "Multidisciplinary cancer care standards"
            ]
        )
    
    def _create_stroke_tpa_scenario(self) -> MedicalScenario:
        """Create acute stroke with tPA scenario"""
        patient = self._create_patient_profile(
            patient_id="PT004",
            age=72,
            gender="Female",
            chief_complaint="Sudden onset right-sided weakness and speech difficulty",
            history="Patient was eating breakfast when sudden right-sided weakness and slurred speech developed.症状 started 90 minutes ago. NIHSS score: 14. No contraindications to thrombolysis identified.",
            past_med=["Atrial fibrillation", "Hypertension", "Previous TIA 2 years ago"],
            meds=["Warfarin 5mg daily", "Lisinopril 10mg daily", "Aspirin 81mg daily"],
            allergies=["None known"],
            vitals={"BP": "168/92", "HR": "110 irregular", "RR": "18", "O2_Sat": "96% RA", "Temp": "98.8F"},
            labs={"INR": "1.2", "Platelet count": "245K", "Glucose": "120 mg/dL"},
            symptoms=["Right hemiparesis", "Dysarthria", "Facial droop", "Right visual field defect"]
        )
        
        steps = [
            ScenarioStep(
                step_id="rapid_assessment",
                title="Emergency Stroke Assessment",
                description="Rapid triage and assessment with AI-assisted NIHSS scoring and imaging",
                duration_minutes=3,
                presenter_notes="Emphasize time-critical nature of stroke care",
                expected_ai_responses=[
                    "NIHSS score: 14 (moderate to severe stroke)",
                    "CT head ordered stat - no hemorrhage identified",
                    "Large vessel occlusion suspected",
                    "Stroke team activated, door-to-needle time target: <60 minutes"
                ],
                visual_elements=["NIHSS assessment tool", "CT scan results", "Time metrics"],
                key_learning_points=["Rapid stroke assessment", "Time-critical interventions"],
                stakeholder_focus=["Clinical: Protocol compliance", "Regulatory: Quality metrics"]
            ),
            ScenarioStep(
                step_id="tpa_decision",
                title="Thrombolysis Decision Making",
                description="AI-assisted decision support for tPA eligibility and risk assessment",
                duration_minutes=4,
                presenter_notes="Show complex decision support capabilities",
                expected_ai_responses=[
                    "tPA eligibility confirmed: Within 4.5-hour window",
                    "No hemorrhage on CT, INR <1.7, glucose normal",
                    "SICH risk estimated: 3.5% (acceptable for potential benefit)",
                    "Family discussion and consent obtained",
                    "tPA dose calculated: 0.9 mg/kg (max 90mg)"
                ],
                visual_elements=["Decision algorithm", "Risk-benefit analysis", "Consent process"],
                key_learning_points=["Evidence-based thrombolysis", "Risk stratification"],
                stakeholder_focus=["Clinical: Decision support", "Patient: Informed consent"]
            ),
            ScenarioStep(
                step_id="intervention",
                title="Thrombolytic Therapy and Thrombectomy",
                description="Rapid tPA administration and endovascular intervention coordination",
                duration_minutes=4,
                presenter_notes="Highlight system integration and coordination",
                expected_ai_responses=[
                    "tPA administered at 120 minutes from symptom onset",
                    "CTA performed: Left MCA occlusion confirmed",
                    "Mechanical thrombectomy coordination initiated",
                    "Door-to-needle time: 45 minutes (exceeds target)",
                    "ICU bed reserved for post-tPA monitoring"
                ],
                visual_elements=["tPA administration timeline", "Thrombectomy coordination", "Monitoring protocols"],
                key_learning_points=["Rapid therapy delivery", "System coordination"],
                stakeholder_focus=["C-Suite: Operational excellence", "Clinical: Team coordination"]
            ),
            ScenarioStep(
                step_id="outcome",
                title="Stroke Recovery and Rehabilitation",
                description="Comprehensive post-stroke care with rehabilitation and secondary prevention",
                duration_minutes=3,
                presenter_notes="Show long-term value and outcome improvements",
                expected_ai_responses=[
                    "24-hour NIHSS improvement: 14 to 6 (significant recovery)",
                    "Early rehabilitation initiated: Physical and speech therapy",
                    "Secondary prevention: Anticoagulation optimization",
                    "Stroke education and lifestyle modification program",
                    "3-month outcome: mRS 2 (slight disability, independent)"
                ],
                visual_elements=["Recovery timeline", "Rehabilitation milestones", "Outcome metrics"],
                key_learning_points=["Stroke recovery optimization", "Secondary prevention"],
                stakeholder_focus=["All stakeholders: Comprehensive stroke care value"]
            )
        ]
        
        return MedicalScenario(
            scenario_id="stroke_tpa",
            title="Acute Ischemic Stroke - Thrombolysis and Thrombectomy",
            specialty="Emergency Medicine",
            complexity_level="Advanced",
            duration_minutes=14,
            patient_profile=patient,
            scenario_steps=steps,
            learning_objectives=[
                "Demonstrate rapid stroke assessment and treatment",
                "Show AI-assisted decision support for thrombolysis",
                "Highlight system coordination and timing",
                "Illustrate stroke recovery and rehabilitation"
            ],
            stakeholder_outcomes={
                "C-Suite": "Superior stroke program metrics, reduced disability, system reputation",
                "Clinical": "Evidence-based stroke care, improved outcomes, protocol adherence",
                "Patient": "Reduced disability, improved quality of life, comprehensive care",
                "Investor": "Market-leading stroke care solution, regulatory approval pathway"
            },
            clinical_guidelines=[
                "AHA/ASA Stroke Guidelines 2024",
                "Door-to-needle time targets",
                "tPA eligibility criteria"
            ]
        )
    
    def _create_diabetes_scenario(self) -> MedicalScenario:
        """Create comprehensive diabetes management scenario"""
        patient = self._create_patient_profile(
            patient_id="PT005",
            age=58,
            gender="Male",
            chief_complaint="Poorly controlled diabetes with complications",
            history="Patient has difficulty managing diabetes despite multiple medications. Recent HbA1c 8.7%. Experiencing polyuria, polydipsia, and fatigue. Non-compliant with diet recommendations.",
            past_med=["Type 2 Diabetes (15 years)", "Hypertension", "Hyperlipidemia", "Early diabetic nephropathy"],
            meds=["Metformin 1000mg twice daily", "Glipizide 10mg daily", "Lisinopril 20mg daily"],
            allergies=["Sulfa drugs - rash"],
            vitals={"BP": "142/88", "HR": "76", "BMI": "32.5", "Weight": "215 lbs"},
            labs={"HbA1c": "8.7%", "Fasting glucose": "180 mg/dL", "Creatinine": "1.3 mg/dL"},
            symptoms=["Polyuria", "Polydipsia", "Fatigue", "Poor appetite"]
        )
        
        steps = [
            ScenarioStep(
                step_id="comprehensive_assessment",
                title="Comprehensive Diabetes Assessment",
                description="AI-assisted assessment including complications screening and medication review",
                duration_minutes=4,
                presenter_notes="Show comprehensive diabetes evaluation capabilities",
                expected_ai_responses=[
                    "HbA1c elevated at 8.7% (target <7%)",
                    "Comprehensive diabetes screening: Foot exam, eye exam, microalbumin",
                    "Medication adherence assessment: 60% compliance rate identified",
                    "Insulin resistance calculation: HOMA-IR = 4.2 (elevated)",
                    "Cardiovascular risk assessment: High risk due to diabetes + hypertension"
                ],
                visual_elements=["HbA1c trends", "Complication screening results", "Risk stratification"],
                key_learning_points=["Comprehensive diabetes care", "Complications screening"],
                stakeholder_focus=["Clinical: Evidence-based care", "Patient: Holistic approach"]
            ),
            ScenarioStep(
                step_id="personalized_plan",
                title="Personalized Treatment Optimization",
                description="AI-driven medication optimization and lifestyle intervention planning",
                duration_minutes=5,
                presenter_notes="Highlight personalized medicine and behavioral support",
                expected_ai_responses=[
                    "Medication optimization: Add SGLT2 inhibitor (CVD benefit + weight loss)",
                    "Insulin therapy initiation: Basal-bolus regimen recommended",
                    "CGM device prescription for glucose monitoring",
                    "Diabetes education referral: Nutritionist and diabetes educator",
                    "Weight management program enrollment",
                    "Medication adherence support: Pill organizer + reminders"
                ],
                visual_elements=["Medication algorithm", "CGM data visualization", "Education plan"],
                key_learning_points=["Personalized therapy", "Technology integration"],
                stakeholder_focus=["Clinical: Improved outcomes", "Patient: Technology support"]
            ),
            ScenarioStep(
                step_id="continuous_monitoring",
                title="Continuous Monitoring and Support",
                description="Remote monitoring with AI-powered insights and early intervention",
                duration_minutes=3,
                presenter_notes="Show remote care capabilities and prevention",
                expected_ai_responses=[
                    "CGM integration: Real-time glucose alerts and insights",
                    "Medication adherence monitoring: Smart pill bottle data",
                    "Blood pressure tracking: Home BP monitoring integration",
                    "Early intervention alerts: Hypoglycemia prevention",
                    "Behavioral coaching: Weekly check-ins and motivation",
                    "Family involvement: Caregiver education and support"
                ],
                visual_elements=["CGM dashboard", "Adherence monitoring", "Alert system"],
                key_learning_points=["Remote monitoring benefits", "Predictive interventions"],
                stakeholder_focus=["C-Suite: Reduced complications", "Patient: Peace of mind"]
            ),
            ScenarioStep(
                step_id="outcome_tracking",
                title="Long-term Outcomes and Quality of Life",
                description="6-month follow-up showing dramatic improvement in control",
                duration_minutes=2,
                presenter_notes="Demonstrate long-term value and patient transformation",
                expected_ai_responses=[
                    "6-month HbA1c: 7.1% (significant improvement from 8.7%)",
                    "Weight loss: 15 pounds (improved insulin sensitivity)",
                    "No hypoglycemic episodes in 6 months",
                    "Blood pressure controlled: 128/78 average",
                    "Quality of life scores: Dramatically improved",
                    "Patient engagement: 95% medication adherence"
                ],
                visual_elements=["Outcome trends", "Quality of life metrics", "Engagement scores"],
                key_learning_points=["Long-term diabetes control", "Patient empowerment"],
                stakeholder_focus=["All stakeholders: Transformation success story"]
            )
        ]
        
        return MedicalScenario(
            scenario_id="diabetes_management",
            title="Complex Diabetes Management with Technology Integration",
            specialty="Endocrinology",
            complexity_level="Advanced",
            duration_minutes=14,
            patient_profile=patient,
            scenario_steps=steps,
            learning_objectives=[
                "Demonstrate comprehensive diabetes care coordination",
                "Show AI-assisted medication optimization",
                "Highlight technology integration for monitoring",
                "Illustrate long-term outcome improvements"
            ],
            stakeholder_outcomes={
                "C-Suite": "30% reduction in diabetes complications, improved quality metrics",
                "Clinical": "Evidence-based care, patient engagement, reduced complexity",
                "Patient": "Better control, technology support, improved quality of life",
                "Investor": "Proven diabetes management platform, market expansion"
            },
            clinical_guidelines=[
                "ADA Standards of Medical Care in Diabetes 2024",
                "ACC/AHA Cardiovascular guidelines",
                "Technology integration recommendations"
            ]
        )
    
    def get_scenario(self, scenario_id: str) -> Optional[MedicalScenario]:
        """Get specific scenario by ID"""
        return self.scenarios_db.get(scenario_id)
    
    def get_scenarios_by_specialty(self, specialty: str) -> List[MedicalScenario]:
        """Get scenarios by medical specialty"""
        return [scenario for scenario in self.scenarios_db.values() 
                if scenario.specialty.lower() == specialty.lower()]
    
    def get_scenarios_by_stakeholder(self, stakeholder: str) -> List[Dict[str, Any]]:
        """Get scenarios optimized for specific stakeholder"""
        scenarios = []
        
        for scenario in self.scenarios_db.values():
            # Select scenarios that have strong stakeholder outcomes
            if stakeholder.upper() in scenario.stakeholder_outcomes:
                scenarios.append({
                    "scenario": scenario,
                    "key_outcomes": scenario.stakeholder_outcomes[stakeholder.upper()],
                    "focus_areas": self._get_stakeholder_focus_areas(scenario, stakeholder)
                })
        
        return scenarios
    
    def _get_stakeholder_focus_areas(self, scenario: MedicalScenario, stakeholder: str) -> List[str]:
        """Get specific focus areas for stakeholder"""
        focus_areas = {
            "C-Suite": ["ROI", "Cost savings", "Quality metrics", "Operational efficiency"],
            "Clinical": ["Evidence-based care", "Patient outcomes", "Workflow optimization"],
            "Regulatory": ["Compliance", "Quality measures", "Safety protocols"],
            "Investor": ["Market opportunity", "Competitive advantage", "Scalability"],
            "Technical": ["AI capabilities", "Integration", "Performance"],
            "Patient": ["Quality of life", "Experience", "Support"]
        }
        
        return focus_areas.get(stakeholder.upper(), [])
    
    def generate_scenario_summary(self, scenario: MedicalScenario) -> Dict[str, Any]:
        """Generate comprehensive scenario summary"""
        return {
            "scenario_overview": {
                "id": scenario.scenario_id,
                "title": scenario.title,
                "specialty": scenario.specialty,
                "complexity": scenario.complexity_level,
                "duration": f"{scenario.duration_minutes} minutes"
            },
            "patient_profile": asdict(scenario.patient_profile),
            "scenario_steps": [asdict(step) for step in scenario.scenario_steps],
            "learning_objectives": scenario.learning_objectives,
            "stakeholder_outcomes": scenario.stakeholder_outcomes,
            "clinical_guidelines": scenario.clinical_guidelines,
            "demo_preparation": {
                "required_materials": self._get_required_materials(scenario),
                "key_demonstration_points": self._get_demo_points(scenario),
                "audience_engagement": self._get_engagement_strategies(scenario)
            }
        }
    
    def _get_required_materials(self, scenario: MedicalScenario) -> List[str]:
        """Get required materials for scenario demonstration"""
        materials = [
            "Patient profile handouts",
            "Clinical decision support interface",
            "Medical imaging displays (if applicable)",
            "Timeline/progression charts",
            "Outcome metrics dashboard"
        ]
        
        # Add specialty-specific materials
        if scenario.specialty.lower() == "cardiology":
            materials.extend(["ECG displays", "Echo results", "Cath lab coordination"])
        elif scenario.specialty.lower() == "oncology":
            materials.extend(["Imaging results", "Pathology reports", "Treatment timelines"])
        elif scenario.specialty.lower() == "emergency":
            materials.extend(["Triage protocols", "Time-critical metrics", "Alert systems"])
        
        return materials
    
    def _get_demo_points(self, scenario: MedicalScenario) -> List[str]:
        """Get key demonstration points"""
        return [
            f"AI-powered clinical decision support in {scenario.specialty}",
            "Evidence-based protocol automation",
            "Real-time patient monitoring and alerts",
            "Care coordination and team communication",
            "Patient outcome improvements",
            "System efficiency gains"
        ]
    
    def _get_engagement_strategies(self, scenario: MedicalScenario) -> List[str]:
        """Get audience engagement strategies"""
        return [
            "Interactive audience polling during scenario",
            "Real-time Q&A integration",
            "Live system demonstrations",
            "Before/after outcome comparisons",
            "Stakeholder-specific value discussions"
        ]

def main():
    """Main function for scenario management CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Demo Scenario Manager")
    parser.add_argument("--list-scenarios", action="store_true", help="List all available scenarios")
    parser.add_argument("--specialty", type=str, help="Filter by medical specialty")
    parser.add_argument("--stakeholder", type=str, help="Get scenarios for specific stakeholder")
    parser.add_argument("--scenario-id", type=str, help="Get specific scenario details")
    parser.add_argument("--output", type=str, help="Output file for scenario details")
    
    args = parser.parse_args()
    
    manager = DemoScenarioManager()
    
    if args.list_scenarios:
        print("Available Medical Scenarios:")
        print("-" * 50)
        for scenario_id, scenario in manager.scenarios_db.items():
            print(f"{scenario_id}: {scenario.title}")
            print(f"  Specialty: {scenario.specialty}")
            print(f"  Complexity: {scenario.complexity_level}")
            print(f"  Duration: {scenario.duration_minutes} minutes")
            print()
    
    elif args.scenario_id:
        scenario = manager.get_scenario(args.scenario_id)
        if scenario:
            summary = manager.generate_scenario_summary(scenario)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Scenario details saved to {args.output}")
            else:
                print(json.dumps(summary, indent=2))
        else:
            print(f"Scenario not found: {args.scenario_id}")
    
    elif args.specialty:
        scenarios = manager.get_scenarios_by_specialty(args.specialty)
        print(f"Scenarios for {args.specialty}:")
        for scenario in scenarios:
            print(f"- {scenario.title} ({scenario.scenario_id})")
    
    elif args.stakeholder:
        scenarios = manager.get_scenarios_by_stakeholder(args.stakeholder)
        print(f"Scenarios optimized for {args.stakeholder}:")
        for scenario_info in scenarios:
            scenario = scenario_info["scenario"]
            print(f"- {scenario.title}")
            print(f"  Key outcomes: {scenario_info['key_outcomes']}")
            print()

if __name__ == "__main__":
    main()
