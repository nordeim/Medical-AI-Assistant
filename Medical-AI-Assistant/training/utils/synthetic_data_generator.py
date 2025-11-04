"""
Synthetic Data Generator for Medical AI Assistant Training

This module generates synthetic medical scenarios, patient conversations, and
training data for augmentation purposes.
"""

import random
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
from collections import defaultdict


class TriageLevel(Enum):
    """Emergency triage levels"""
    IMMEDIATE = 1  # Life-threatening
    EMERGENT = 2   # Urgent care needed within minutes
    URGENT = 3     # Care needed within 30 minutes
    LESS_URGENT = 4  # Care needed within 2 hours
    NON_URGENT = 5   # Care needed within 24 hours


class AgeGroup(Enum):
    """Patient age groups"""
    INFANT = "0-2"
    CHILD = "3-12"
    ADOLESCENT = "13-17"
    YOUNG_ADULT = "18-25"
    ADULT = "26-64"
    GERIATRIC = "65+"


class MedicalSpecialty(Enum):
    """Medical specialties"""
    GENERAL = "general"
    CARDIOLOGY = "cardiology"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    NEUROLOGY = "neurology"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    ORTHOPEDICS = "orthopedics"
    DERMATOLOGY = "dermatology"
    MENTAL_HEALTH = "mental_health"


@dataclass
class Patient:
    """Patient profile"""
    id: str
    age: int
    age_group: AgeGroup
    gender: str
    medical_history: List[str]
    current_medications: List[str]
    allergies: List[str]
    vital_signs: Dict[str, float]


@dataclass
class Symptom:
    """Medical symptom"""
    name: str
    category: str
    severity: str
    duration: str
    description: str
    red_flags: List[str]


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    speaker: str  # 'patient' or 'ai'
    text: str
    timestamp: str
    context: Dict[str, Any]


@dataclass
class MedicalScenario:
    """Complete medical scenario"""
    scenario_id: str
    patient: Patient
    primary_complaint: str
    symptoms: List[Symptom]
    triage_level: TriageLevel
    specialty: MedicalSpecialty
    urgency_score: int
    conversation: List[ConversationTurn]
    expected_outcomes: Dict[str, Any]
    metadata: Dict[str, Any]


class MedicalKnowledgeBase:
    """Knowledge base for medical data generation"""
    
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_db()
        self.medications_db = self._initialize_medications_db()
        self.conditions_db = self._initialize_conditions_db()
        self.conversation_templates = self._initialize_conversation_templates()
        
    def _initialize_symptoms_db(self) -> Dict[str, List[Symptom]]:
        """Initialize symptoms database"""
        return {
            "respiratory": [
                Symptom("Shortness of breath", "respiratory", "moderate", "2 days", 
                       "Difficulty breathing, especially during physical activity", []),
                Symptom("Persistent cough", "respiratory", "mild", "1 week", 
                       "Dry cough that won't go away", []),
                Symptom("Chest pain", "respiratory", "severe", "few hours", 
                       "Sharp pain when breathing", ["shortness_of_breath", "dizziness"]),
                Symptom("Wheezing", "respiratory", "moderate", "3 days", 
                       "High-pitched whistling sound when breathing", []),
            ],
            "cardiac": [
                Symptom("Chest pressure", "cardiac", "severe", "30 minutes", 
                       "Feeling of heaviness in chest", ["arm_pain", "shortness_of_breath", "sweating"]),
                Symptom("Palpitations", "cardiac", "moderate", "1 hour", 
                       "Irregular or rapid heartbeat", ["dizziness", "shortness_of_breath"]),
                Symptom("Arm pain", "cardiac", "moderate", "2 hours", 
                       "Pain radiating down left arm", ["chest_pain", "sweating"]),
            ],
            "gastrointestinal": [
                Symptom("Nausea", "gastrointestinal", "mild", "6 hours", 
                       "Feeling of sickness", []),
                Symptom("Abdominal pain", "gastrointestinal", "moderate", "4 hours", 
                       "Stomach cramps and discomfort", ["vomiting", "diarrhea"]),
                Symptom("Vomiting", "gastrointestinal", "moderate", "2 hours", 
                       "Forceful expulsion of stomach contents", ["dehydration"]),
            ],
            "neurological": [
                Symptom("Headache", "neurological", "moderate", "3 hours", 
                       "Persistent head pain", ["sensitivity_to_light", "nausea"]),
                Symptom("Dizziness", "neurological", "mild", "2 hours", 
                       "Feeling of spinning or imbalance", ["nausea"]),
                Symptom("Confusion", "neurological", "severe", "1 hour", 
                       "Difficulty thinking clearly", ["memory_issues"]),
            ]
        }
    
    def _initialize_medications_db(self) -> Dict[str, List[str]]:
        """Initialize medications database"""
        return {
            "cardiovascular": ["aspirin", "metoprolol", "lisinopril", "atorvastatin"],
            "respiratory": ["albuterol", "fluticasone", "montelukast", "theophylline"],
            "gastrointestinal": ["omeprazole", "lansoprazole", "simethicone", "bismuth"],
            "neurological": ["acetaminophen", "ibuprofen", "sumatriptan", "gabapentin"]
        }
    
    def _initialize_conditions_db(self) -> Dict[str, List[str]]:
        """Initialize common medical conditions"""
        return {
            "cardiovascular": ["hypertension", "coronary_artery_disease", "heart_failure", "arrhythmia"],
            "respiratory": ["asthma", "copd", "pneumonia", "bronchitis"],
            "gastrointestinal": ["gastroenteritis", "gastric_ulcer", "irritable_bowel_syndrome", "appendicitis"],
            "neurological": ["migraine", "tension_headache", "vertigo", "seizure_disorder"]
        }
    
    def _initialize_conversation_templates(self) -> Dict[str, Dict]:
        """Initialize conversation templates"""
        return {
            "patient_greeting": [
                "Hi, I'm not feeling well today.",
                "I'm having some health concerns.",
                "I need some medical advice.",
                "I'm experiencing some symptoms."
            ],
            "patient_description": [
                "I've been feeling {symptom} for {duration}.",
                "My {symptom} started {duration} ago.",
                "I've had {symptom} since {duration}.",
                "The {symptom} is getting worse."
            ],
            "ai_greeting": [
                "Hello! I'm here to help with your health concerns.",
                "Good day! I'm the medical assistant. How can I assist you today?",
                "Hi there! What brings you to talk with me today?"
            ],
            "ai_follow_up": [
                "Can you describe your symptoms in more detail?",
                "How long have you been experiencing this?",
                "On a scale of 1-10, how would you rate your pain?",
                "Are you currently taking any medications?",
                "Do you have any known allergies?",
                "Have you experienced this before?",
                "Are there any other symptoms you're noticing?"
            ],
            "ai_recommendation": [
                "Based on your symptoms, I recommend {recommendation}.",
                "Given your condition, it's important to {recommendation}.",
                "For your safety, please {recommendation}.",
                "I suggest that you {recommendation}."
            ]
        }


class SyntheticDataGenerator:
    """Main synthetic data generator"""
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.knowledge_base = MedicalKnowledgeBase()
        self.patient_id_counter = 1
        self.scenario_id_counter = 1
        
    def generate_patient(self, 
                        age_range: Tuple[int, int] = (18, 80),
                        gender_ratio: float = 0.5) -> Patient:
        """Generate a synthetic patient"""
        
        age = self.random.randint(age_range[0], age_range[1])
        age_group = self._get_age_group(age)
        gender = "male" if self.random.random() < gender_ratio else "female"
        
        # Generate medical history
        medical_history = self._generate_medical_history()
        
        # Generate current medications
        current_medications = self._generate_current_medications()
        
        # Generate allergies
        allergies = self._generate_allergies()
        
        # Generate vital signs
        vital_signs = self._generate_vital_signs(age)
        
        patient_id = f"patient_{self.patient_id_counter}"
        self.patient_id_counter += 1
        
        return Patient(
            id=patient_id,
            age=age,
            age_group=age_group,
            gender=gender,
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies,
            vital_signs=vital_signs
        )
    
    def generate_symptoms(self, 
                         category: Optional[str] = None,
                         num_symptoms: Tuple[int, int] = (1, 3)) -> List[Symptom]:
        """Generate synthetic symptoms"""
        
        if category is None:
            category = self.random.choice(list(self.knowledge_base.symptoms_db.keys()))
        
        available_symptoms = self.knowledge_base.symptoms_db[category]
        num = self.random.randint(num_symptoms[0], num_symptoms[1])
        
        selected_symptoms = self.random.sample(available_symptoms, min(num, len(available_symptoms)))
        
        # Create copies with slight variations
        symptoms = []
        for symptom in selected_symptoms:
            # Add some variation to duration
            if self.random.random() < 0.3:
                durations = ["1 day", "2 days", "1 week", "2 weeks"]
                symptom.duration = self.random.choice(durations)
            
            symptoms.append(symptom)
        
        return symptoms
    
    def generate_triage_level(self, symptoms: List[Symptom], 
                             patient_age: int) -> TriageLevel:
        """Generate appropriate triage level based on symptoms and patient"""
        
        # Check for red flags
        red_flag_count = 0
        for symptom in symptoms:
            if symptom.red_flags:
                red_flag_count += len(symptom.red_flags)
        
        # Age factor
        age_factor = 0
        if patient_age < 18 or patient_age > 75:
            age_factor = 1
        
        # Calculate urgency score
        urgency_score = 0
        
        for symptom in symptoms:
            if symptom.severity == "severe":
                urgency_score += 3
            elif symptom.severity == "moderate":
                urgency_score += 2
            else:
                urgency_score += 1
        
        urgency_score += red_flag_count
        urgency_score += age_factor
        
        # Map to triage level
        if urgency_score >= 6:
            return TriageLevel.IMMEDIATE
        elif urgency_score >= 4:
            return TriageLevel.EMERGENT
        elif urgency_score >= 3:
            return TriageLevel.URGENT
        elif urgency_score >= 2:
            return TriageLevel.LESS_URGENT
        else:
            return TriageLevel.NON_URGENT
    
    def generate_conversation(self, 
                            patient: Patient,
                            symptoms: List[Symptom],
                            triage_level: TriageLevel,
                            specialty: MedicalSpecialty) -> List[ConversationTurn]:
        """Generate synthetic medical conversation"""
        
        conversation = []
        timestamp = datetime.now()
        
        # Patient greeting
        greeting = self.random.choice(self.knowledge_base.conversation_templates["patient_greeting"])
        conversation.append(ConversationTurn(
            speaker="patient",
            text=greeting,
            timestamp=timestamp.isoformat(),
            context={"stage": "greeting"}
        ))
        timestamp += timedelta(seconds=30)
        
        # AI response
        ai_greeting = self.random.choice(self.knowledge_base.conversation_templates["ai_greeting"])
        conversation.append(ConversationTurn(
            speaker="ai",
            text=ai_greeting,
            timestamp=timestamp.isoformat(),
            context={"stage": "greeting_response"}
        ))
        timestamp += timedelta(seconds=45)
        
        # Symptom description
        primary_symptom = symptoms[0]
        description = self.random.choice(self.knowledge_base.conversation_templates["patient_description"]).format(
            symptom=primary_symptom.name.lower(),
            duration=primary_symptom.duration
        )
        conversation.append(ConversationTurn(
            speaker="patient",
            text=description,
            timestamp=timestamp.isoformat(),
            context={"stage": "chief_complaint", "symptom": primary_symptom.name}
        ))
        timestamp += timedelta(seconds=60)
        
        # AI follow-up questions
        num_followups = self.random.randint(2, 5)
        follow_ups_used = set()
        
        for i in range(num_followups):
            follow_up = self.random.choice(self.knowledge_base.conversation_templates["ai_follow_up"])
            while follow_up in follow_ups_used and len(follow_ups_used) < len(self.knowledge_base.conversation_templates["ai_follow_up"]):
                follow_up = self.random.choice(self.knowledge_base.conversation_templates["ai_follow_up"])
            
            follow_ups_used.add(follow_up)
            
            conversation.append(ConversationTurn(
                speaker="ai",
                text=follow_up,
                timestamp=timestamp.isoformat(),
                context={"stage": f"follow_up_{i+1}"}
            ))
            timestamp += timedelta(seconds=30)
            
            # Patient response (simulated)
            response = self._generate_patient_response(follow_up)
            conversation.append(ConversationTurn(
                speaker="patient",
                text=response,
                timestamp=timestamp.isoformat(),
                context={"stage": f"follow_up_response_{i+1}"}
            ))
            timestamp += timedelta(seconds=45)
        
        # AI recommendation
        recommendation = self._generate_recommendation(triage_level, specialty)
        ai_recommendation = self.random.choice(self.knowledge_base.conversation_templates["ai_recommendation"]).format(
            recommendation=recommendation
        )
        
        conversation.append(ConversationTurn(
            speaker="ai",
            text=ai_recommendation,
            timestamp=timestamp.isoformat(),
            context={"stage": "recommendation", "urgency": triage_level.value}
        ))
        
        return conversation
    
    def generate_scenario(self,
                         specialty: Optional[MedicalSpecialty] = None,
                         age_range: Tuple[int, int] = (18, 80),
                         triage_distribution: Optional[Dict[TriageLevel, float]] = None) -> MedicalScenario:
        """Generate complete medical scenario"""
        
        if specialty is None:
            specialty = self.random.choice(list(MedicalSpecialty))
        
        # Generate patient
        patient = self.generate_patient(age_range=age_range)
        
        # Generate symptoms based on specialty
        category_map = {
            MedicalSpecialty.CARDIOLOGY: "cardiac",
            MedicalSpecialty.RESPIRATORY: "respiratory",
            MedicalSpecialty.GASTROINTESTINAL: "gastrointestinal",
            MedicalSpecialty.NEUROLOGY: "neurological",
            MedicalSpecialty.GENERAL: self.random.choice(["respiratory", "cardiac", "gastrointestinal", "neurological"])
        }
        
        category = category_map.get(specialty, "respiratory")
        symptoms = self.generate_symptoms(category=category)
        
        # Generate triage level
        triage_level = self.generate_triage_level(symptoms, patient.age)
        
        # Adjust based on desired distribution
        if triage_distribution:
            if triage_level not in triage_distribution or self.random.random() > triage_distribution[triage_level]:
                triage_level = self.random.choice(list(TriageLevel))
        
        # Generate conversation
        conversation = self.generate_conversation(patient, symptoms, triage_level, specialty)
        
        # Generate expected outcomes
        expected_outcomes = self._generate_expected_outcomes(triage_level, specialty)
        
        # Generate metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "generator_version": "1.0",
            "complexity_score": len(symptoms) + (patient.age // 20),
            "specialty_mapping": specialty.value
        }
        
        scenario_id = f"scenario_{self.scenario_id_counter}"
        self.scenario_id_counter += 1
        
        primary_complaint = symptoms[0].name if symptoms else "General consultation"
        
        return MedicalScenario(
            scenario_id=scenario_id,
            patient=patient,
            primary_complaint=primary_complaint,
            symptoms=symptoms,
            triage_level=triage_level,
            specialty=specialty,
            urgency_score=triage_level.value,
            conversation=conversation,
            expected_outcomes=expected_outcomes,
            metadata=metadata
        )
    
    def _get_age_group(self, age: int) -> AgeGroup:
        """Get age group from age"""
        if age <= 2:
            return AgeGroup.INFANT
        elif age <= 12:
            return AgeGroup.CHILD
        elif age <= 17:
            return AgeGroup.ADOLESCENT
        elif age <= 25:
            return AgeGroup.YOUNG_ADULT
        elif age <= 64:
            return AgeGroup.ADULT
        else:
            return AgeGroup.GERIATRIC
    
    def _generate_medical_history(self) -> List[str]:
        """Generate medical history"""
        common_conditions = [
            "hypertension", "diabetes", "asthma", "allergies", "arthritis",
            "migraine", "depression", "anxiety", "hypothyroidism", "gerd"
        ]
        
        num_conditions = self.random.randint(0, 3)
        return self.random.sample(common_conditions, min(num_conditions, len(common_conditions)))
    
    def _generate_current_medications(self) -> List[str]:
        """Generate current medications"""
        all_medications = list(itertools.chain.from_iterable(self.knowledge_base.medications_db.values()))
        
        num_medications = self.random.randint(0, 4)
        return self.random.sample(all_medications, min(num_medications, len(all_medications)))
    
    def _generate_allergies(self) -> List[str]:
        """Generate allergies"""
        common_allergies = [
            "penicillin", "aspirin", "shellfish", "nuts", "dairy", "latex",
            "pollen", "dust", "pet dander", "sulfa drugs"
        ]
        
        num_allergies = self.random.randint(0, 2)
        return self.random.sample(common_allergies, min(num_allergies, len(common_allergies)))
    
    def _generate_vital_signs(self, age: int) -> Dict[str, float]:
        """Generate realistic vital signs"""
        base_bp_systolic = 120
        base_bp_diastolic = 80
        base_hr = 70
        
        # Age adjustments
        if age > 65:
            base_bp_systolic += 10
            base_hr += 5
        elif age < 18:
            base_bp_systolic = int(base_bp_systolic * 0.8)
            base_hr = int(base_hr * 1.2)
        
        return {
            "blood_pressure_systolic": base_bp_systolic + self.np_random.normal(0, 15),
            "blood_pressure_diastolic": base_bp_diastolic + self.np_random.normal(0, 10),
            "heart_rate": base_hr + self.np_random.normal(0, 10),
            "temperature": 98.6 + self.np_random.normal(0, 1),
            "oxygen_saturation": 98 + self.np_random.normal(0, 2)
        }
    
    def _generate_patient_response(self, question: str) -> str:
        """Generate appropriate patient response"""
        responses = {
            "scale": [
                "I'd say it's about a 7 out of 10.",
                "It's pretty bad, maybe 8 or 9.",
                "Not too bad, maybe a 3 or 4.",
                "It's moderate, I'd say 5 or 6."
            ],
            "medications": [
                "Yes, I take lisinopril for blood pressure.",
                "No, I'm not taking anything right now.",
                "I take metformin for diabetes.",
                "Just some vitamins and supplements."
            ],
            "allergies": [
                "Yes, I'm allergic to penicillin.",
                "No food or medication allergies that I know of.",
                "I'm allergic to shellfish.",
                "I get hives from latex."
            ],
            "duration": [
                "It's been getting worse over the past few days.",
                "Just started today, maybe this morning.",
                "It's been going on for about a week now.",
                "It comes and goes, but this time it's been persistent."
            ],
            "default": [
                "I'm not really sure.",
                "That's a good question.",
                "I'm worried about it.",
                "I hope it gets better soon."
            ]
        }
        
        if "scale" in question.lower() or "1-10" in question.lower():
            return self.random.choice(responses["scale"])
        elif "medication" in question.lower():
            return self.random.choice(responses["medications"])
        elif "allergic" in question.lower() or "allergy" in question.lower():
            return self.random.choice(responses["allergies"])
        elif "how long" in question.lower() or "when" in question.lower():
            return self.random.choice(responses["duration"])
        else:
            return self.random.choice(responses["default"])
    
    def _generate_recommendation(self, triage_level: TriageLevel, specialty: MedicalSpecialty) -> str:
        """Generate appropriate recommendation"""
        recommendations = {
            TriageLevel.IMMEDIATE: [
                "seek emergency medical care immediately",
                "call 911 or go to the emergency room right away",
                "get to the nearest emergency department immediately"
            ],
            TriageLevel.EMERGENT: [
                "go to the emergency department within the next 30 minutes",
                "contact your doctor immediately",
                "seek urgent medical care"
            ],
            TriageLevel.URGENT: [
                "schedule an appointment with your doctor within 24 hours",
                "consider visiting an urgent care center",
                "contact your healthcare provider soon"
            ],
            TriageLevel.LESS_URGENT: [
                "schedule an appointment with your doctor within a few days",
                "monitor your symptoms and contact your doctor if they worsen",
                "consider seeing your healthcare provider this week"
            ],
            TriageLevel.NON_URGENT: [
                "schedule a routine appointment with your doctor",
                "monitor your symptoms and make an appointment if concerned",
                "follow up with your healthcare provider during your next visit"
            ]
        }
        
        return self.random.choice(recommendations[triage_level])
    
    def _generate_expected_outcomes(self, triage_level: TriageLevel, specialty: MedicalSpecialty) -> Dict[str, Any]:
        """Generate expected outcomes for the scenario"""
        return {
            "triage_prediction": triage_level.value,
            "specialty_prediction": specialty.value,
            "urgency_score": triage_level.value,
            "recommended_actions": [
                self._generate_recommendation(triage_level, specialty)
            ],
            "red_flags_checked": any(symptom.red_flags for symptom in self.knowledge_base.symptoms_db.get(specialty.value, [])),
            "conversation_quality_score": self.random.uniform(0.7, 1.0)
        }
    
    def generate_dataset(self, 
                        num_scenarios: int = 100,
                        specialty_distribution: Optional[Dict[MedicalSpecialty, float]] = None,
                        **kwargs) -> List[MedicalScenario]:
        """Generate complete synthetic dataset"""
        
        scenarios = []
        
        for _ in range(num_scenarios):
            specialty = None
            if specialty_distribution:
                specialties = list(specialty_distribution.keys())
                weights = list(specialty_distribution.values())
                specialty = self.random.choices(specialties, weights=weights)[0]
            
            scenario = self.generate_scenario(specialty=specialty, **kwargs)
            scenarios.append(scenario)
        
        return scenarios


def save_scenarios_to_json(scenarios: List[MedicalScenario], filepath: str):
    """Save scenarios to JSON file"""
    data = {
        "generated_at": datetime.now().isoformat(),
        "num_scenarios": len(scenarios),
        "scenarios": [asdict(scenario) for scenario in scenarios]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_scenarios_to_csv(scenarios: List[MedicalScenario], filepath: str):
    """Save scenarios to CSV file"""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'scenario_id', 'patient_id', 'age', 'gender', 'age_group',
            'primary_complaint', 'triage_level', 'specialty', 'urgency_score',
            'num_symptoms', 'conversation_length', 'expected_outcomes'
        ])
        
        # Write data
        for scenario in scenarios:
            writer.writerow([
                scenario.scenario_id,
                scenario.patient.id,
                scenario.patient.age,
                scenario.patient.gender,
                scenario.patient.age_group.value,
                scenario.primary_complaint,
                scenario.triage_level.value,
                scenario.specialty.value,
                scenario.urgency_score,
                len(scenario.symptoms),
                len(scenario.conversation),
                json.dumps(scenario.expected_outcomes)
            ])


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator()
    
    # Generate a single scenario
    scenario = generator.generate_scenario()
    print(f"Generated scenario: {scenario.scenario_id}")
    print(f"Patient: {scenario.patient.age}-year-old {scenario.patient.gender}")
    print(f"Primary complaint: {scenario.primary_complaint}")
    print(f"Triage level: {scenario.triage_level}")
    
    # Generate multiple scenarios
    scenarios = generator.generate_dataset(num_scenarios=10)
    print(f"\nGenerated {len(scenarios)} scenarios")
    
    # Save to files
    save_scenarios_to_json(scenarios, "synthetic_medical_data.json")
    save_scenarios_to_csv(scenarios, "synthetic_medical_data.csv")