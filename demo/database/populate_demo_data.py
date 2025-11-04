#!/usr/bin/env python3
"""
Demo Database Population Script
Generates HIPAA-compliant synthetic medical data for demonstration purposes
"""

import sqlite3
import hashlib
import json
import random
from datetime import datetime, timedelta
from faker import Faker
import uuid

# Initialize Faker for generating synthetic data
fake = Faker()
Faker.seed(42)  # For reproducible demo data

class DemoDatabasePopulator:
    def __init__(self, db_path: str = "demo.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to the demo database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def hash_password(self, password: str) -> str:
        """Hash password for demo users"""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000).hex()
        
    def generate_synthetic_patients(self, count: int = 10):
        """Generate synthetic patient data"""
        patients = []
        for i in range(count):
            # Generate basic patient info
            patient_id = self._get_next_id('patients')
            user_id = self._get_next_id('users')
            mrn = f"DEMO{patient_id:06d}"
            
            # Generate realistic patient data
            age = random.randint(18, 85)
            birth_date = fake.date_of_birth(minimum_age=age, maximum_age=age).strftime('%Y-%m-%d')
            gender = random.choice(['male', 'female'])
            
            patient_data = {
                'user_id': user_id,
                'medical_record_number': mrn,
                'date_of_birth': birth_date,
                'gender': gender,
                'blood_type': random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']),
                'height_cm': random.randint(150, 190),
                'weight_kg': random.uniform(50.0, 100.0),
                'emergency_contact_name': f"{fake.first_name()} {fake.last_name()}",
                'emergency_contact_phone': fake.phone_number(),
                'primary_physician_id': self._get_demo_nurse_id(),
                'insurance_provider': random.choice(['Blue Shield', 'Aetna', 'Cigna', 'UnitedHealth', 'Kaiser']),
                'policy_number': f"POL{random.randint(100000, 999999)}"
            }
            patients.append(patient_data)
            
        return patients
        
    def _get_next_id(self, table: str) -> int:
        """Get next available ID for table"""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        return count + 1
        
    def _get_demo_nurse_id(self) -> int:
        """Get ID of a demo nurse"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM users WHERE role = 'nurse' LIMIT 1")
        result = cursor.fetchone()
        return result[0] if result else 1
        
    def populate_users(self):
        """Populate demo users"""
        cursor = self.conn.cursor()
        
        # Create additional demo users
        demo_users = [
            {
                'email': f'patient.{fake.last_name().lower()}@demo.medai.com',
                'password': 'DemoPatient789!',
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'role': 'patient',
                'session_id': f'DEMO_PATIENT_{random.randint(3, 100):03d}'
            }
            for _ in range(8)
        ]
        
        # Add more nurses
        demo_users.extend([
            {
                'email': f'nurse.{fake.last_name().lower()}@demo.medai.com',
                'password': 'DemoNurse456!',
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'role': 'nurse',
                'session_id': f'DEMO_NURSE_{random.randint(3, 100):03d}'
            }
            for _ in range(3)
        ])
        
        for user_data in demo_users:
            password_hash = self.hash_password(user_data['password'])
            cursor.execute("""
                INSERT OR IGNORE INTO users (email, password_hash, first_name, last_name, role, demo_session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_data['email'],
                password_hash,
                user_data['first_name'],
                user_data['last_name'],
                user_data['role'],
                user_data['session_id']
            ))
            
    def populate_patients(self):
        """Populate synthetic patient records"""
        cursor = self.conn.cursor()
        patients = self.generate_synthetic_patients(10)
        
        for patient_data in patients:
            cursor.execute("""
                INSERT INTO patients (
                    user_id, medical_record_number, date_of_birth, gender, blood_type,
                    height_cm, weight_kg, emergency_contact_name, emergency_contact_phone,
                    primary_physician_id, insurance_provider, policy_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_data['user_id'],
                patient_data['medical_record_number'],
                patient_data['date_of_birth'],
                patient_data['gender'],
                patient_data['blood_type'],
                patient_data['height_cm'],
                patient_data['weight_kg'],
                patient_data['emergency_contact_name'],
                patient_data['emergency_contact_phone'],
                patient_data['primary_physician_id'],
                patient_data['insurance_provider'],
                patient_data['policy_number']
            ))
            
    def populate_medical_conditions(self):
        """Populate medical conditions for patients"""
        cursor = self.conn.cursor()
        
        # Common medical conditions
        conditions = [
            ('Type 2 Diabetes Mellitus', 'E11.9'),
            ('Essential Hypertension', 'I10'),
            ('Hyperlipidemia', 'E78.5'),
            ('Coronary Artery Disease', 'I25.10'),
            ('Osteoarthritis', 'M19.90'),
            ('Depression', 'F32.9'),
            ('Anxiety Disorder', 'F41.9'),
            ('Chronic Kidney Disease', 'N18.6'),
            ('Asthma', 'J45.90'),
            ('Gastroesophageal Reflux Disease', 'K21.9')
        ]
        
        # Get all patient IDs
        cursor.execute("SELECT id FROM patients")
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        # Assign conditions to patients
        for patient_id in patient_ids:
            # Each patient gets 1-3 conditions
            num_conditions = random.randint(1, 3)
            selected_conditions = random.sample(conditions, num_conditions)
            
            for condition_name, icd_code in selected_conditions:
                diagnosed_date = fake.date_between(start_date='-2y', end_date='today')
                severity = random.choice(['mild', 'moderate', 'severe'])
                status = random.choice(['active', 'chronic', 'resolved'])
                
                cursor.execute("""
                    INSERT INTO medical_conditions (
                        patient_id, condition_name, icd10_code, diagnosed_date,
                        severity, status, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id, condition_name, icd_code, diagnosed_date,
                    severity, status,
                    f"Generated demo condition: {condition_name}"
                ))
                
    def populate_vital_signs(self):
        """Generate realistic vital signs data"""
        cursor = self.conn.cursor()
        
        # Get all patient IDs
        cursor.execute("SELECT id FROM patients")
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        # Generate vital signs for the past 30 days
        for patient_id in patient_ids:
            for days_ago in range(30):
                record_date = datetime.now() - timedelta(days=days_ago)
                
                # Generate realistic vital signs
                vitals = self._generate_realistic_vitals()
                
                cursor.execute("""
                    INSERT INTO vital_signs (
                        patient_id, blood_pressure_systolic, blood_pressure_diastolic,
                        heart_rate, temperature, respiratory_rate, oxygen_saturation,
                        glucose_level, weight_kg, bmi, recorded_at, source, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id,
                    vitals['bp_systolic'],
                    vitals['bp_diastolic'],
                    vitals['heart_rate'],
                    vitals['temperature'],
                    vitals['respiratory_rate'],
                    vitals['oxygen_saturation'],
                    vitals['glucose_level'],
                    vitals['weight'],
                    vitals['bmi'],
                    record_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'simulation',
                    f"Demo vital signs generated on {record_date.strftime('%Y-%m-%d')}"
                ))
                
    def _generate_realistic_vitals(self):
        """Generate realistic vital signs values"""
        return {
            'bp_systolic': random.randint(110, 140),
            'bp_diastolic': random.randint(70, 90),
            'heart_rate': random.randint(60, 100),
            'temperature': round(random.uniform(36.1, 37.2), 1),
            'respiratory_rate': random.randint(12, 20),
            'oxygen_saturation': random.randint(95, 100),
            'glucose_level': round(random.uniform(80, 180), 1),
            'weight': round(random.uniform(50.0, 100.0), 1),
            'bmi': round(random.uniform(18.5, 30.0), 1)
        }
        
    def populate_medications(self):
        """Generate medication prescriptions"""
        cursor = self.conn.cursor()
        
        # Common medications
        medications = [
            ('Metformin 500mg', '500mg', 'twice daily'),
            ('Lisinopril 10mg', '10mg', 'once daily'),
            ('Atorvastatin 20mg', '20mg', 'once daily'),
            ('Aspirin 81mg', '81mg', 'once daily'),
            ('Omeprazole 20mg', '20mg', 'once daily'),
            ('Levothyroxine 50mcg', '50mcg', 'once daily'),
            ('Amlodipine 5mg', '5mg', 'once daily'),
            ('Metoprolol 50mg', '50mg', 'twice daily')
        ]
        
        # Get all patient IDs
        cursor.execute("SELECT id FROM patients")
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        for patient_id in patient_ids:
            # Each patient gets 1-4 medications
            num_medications = random.randint(1, 4)
            selected_meds = random.sample(medications, num_medications)
            
            for med_name, dosage, frequency in selected_meds:
                start_date = fake.date_between(start_date='-6m', end_date='today')
                status = random.choice(['active', 'active', 'active', 'completed'])  # Mostly active
                prescriber_id = self._get_demo_nurse_id()
                
                cursor.execute("""
                    INSERT INTO medications (
                        patient_id, medication_name, dosage, frequency,
                        start_date, prescribed_by, status, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id, med_name, dosage, frequency,
                    start_date, prescriber_id, status,
                    f"Demo prescription: {med_name}"
                ))
                
    def populate_lab_results(self):
        """Generate laboratory test results"""
        cursor = self.conn.cursor()
        
        # Common lab tests
        lab_tests = [
            ('Hemoglobin A1c', '4548-4', 6.5, '%', 4.0, 7.0),
            ('Glucose, Fasting', '2345-7', 100, 'mg/dL', 70, 100),
            ('Cholesterol, Total', '2093-3', 200, 'mg/dL', 0, 200),
            ('HDL Cholesterol', '2085-9', 50, 'mg/dL', 40, 1000),
            ('LDL Cholesterol', '2089-1', 120, 'mg/dL', 0, 100),
            ('Creatinine', '2160-0', 1.0, 'mg/dL', 0.7, 1.3),
            ('eGFR', '33914-3', 90, 'mL/min/1.73m²', 90, 1000),
            ('Potassium', '2823-3', 4.0, 'mmol/L', 3.5, 5.0),
            ('Sodium', '2951-2', 140, 'mmol/L', 135, 145)
        ]
        
        # Get all patient IDs
        cursor.execute("SELECT id FROM patients")
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        for patient_id in patient_ids:
            # Each patient gets 3-6 lab tests
            num_tests = random.randint(3, 6)
            selected_tests = random.sample(lab_tests, num_tests)
            
            for test_name, test_code, value, unit, ref_min, ref_max in selected_tests:
                collected_date = fake.date_between(start_date='-1m', end_date='today')
                ordered_by = self._get_demo_nurse_id()
                
                # Determine status based on value vs reference range
                if ref_min <= value <= ref_max:
                    status = 'normal'
                elif value < ref_min * 0.8 or value > ref_max * 1.2:
                    status = 'critical'
                else:
                    status = 'abnormal'
                
                cursor.execute("""
                    INSERT INTO lab_results (
                        patient_id, test_name, test_code, value, unit,
                        reference_range_min, reference_range_max, status,
                        ordered_by, collected_date, reported_date, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id, test_name, test_code, value, unit,
                    ref_min, ref_max, status, ordered_by, collected_date,
                    collected_date + timedelta(days=1),
                    f"Demo lab result for {test_name}"
                ))
                
    def populate_demo_scenarios(self):
        """Create demo scenarios for presentations"""
        scenarios = [
            {
                'name': 'Diabetes Management - John Smith',
                'type': 'diabetes',
                'description': 'Comprehensive diabetes monitoring and management for a 45-year-old male patient',
                'patient_id': 1,
                'steps': [
                    'Review recent glucose readings',
                    'Assess medication adherence',
                    'Analyze HbA1c trends',
                    'Provide dietary recommendations',
                    'Schedule follow-up appointment'
                ],
                'outcomes': ['Improved glucose control', 'Medication optimization', 'Patient education'],
                'duration': 15
            },
            {
                'name': 'Hypertension Monitoring - Emily Davis',
                'type': 'hypertension',
                'description': 'Blood pressure monitoring and cardiovascular risk assessment',
                'patient_id': 2,
                'steps': [
                    'Analyze blood pressure trends',
                    'Review cardiovascular medications',
                    'Assess lifestyle factors',
                    'Calculate cardiovascular risk',
                    'Update treatment plan'
                ],
                'outcomes': ['Blood pressure stabilization', 'Risk reduction', 'Medication adjustment'],
                'duration': 12
            },
            {
                'name': 'Chest Pain Assessment - Michael Johnson',
                'type': 'chest_pain',
                'description': 'Emergency triage and chest pain evaluation with risk stratification',
                'patient_id': 3,
                'steps': [
                    'Symptom assessment',
                    'ECG interpretation',
                    'Risk stratification',
                    'Emergency protocol activation',
                    'Specialist consultation'
                ],
                'outcomes': ['Risk assessment completed', 'Emergency protocols activated', 'Specialist referral'],
                'duration': 10
            }
        ]
        
        cursor = self.conn.cursor()
        for scenario in scenarios:
            cursor.execute("""
                INSERT INTO demo_scenarios (
                    scenario_name, scenario_type, description, patient_id,
                    workflow_steps, expected_outcomes, difficulty_level,
                    estimated_duration_minutes, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scenario['name'],
                scenario['type'],
                scenario['description'],
                scenario['patient_id'],
                json.dumps(scenario['steps']),
                json.dumps(scenario['outcomes']),
                'intermediate',
                scenario['duration'],
                True
            ))
            
    def run_population(self):
        """Run complete database population"""
        print("Starting demo database population...")
        
        self.connect()
        
        try:
            # Load schema
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(current_dir, 'demo_schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            self.conn.executescript(schema_sql)
            
            print("✓ Database schema created")
            
            # Populate data
            print("Populating users...")
            self.populate_users()
            print(f"✓ Users populated")
            
            print("Populating patients...")
            self.populate_patients()
            print(f"✓ Patients populated")
            
            print("Populating medical conditions...")
            self.populate_medical_conditions()
            print(f"✓ Medical conditions populated")
            
            print("Populating vital signs...")
            self.populate_vital_signs()
            print(f"✓ Vital signs populated")
            
            print("Populating medications...")
            self.populate_medications()
            print(f"✓ Medications populated")
            
            print("Populating lab results...")
            self.populate_lab_results()
            print(f"✓ Lab results populated")
            
            print("Creating demo scenarios...")
            self.populate_demo_scenarios()
            print(f"✓ Demo scenarios created")
            
            self.conn.commit()
            print("\n✓ Demo database population completed successfully!")
            
        except Exception as e:
            print(f"✗ Error during population: {e}")
            self.conn.rollback()
            raise
        finally:
            self.close()

if __name__ == "__main__":
    populator = DemoDatabasePopulator("demo.db")
    populator.run_population()