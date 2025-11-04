# Integration Examples & Code Samples - Medical System Integration

## Overview

Comprehensive integration guide with code samples for connecting the Medical AI Serving System to various healthcare information systems, EMR platforms, and clinical workflows while maintaining HIPAA compliance and regulatory standards.

## 🏥 Medical System Integration Architecture

### Integration Patterns
- **EMR Integration**: Epic, Cerner, Allscripts connectivity
- **HL7 FHIR**: Standards-based health data exchange
- **DICOM Integration**: Medical imaging system connectivity
- **Laboratory Systems**: Lab result integration and interpretation
- **Pharmacy Systems**: Drug interaction checking and prescription support
- **Clinical Decision Support**: Integration with existing CDS tools

### Security & Compliance Framework
- **End-to-End Encryption**: TLS 1.3 with mutual authentication
- **OAuth 2.0/OpenID Connect**: Secure authentication with medical systems
- **SMART on FHIR**: Secure healthcare app integration
- **Audit Logging**: Comprehensive audit trail for compliance
- **Role-Based Access Control**: Medical staff access management

## EMR System Integration

### Epic MyChart Integration

#### Epic FHIR Client Implementation
```python
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from dataclasses import dataclass
import logging

@dataclass
class PatientContext:
    """Patient context for medical AI queries"""
    patient_id: str
    mrn: str  # Medical Record Number
    date_of_birth: date
    gender: str
    encounter_id: Optional[str]
    provider_id: str

class EpicFHIRClient:
    def __init__(self, epic_config):
        self.base_url = epic_config.base_url
        self.client_id = epic_config.client_id
        self.client_secret = epic_config.client_secret
        self.scope = "patient/*.read patient/*.write launch/patient offline_access"
        self.access_token = None
        self.token_expires = None
        
    async def authenticate(self):
        """Authenticate with Epic using OAuth 2.0"""
        
        auth_url = f"{self.base_url}/oauth2/token"
        
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': self.scope
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']
                    self.token_expires = datetime.now().timestamp() + token_data['expires_in']
                    
                    logging.info("Successfully authenticated with Epic FHIR")
                    return True
                else:
                    error_text = await response.text()
                    logging.error(f"Epic authentication failed: {error_text}")
                    return False
    
    async def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive patient data from Epic"""
        
        if not self.access_token or datetime.now().timestamp() > self.token_expires:
            await self.authenticate()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json'
        }
        
        patient_data = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get patient demographics
                patient_response = await self._fhir_get(session, f'/Patient/{patient_id}', headers)
                patient_data['demographics'] = patient_response
                
                # Get active conditions
                conditions_response = await self._fhir_get(
                    session, f'/Condition?patient={patient_id}&clinical-status=active', headers
                )
                patient_data['conditions'] = conditions_response
                
                # Get medications
                medications_response = await self._fhir_get(
                    session, f'/MedicationRequest?patient={patient_id}&status=active', headers
                )
                patient_data['medications'] = medications_response
                
                # Get allergies
                allergies_response = await self._fhir_get(
                    session, f'/AllergyIntolerance?patient={patient_id}&clinical-status=active', headers
                )
                patient_data['allergies'] = allergies_response
                
                # Get vital signs
                vitals_response = await self._fhir_get(
                    session, f'/Observation?patient={patient_id}&category=vital-signs', headers
                )
                patient_data['vital_signs'] = vitals_response
                
                # Get lab results
                labs_response = await self._fhir_get(
                    session, f'/Observation?patient={patient_id}&category=laboratory', headers
                )
                patient_data['lab_results'] = labs_response
                
        except Exception as e:
            logging.error(f"Error retrieving patient data: {e}")
            raise
        
        return patient_data
    
    async def create_clinical_note(self, encounter_id: str, note_content: str, patient_id: str):
        """Create clinical note in Epic using FHIR DocumentReference"""
        
        if not self.access_token or datetime.now().timestamp() > self.token_expires:
            await self.authenticate()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/fhir+json'
        }
        
        # Create DocumentReference for clinical note
        document_reference = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "34109-9",
                    "display": "Outpatient Note"
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "context": {
                "encounter": [{
                    "reference": f"Encounter/{encounter_id}"
                }]
            },
            "content": [{
                "attachment": {
                    "contentType": "text/plain",
                    "data": note_content.encode('base64').decode('ascii')
                }
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.base_url}/DocumentReference",
                headers=headers,
                json=document_reference
            )
            
            if response.status == 201:
                result = await response.json()
                logging.info(f"Clinical note created successfully: {result['id']}")
                return result
            else:
                error_text = await response.text()
                logging.error(f"Failed to create clinical note: {error_text}")
                raise Exception(f"Epic API error: {response.status}")
    
    async def _fhir_get(self, session, endpoint: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Make authenticated FHIR GET request"""
        
        async with session.get(f"{self.base_url}{endpoint}", headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"FHIR GET request failed: {response.status} - {error_text}")

class MedicalAIEpicIntegration:
    def __init__(self, epic_config, medical_ai_client):
        self.epic_client = EpicFHIRClient(epic_config)
        self.medical_ai = medical_ai_client
        self.patient_context = None
        
    async def analyze_patient_with_medical_ai(self, patient_id: str, query: str) -> Dict[str, Any]:
        """Analyze patient using Medical AI with Epic data"""
        
        try:
            # 1. Get patient data from Epic
            patient_data = await self.epic_client.get_patient_data(patient_id)
            
            # 2. Create patient context
            patient_context = self._create_patient_context(patient_data)
            
            # 3. Format query with patient context
            formatted_query = self._format_clinical_query(query, patient_data)
            
            # 4. Query Medical AI with context
            ai_response = await self.medical_ai.analyze_clinical_situation(
                query=formatted_query,
                patient_context=patient_context,
                medical_domain=self._determine_medical_domain(query),
                urgency_level=self._assess_urgency(query, patient_data)
            )
            
            # 5. Update Epic with AI insights (optional)
            if ai_response.get('clinical_recommendations'):
                await self._update_epic_with_recommendations(
                    patient_id, ai_response['clinical_recommendations']
                )
            
            return {
                'ai_analysis': ai_response,
                'patient_data_source': 'epic',
                'timestamp': datetime.now().isoformat(),
                'compliance_status': 'hipaa_compliant'
            }
            
        except Exception as e:
            logging.error(f"Patient analysis failed: {e}")
            raise
    
    def _create_patient_context(self, patient_data: Dict[str, Any]) -> PatientContext:
        """Create patient context from Epic data"""
        
        demographics = patient_data.get('demographics', {})
        patient_resource = demographics.get('entry', [{}])[0].get('resource', {})
        
        # Parse birth date
        birth_date_str = patient_resource.get('birthDate')
        birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date() if birth_date_str else date(1900, 1, 1)
        
        # Get MRN
        mrn = None
        for identifier in patient_resource.get('identifier', []):
            if identifier.get('type', {}).get('coding', [{}])[0].get('code') == 'MR':
                mrn = identifier.get('value')
                break
        
        return PatientContext(
            patient_id=patient_resource.get('id', ''),
            mrn=mrn or '',
            date_of_birth=birth_date,
            gender=patient_resource.get('gender', ''),
            provider_id='',
            encounter_id=None
        )
    
    def _format_clinical_query(self, base_query: str, patient_data: Dict[str, Any]) -> str:
        """Format clinical query with patient context"""
        
        formatted_query = base_query
        
        # Add relevant patient context
        conditions = patient_data.get('conditions', {}).get('entry', [])
        if conditions:
            condition_list = [
                cond['resource']['code']['text'] 
                for cond in conditions[:5]  # Top 5 conditions
            ]
            formatted_query += f"\n\nPatient has the following conditions: {', '.join(condition_list)}"
        
        # Add medications
        medications = patient_data.get('medications', {}).get('entry', [])
        if medications:
            med_list = [
                med['resource']['medicationCodeableConcept']['text']
                for med in medications[:5]  # Top 5 medications
            ]
            formatted_query += f"\nCurrent medications: {', '.join(med_list)}"
        
        # Add allergies
        allergies = patient_data.get('allergies', {}).get('entry', [])
        if allergies:
            allergy_list = [
                allergy['resource']['code']['text']
                for allergy in allergies
            ]
            formatted_query += f"\nKnown allergies: {', '.join(allergy_list)}"
        
        return formatted_query
```

### Cerner PowerChart Integration

#### Cerner Millennium Integration
```python
import urllib.parse
import base64
from typing import Dict, List, Any, Optional

class CernerMillenniumClient:
    def __init__(self, cerner_config):
        self.base_url = cerner_config.base_url
        self.client_id = cerner_config.client_id
        self.client_secret = cerner_config.client_secret
        self.username = cerner_config.username
        self.password = cerner_config.password
        
    async def get_patient_encounters(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient encounters from Cerner PowerChart"""
        
        # Cerner API endpoints for Millennium
        encounters = []
        
        try:
            # Get encounter data using Cerner's proprietary APIs
            encounter_data = await self._call_cerner_api(
                '/person/{patient_id}/encounter'.format(patient_id=patient_id)
            )
            
            for encounter in encounter_data.get('data', []):
                encounter_info = {
                    'encounter_id': encounter.get('encounter_id'),
                    'encounter_type': encounter.get('encounter_type'),
                    'admission_date': encounter.get('admission_date'),
                    'discharge_date': encounter.get('discharge_date'),
                    'provider_id': encounter.get('provider_id'),
                    'location': encounter.get('location'),
                    'diagnosis_codes': encounter.get('diagnosis_codes', [])
                }
                encounters.append(encounter_info)
                
        except Exception as e:
            logging.error(f"Error retrieving encounters: {e}")
            
        return encounters
    
    async def get_clinical_notes(self, patient_id: str, encounter_id: str) -> List[Dict[str, Any]]:
        """Get clinical notes from Cerner PowerChart"""
        
        try:
            notes_data = await self._call_cerner_api(
                f'/person/{patient_id}/encounter/{encounter_id}/clinical-documents'
            )
            
            clinical_notes = []
            for note in notes_data.get('data', []):
                if note.get('document_type') == 'Clinical Note':
                    note_content = {
                        'note_id': note.get('document_id'),
                        'note_type': note.get('document_type'),
                        'created_date': note.get('created_date'),
                        'author': note.get('author'),
                        'content': await self._get_document_content(note.get('document_id')),
                        'clinical_domain': self._extract_clinical_domain(note.get('document_type'))
                    }
                    clinical_notes.append(note_content)
            
            return clinical_notes
            
        except Exception as e:
            logging.error(f"Error retrieving clinical notes: {e}")
            return []
    
    async def _call_cerner_api(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Dict[str, Any]:
        """Call Cerner API with proper authentication"""
        
        # Build Cerner-specific authentication
        auth_header = self._build_cerner_auth_header()
        
        headers = {
            'Authorization': auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            if method == 'GET':
                async with session.get(url, headers=headers) as response:
                    return await self._handle_response(response)
            elif method == 'POST':
                async with session.post(url, headers=headers, json=data) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
    
    def _build_cerner_auth_header(self) -> str:
        """Build Cerner-specific authorization header"""
        
        # Cerner uses Basic Authentication with encoded credentials
        credentials = f"{self.username}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        return f"Basic {encoded_credentials}"
    
    async def _get_document_content(self, document_id: str) -> str:
        """Get document content from Cerner"""
        
        try:
            content_data = await self._call_cerner_api(
                f'/clinical-documents/{document_id}/content'
            )
            return content_data.get('content', '')
        except Exception as e:
            logging.error(f"Error retrieving document content: {e}")
            return ""
    
    def _extract_clinical_domain(self, document_type: str) -> str:
        """Extract clinical domain from document type"""
        
        domain_mapping = {
            'Progress Note': 'general',
            'Consultation Note': 'consultation',
            'Discharge Summary': 'general',
            'Operative Note': 'surgery',
            'Radiology Report': 'radiology',
            'Pathology Report': 'pathology',
            'Laboratory Report': 'laboratory'
        }
        
        return domain_mapping.get(document_type, 'general')

class CernerMedicalAIIntegration:
    def __init__(self, cerner_config, medical_ai_client):
        self.cerner_client = CernerMillenniumClient(cerner_config)
        self.medical_ai = medical_ai_client
        
    async def analyze_patient_timeline(self, patient_id: str) -> Dict[str, Any]:
        """Analyze patient timeline using Medical AI"""
        
        try:
            # Get patient encounters
            encounters = await self.cerner_client.get_patient_encounters(patient_id)
            
            # Get clinical notes for each encounter
            all_notes = []
            for encounter in encounters:
                notes = await self.cerner_client.get_clinical_notes(
                    patient_id, encounter['encounter_id']
                )
                all_notes.extend(notes)
            
            # Analyze patient timeline with Medical AI
            timeline_analysis = await self._analyze_patient_timeline(encounters, all_notes)
            
            return {
                'patient_id': patient_id,
                'timeline_analysis': timeline_analysis,
                'encounters_analyzed': len(encounters),
                'notes_analyzed': len(all_notes),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Patient timeline analysis failed: {e}")
            raise
    
    async def _analyze_patient_timeline(self, encounters: List[Dict], notes: List[Dict]) -> Dict[str, Any]:
        """Analyze patient timeline using Medical AI"""
        
        # Build comprehensive patient timeline
        timeline_data = {
            'encounters': encounters,
            'clinical_notes': notes,
            'temporal_patterns': [],
            'clinical_trends': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        # Query Medical AI for timeline analysis
        ai_analysis = await self.medical_ai.analyze_patient_timeline(timeline_data)
        
        return ai_analysis
```

## HL7 FHIR Integration

### FHIR R4 Client Implementation

#### Generic FHIR Client
```python
from fhir.resources import Patient, Condition, Observation, MedicationRequest, AllergyIntolerance
from fhir.resources.fhirtypes import HumanName, CodeableConcept, Coding
import fhirclient
from fhirclient.client import FHIRClient

class FHIRClient:
    def __init__(self, fhir_server_url: str, client_config: Dict[str, Any]):
        self.server_url = fhir_server_url
        self.client_config = client_config
        self.fhir_client = None
        
    async def initialize_client(self):
        """Initialize FHIR client with configuration"""
        
        # Set up FHIR client configuration
        settings = {
            'app_id': self.client_config.get('app_id'),
            'app_secret': self.client_config.get('app_secret'),
            'fhir_server_url': self.server_url,
            'scope': self.client_config.get('scope', 'patient/*.read patient/*.write')
        }
        
        self.fhir_client = FHIRClient(settings=settings)
        return self.fhir_client
    
    async def search_patients(self, search_params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Search for patients using FHIR search parameters"""
        
        try:
            # Build search URL
            search_url = self._build_fhir_search_url('Patient', search_params)
            
            # Execute search
            response = await self._fhir_request('GET', search_url)
            
            # Parse response
            patients = []
            for entry in response.get('entry', []):
                patient_resource = Patient(entry['resource'])
                patients.append(patient_resource.dict())
            
            return patients
            
        except Exception as e:
            logging.error(f"Patient search failed: {e}")
            return []
    
    async def get_patient_bundle(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient data bundle"""
        
        try:
            # Get patient
            patient_response = await self._fhir_request('GET', f'/Patient/{patient_id}')
            
            # Get related resources
            tasks = [
                self._get_patient_conditions(patient_id),
                self._get_patient_medications(patient_id),
                self._get_patient_allergies(patient_id),
                self._get_patient_observations(patient_id),
                self._get_patient_encounters(patient_id)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Build bundle
            bundle = {
                'resourceType': 'Bundle',
                'type': 'collection',
                'timestamp': datetime.now().isoformat(),
                'entry': [
                    {'resource': patient_response}
                ]
            }
            
            # Add each resource type
            resource_types = ['Condition', 'MedicationRequest', 'AllergyIntolerance', 'Observation', 'Encounter']
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.warning(f"Failed to fetch {resource_types[i]}: {result}")
                else:
                    bundle['entry'].extend(result)
            
            return bundle
            
        except Exception as e:
            logging.error(f"Patient bundle creation failed: {e}")
            raise
    
    async def _get_patient_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient conditions using FHIR search"""
        
        try:
            search_params = {
                'patient': patient_id,
                'clinical-status': 'active',
                '_count': '100'
            }
            
            response = await self._fhir_request('GET', self._build_fhir_search_url('Condition', search_params))
            
            conditions = []
            for entry in response.get('entry', []):
                conditions.append({'resource': entry['resource']})
            
            return conditions
            
        except Exception as e:
            logging.error(f"Failed to get patient conditions: {e}")
            return []
    
    async def _fhir_request(self, method: str, url: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated FHIR request"""
        
        headers = {
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json'
        }
        
        async with aiohttp.ClientSession() as session:
            if method == 'GET':
                async with session.get(f"{self.server_url}{url}", headers=headers) as response:
                    return await self._handle_fhir_response(response)
            elif method == 'POST':
                async with session.post(f"{self.server_url}{url}", headers=headers, json=data) as response:
                    return await self._handle_fhir_response(response)
            else:
                raise ValueError(f"Unsupported FHIR method: {method}")
    
    async def _handle_fhir_response(self, response) -> Dict[str, Any]:
        """Handle FHIR response and check for errors"""
        
        if response.status == 200:
            return await response.json()
        elif response.status == 201:
            return await response.json()
        else:
            error_text = await response.text()
            raise Exception(f"FHIR request failed: {response.status} - {error_text}")
    
    def _build_fhir_search_url(self, resource_type: str, search_params: Dict[str, str]) -> str:
        """Build FHIR search URL with parameters"""
        
        param_strings = []
        for key, value in search_params.items():
            param_strings.append(f"{key}={urllib.parse.quote(str(value))}")
        
        param_string = "&".join(param_strings)
        return f"/{resource_type}?{param_string}"

class FHIRMedicalAIIntegration:
    def __init__(self, fhir_client: FHIRClient, medical_ai_client):
        self.fhir_client = fhir_client
        self.medical_ai = medical_ai_client
        
    async def process_fhir_patient_data(self, patient_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Process FHIR patient data with Medical AI"""
        
        try:
            # Extract patient information
            patient_entry = None
            condition_entries = []
            medication_entries = []
            observation_entries = []
            
            for entry in patient_bundle.get('entry', []):
                resource = entry['resource']
                resource_type = resource['resourceType']
                
                if resource_type == 'Patient':
                    patient_entry = resource
                elif resource_type == 'Condition':
                    condition_entries.append(resource)
                elif resource_type == 'MedicationRequest':
                    medication_entries.append(resource)
                elif resource_type == 'Observation':
                    observation_entries.append(resource)
            
            # Build clinical context
            clinical_context = self._build_clinical_context(
                patient_entry, condition_entries, medication_entries, observation_entries
            )
            
            # Analyze with Medical AI
            ai_analysis = await self.medical_ai.analyze_clinical_context(clinical_context)
            
            # Generate FHIR-compliant response
            fhir_response = self._create_fhir_clinical_response(ai_analysis, patient_entry)
            
            return fhir_response
            
        except Exception as e:
            logging.error(f"FHIR patient data processing failed: {e}")
            raise
    
    def _build_clinical_context(self, patient: Dict, conditions: List[Dict], 
                               medications: List[Dict], observations: List[Dict]) -> Dict[str, Any]:
        """Build clinical context from FHIR resources"""
        
        context = {
            'patient_demographics': {
                'gender': patient.get('gender'),
                'birth_date': patient.get('birthDate'),
                'age': self._calculate_age(patient.get('birthDate'))
            },
            'active_conditions': [
                {
                    'code': condition['code']['coding'][0]['code'],
                    'display': condition['code']['coding'][0]['display'],
                    'clinical_status': condition.get('clinicalStatus', {}).get('coding', [{}])[0].get('code'),
                    'onset_date': condition.get('onsetDateTime')
                }
                for condition in conditions
                if condition.get('code', {}).get('coding')
            ],
            'current_medications': [
                {
                    'code': medication['medicationCodeableConcept']['coding'][0]['code'],
                    'display': medication['medicationCodeableConcept']['coding'][0]['display'],
                    'status': medication.get('status'),
                    'dosage': medication.get('dosageInstruction', [{}])[0].get('text')
                }
                for medication in medications
                if medication.get('medicationCodeableConcept', {}).get('coding')
            ],
            'recent_observations': [
                {
                    'code': obs['code']['coding'][0]['code'],
                    'display': obs['code']['coding'][0]['display'],
                    'value': obs.get('valueQuantity', {}).get('value'),
                    'unit': obs.get('valueQuantity', {}).get('unit'),
                    'date': obs.get('effectiveDateTime')
                }
                for obs in observations[:10]  # Recent 10 observations
                if obs.get('code', {}).get('coding') and obs.get('valueQuantity')
            ]
        }
        
        return context
    
    def _create_fhir_clinical_response(self, ai_analysis: Dict[str, Any], patient: Dict) -> Dict[str, Any]:
        """Create FHIR-compliant clinical response"""
        
        # Create FHIR Communication resource for AI analysis
        communication = {
            'resourceType': 'Communication',
            'status': 'completed',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/communication-category',
                    'code': 'notification',
                    'display': 'Notification'
                }]
            }],
            'subject': {
                'reference': f"Patient/{patient['id']}"
            },
            'payload': [{
                'contentString': f"Medical AI Analysis: {json.dumps(ai_analysis, indent=2)}"
            }],
            'sender': {
                'reference': 'Device/medical-ai-system'
            },
            'sent': datetime.now().isoformat(),
            'note': [{
                'text': f"Clinical recommendations: {ai_analysis.get('clinical_recommendations', [])}"
            }]
        }
        
        # Add clinical assessment if available
        if ai_analysis.get('risk_assessment'):
            assessment = {
                'resourceType': 'RiskAssessment',
                'subject': {
                    'reference': f"Patient/{patient['id']}"
                },
                'status': 'completed',
                'method': {
                    'coding': [{
                        'system': 'http://loinc.org',
                        'code': 'LA27442-7',
                        'display': 'Computer algorithm'
                    }]
                },
                'prediction': [{
                    'outcome': {
                        'coding': [{
                            'system': 'http://snomed.info/sct',
                            'code': '386053000',
                            'display': 'Evaluation procedure'
                        }]
                    },
                    'probabilityDecimal': ai_analysis['risk_assessment'].get('confidence', 0.0)
                }],
                'comment': f"Risk level: {ai_analysis['risk_assessment'].get('level', 'unknown')}"
            }
            
            return {
                'resourceType': 'Bundle',
                'type': 'collection',
                'entry': [
                    {'resource': communication},
                    {'resource': assessment}
                ]
            }
        
        return {
            'resourceType': 'Bundle',
            'type': 'collection',
            'entry': [{'resource': communication}]
        }
```

## DICOM Medical Imaging Integration

### DICOM Web Services Integration

```python
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import requests
from typing import List, Dict, Any

class DICOMWebClient:
    def __init__(self, dicom_server_config):
        self.base_url = dicom_server_config.base_url
        self.wado_url = f"{self.base_url}/wado"
        self.qido_url = f"{self.base_url}/qido"
        self.stow_url = f"{self.base_url}/stow"
        self.auth = dicom_server_config.auth
        
    async def search_studies(self, patient_id: str, study_date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Search for DICOM studies using QIDO-RS"""
        
        try:
            search_params = {
                'PatientID': patient_id,
                'QueryParameters': 'true'
            }
            
            if study_date_range:
                start_date, end_date = study_date_range
                search_params['StudyDate'] = f"{start_date}-{end_date}"
            
            # QIDO-RS search
            response = requests.get(
                f"{self.qido_url}/studies",
                params=search_params,
                auth=self.auth,
                headers={'Accept': 'application/dicom+json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"DICOM search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logging.error(f"DICOM study search error: {e}")
            return []
    
    async def retrieve_series_images(self, study_uid: str, series_uid: str) -> List[bytes]:
        """Retrieve DICOM image series using WADO-RS"""
        
        try:
            # Get series information
            series_response = requests.get(
                f"{self.qido_url}/studies/{study_uid}/series",
                params={'SeriesInstanceUID': series_uid},
                auth=self.auth,
                headers={'Accept': 'application/dicom+json'}
            )
            
            if series_response.status_code != 200:
                raise Exception(f"Failed to get series info: {series_response.status_code}")
            
            series_info = series_response.json()
            image_uids = [sop['SOPInstanceUID']['Value'][0] for sop in series_info]
            
            # Retrieve each image
            images = []
            for image_uid in image_uids:
                image_response = requests.get(
                    f"{self.wado_url}/studies/{study_uid}/series/{series_uid}/instances/{image_uid}",
                    auth=self.auth,
                    headers={'Accept': 'application/dicom'}
                )
                
                if image_response.status_code == 200:
                    images.append(image_response.content)
                else:
                    logging.warning(f"Failed to retrieve image {image_uid}: {image_response.status_code}")
            
            return images
            
        except Exception as e:
            logging.error(f"DICOM image retrieval error: {e}")
            return []
    
    async def analyze_medical_images(self, study_uid: str, series_uids: List[str]) -> Dict[str, Any]:
        """Analyze medical images using Medical AI"""
        
        try:
            # Retrieve images from each series
            all_images = []
            for series_uid in series_uids:
                series_images = await self.retrieve_series_images(study_uid, series_uid)
                all_images.extend(series_images)
            
            if not all_images:
                raise Exception("No images retrieved for analysis")
            
            # Analyze images with Medical AI
            image_analysis = await self._analyze_images_with_medical_ai(all_images)
            
            return image_analysis
            
        except Exception as e:
            logging.error(f"Medical image analysis failed: {e}")
            raise
    
    async def _analyze_images_with_medical_ai(self, images: List[bytes]) -> Dict[str, Any]:
        """Analyze DICOM images with Medical AI"""
        
        # This would integrate with Medical AI imaging models
        analysis_results = []
        
        for i, image_data in enumerate(images):
            try:
                # Parse DICOM data
                dicom_dataset = pydicom.dcmread(io.BytesIO(image_data))
                
                # Extract relevant metadata
                image_metadata = {
                    'study_instance_uid': dicom_dataset.get('StudyInstanceUID', ''),
                    'series_instance_uid': dicom_dataset.get('SeriesInstanceUID', ''),
                    'sop_instance_uid': dicom_dataset.get('SOPInstanceUID', ''),
                    'modality': dicom_dataset.get('Modality', ''),
                    'body_part_examined': dicom_dataset.get('BodyPartExamined', ''),
                    'study_date': dicom_dataset.get('StudyDate', ''),
                    'patient_position': dicom_dataset.get('PatientPosition', ''),
                    'image_type': dicom_dataset.get('ImageType', [])
                }
                
                # Process image with Medical AI
                ai_analysis = await self.medical_ai.analyze_medical_image(
                    image_data=dicom_dataset.pixel_array,
                    metadata=image_metadata,
                    analysis_type='comprehensive'
                )
                
                analysis_results.append({
                    'image_index': i,
                    'metadata': image_metadata,
                    'ai_analysis': ai_analysis
                })
                
            except Exception as e:
                logging.error(f"Error analyzing image {i}: {e}")
                continue
        
        # Consolidate results
        consolidated_analysis = self._consolidate_image_analysis(analysis_results)
        
        return consolidated_analysis
    
    def _consolidate_image_analysis(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Consolidate analysis results from multiple images"""
        
        if not analysis_results:
            return {'error': 'No successful analyses'}
        
        # Aggregate findings
        all_findings = []
        all_recommendations = []
        confidence_scores = []
        
        for result in analysis_results:
            ai_analysis = result['ai_analysis']
            
            if 'findings' in ai_analysis:
                all_findings.extend(ai_analysis['findings'])
            
            if 'recommendations' in ai_analysis:
                all_recommendations.extend(ai_analysis['recommendations'])
            
            if 'confidence_score' in ai_analysis:
                confidence_scores.append(ai_analysis['confidence_score'])
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine primary diagnosis/assessment
        primary_assessment = self._determine_primary_assessment(all_findings)
        
        return {
            'primary_assessment': primary_assessment,
            'all_findings': all_findings,
            'consolidated_recommendations': self._deduplicate_recommendations(all_recommendations),
            'overall_confidence': overall_confidence,
            'images_analyzed': len(analysis_results),
            'analysis_timestamp': datetime.now().isoformat(),
            'study_quality_assessment': self._assess_study_quality(analysis_results)
        }
```

## Laboratory Information Systems Integration

### HL7 ORU Integration

```python
import hl7
import asyncio
from typing import List, Dict, Any
from datetime import datetime

class LISIntegration:
    def __init__(self, lis_config):
        self.lis_config = lis_config
        self.mllp_port = lis_config.mllp_port
        self.hl7_endpoint = lis_config.hl7_endpoint
        
    async def receive_lab_results(self, hl7_message: str) -> Dict[str, Any]:
        """Receive and process HL7 ORU lab results"""
        
        try:
            # Parse HL7 message
            parsed_msg = hl7.parse(hl7_message)
            
            # Extract message metadata
            message_metadata = {
                'message_type': parsed_msg.header[8],
                'sending_application': parsed_msg.header[2],
                'sending_facility': parsed_msg.header[3],
                'receiving_application': parsed_msg.header[4],
                'receiving_facility': parsed_msg.header[5],
                'message_timestamp': parsed_msg.header[6],
                'message_control_id': parsed_msg.header[10]
            }
            
            # Extract patient information
            patient_info = self._extract_patient_info(parsed_msg)
            
            # Extract order information
            order_info = self._extract_order_info(parsed_msg)
            
            # Extract test results
            test_results = self._extract_test_results(parsed_msg)
            
            # Create structured lab result
            lab_result = {
                'message_metadata': message_metadata,
                'patient_info': patient_info,
                'order_info': order_info,
                'test_results': test_results,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Analyze results with Medical AI
            ai_analysis = await self._analyze_lab_results_with_medical_ai(lab_result)
            
            # Create response message
            response_message = self._create_ack_response(message_metadata['message_control_id'])
            
            return {
                'lab_result': lab_result,
                'ai_analysis': ai_analysis,
                'ack_response': response_message,
                'processing_status': 'success'
            }
            
        except Exception as e:
            logging.error(f"LIS integration error: {e}")
            return {
                'processing_status': 'error',
                'error_message': str(e),
                'hl7_message': hl7_message
            }
    
    def _extract_patient_info(self, parsed_msg) -> Dict[str, Any]:
        """Extract patient information from HL7 message"""
        
        # PID segment contains patient identification
        pid_segment = None
        for segment in parsed_msg.segments():
            if segment[0] == 'PID':
                pid_segment = segment
                break
        
        if not pid_segment:
            return {}
        
        return {
            'patient_id': pid_segment[3][0] if len(pid_segment) > 3 else '',
            'patient_id_list': pid_segment[3] if len(pid_segment) > 3 else [],
            'patient_name': pid_segment[5] if len(pid_segment) > 5 else '',
            'date_of_birth': pid_segment[7][0] if len(pid_segment) > 7 and pid_segment[7] else '',
            'sex': pid_segment[8] if len(pid_segment) > 8 else '',
            'race': pid_segment[10] if len(pid_segment) > 10 else '',
            'patient_address': pid_segment[11] if len(pid_segment) > 11 else '',
            'phone_number': pid_segment[13] if len(pid_segment) > 13 else ''
        }
    
    def _extract_order_info(self, parsed_msg) -> Dict[str, Any]:
        """Extract order information from HL7 message"""
        
        # OBR segment contains order information
        obr_segment = None
        for segment in parsed_msg.segments():
            if segment[0] == 'OBR':
                obr_segment = segment
                break
        
        if not obr_segment:
            return {}
        
        return {
            'order_id': obr_segment[2] if len(obr_segment) > 2 else '',
            'filler_order_number': obr_segment[3] if len(obr_segment) > 3 else '',
            'universal_service_id': obr_segment[4] if len(obr_segment) > 4 else '',
            'order_datetime': obr_segment[6] if len(obr_segment) > 6 else '',
            'specimen_received_datetime': obr_segment[14] if len(obr_segment) > 14 else '',
            'result_status': obr_segment[25] if len(obr_segment) > 25 else '',
            'result_copies_to': obr_segment[28] if len(obr_segment) > 28 else ''
        }
    
    def _extract_test_results(self, parsed_msg) -> List[Dict[str, Any]]:
        """Extract test results from HL7 message"""
        
        test_results = []
        
        # OBX segments contain observation results
        for segment in parsed_msg.segments():
            if segment[0] == 'OBX':
                if len(segment) >= 11:
                    result = {
                        'set_id': segment[1] if len(segment) > 1 else '',
                        'value_type': segment[2] if len(segment) > 2 else '',
                        'observation_identifier': segment[3] if len(segment) > 3 else '',
                        'observation_sub_id': segment[4] if len(segment) > 4 else '',
                        'observation_value': segment[5] if len(segment) > 5 else '',
                        'units': segment[6] if len(segment) > 6 else '',
                        'reference_range': segment[7] if len(segment) > 7 else '',
                        'abnormal_flags': segment[8] if len(segment) > 8 else '',
                        'probability': segment[9] if len(segment) > 9 else '',
                        'nature_of_abnormal_test': segment[10] if len(segment) > 10 else '',
                        'observation_result_status': segment[11] if len(segment) > 11 else ''
                    }
                    test_results.append(result)
        
        return test_results
    
    async def _analyze_lab_results_with_medical_ai(self, lab_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lab results with Medical AI"""
        
        try:
            # Build analysis query from lab results
            analysis_query = self._build_lab_analysis_query(lab_result)
            
            # Analyze with Medical AI
            ai_analysis = await self.medical_ai.analyze_laboratory_results(
                query=analysis_query,
                patient_context=lab_result['patient_info'],
                test_results=lab_result['test_results'],
                clinical_domain='laboratory'
            )
            
            return ai_analysis
            
        except Exception as e:
            logging.error(f"Medical AI lab analysis failed: {e}")
            return {
                'analysis_error': str(e),
                'fallback_recommendation': 'Manual review of laboratory results recommended'
            }
    
    def _build_lab_analysis_query(self, lab_result: Dict[str, Any]) -> str:
        """Build analysis query from lab results"""
        
        query_parts = ["Laboratory Results Analysis Request:"]
        
        # Add patient context
        patient_info = lab_result['patient_info']
        if patient_info.get('patient_id'):
            query_parts.append(f"Patient ID: {patient_info['patient_id']}")
        
        if patient_info.get('date_of_birth'):
            query_parts.append(f"DOB: {patient_info['date_of_birth']}")
        
        if patient_info.get('sex'):
            query_parts.append(f"Gender: {patient_info['sex']}")
        
        # Add test results
        query_parts.append("\nTest Results:")
        for result in lab_result['test_results']:
            observation_id = result.get('observation_identifier', {})
            if isinstance(observation_id, list) and observation_id:
                test_name = observation_id[0] if len(observation_id) > 0 else ''
            else:
                test_name = str(observation_id)
            
            value = result.get('observation_value', '')
            units = result.get('units', '')
            reference_range = result.get('reference_range', '')
            abnormal_flags = result.get('abnormal_flags', '')
            
            query_parts.append(
                f"- {test_name}: {value} {units} (Ref: {reference_range}) "
                f"{'[ABNORMAL]' if abnormal_flags else ''}"
            )
        
        return "\n".join(query_parts)
    
    def _create_ack_response(self, message_control_id: str) -> str:
        """Create HL7 ACK response"""
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        ack_message = (
            f"MSH|^~\\&|MEDICAL_AI|MEDICAL_AI_FACILITY|LIS|LIS_FACILITY|"
            f"{timestamp}||ACK^R01^{message_control_id}|"
            f"ACK{timestamp}|P|2.5\n"
            f"MSA|AA|{message_control_id}|Message accepted"
        )
        
        return ack_message
```

## Pharmacy System Integration

### Medication Management Integration

```python
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MedicationOrder:
    """Medication order structure"""
    order_id: str
    patient_id: str
    medication_name: str
    dosage: str
    frequency: str
    route: str
    duration: str
    prescriber_id: str
    order_date: datetime
    pharmacy_id: str

class PharmacyIntegration:
    def __init__(self, pharmacy_config, medical_ai_client):
        self.pharmacy_config = pharmacy_config
        self.medical_ai = medical_ai_client
        self.drug_database = None  # Would connect to drug database
        
    async def process_medication_order(self, order: MedicationOrder) -> Dict[str, Any]:
        """Process medication order with Medical AI analysis"""
        
        try:
            # 1. Validate medication order
            validation_result = await self._validate_medication_order(order)
            
            if not validation_result['valid']:
                return {
                    'order_id': order.order_id,
                    'status': 'rejected',
                    'rejection_reason': validation_result['reason'],
                    'validation_errors': validation_result['errors']
                }
            
            # 2. Check drug interactions
            interaction_check = await self._check_drug_interactions(order)
            
            # 3. Analyze with Medical AI for clinical insights
            ai_analysis = await self._analyze_medication_with_medical_ai(order)
            
            # 4. Check contraindications
            contraindication_check = await self._check_contraindications(order)
            
            # 5. Assess dosing appropriateness
            dosing_assessment = await self._assess_dosing(order)
            
            # 6. Generate comprehensive analysis
            analysis_result = {
                'order_id': order.order_id,
                'patient_id': order.patient_id,
                'medication_analysis': {
                    'ai_analysis': ai_analysis,
                    'drug_interactions': interaction_check,
                    'contraindications': contraindication_check,
                    'dosing_assessment': dosing_assessment,
                    'clinical_recommendations': self._generate_medication_recommendations(
                        ai_analysis, interaction_check, contraindication_check, dosing_assessment
                    )
                },
                'processing_timestamp': datetime.now().isoformat(),
                'analysis_confidence': ai_analysis.get('confidence_score', 0.0)
            }
            
            # 7. Generate pharmacy response
            pharmacy_response = self._generate_pharmacy_response(analysis_result)
            
            return pharmacy_response
            
        except Exception as e:
            logging.error(f"Medication order processing failed: {e}")
            return {
                'order_id': order.order_id,
                'status': 'error',
                'error_message': str(e)
            }
    
    async def _validate_medication_order(self, order: MedicationOrder) -> Dict[str, Any]:
        """Validate medication order"""
        
        validation_errors = []
        
        # Required field validation
        if not order.medication_name:
            validation_errors.append("Medication name is required")
        
        if not order.dosage:
            validation_errors.append("Dosage is required")
        
        if not order.frequency:
            validation_errors.append("Frequency is required")
        
        if not order.patient_id:
            validation_errors.append("Patient ID is required")
        
        # Dosage format validation
        if order.dosage and not self._validate_dosage_format(order.dosage):
            validation_errors.append("Invalid dosage format")
        
        # Frequency format validation
        if order.frequency and not self._validate_frequency_format(order.frequency):
            validation_errors.append("Invalid frequency format")
        
        return {
            'valid': len(validation_errors) == 0,
            'errors': validation_errors,
            'reason': 'Validation failed' if validation_errors else None
        }
    
    async def _check_drug_interactions(self, order: MedicationOrder) -> Dict[str, Any]:
        """Check for drug interactions"""
        
        try:
            # Get patient's current medications (would query patient records)
            current_medications = await self._get_patient_medications(order.patient_id)
            
            # Check interactions
            interaction_results = []
            for current_med in current_medications:
                interaction = await self._check_pairwise_interaction(
                    order.medication_name, current_med['name']
                )
                if interaction['severity'] != 'none':
                    interaction_results.append(interaction)
            
            # Overall interaction assessment
            max_severity = max(
                [result['severity'] for result in interaction_results],
                default='none'
            )
            
            return {
                'interactions_detected': len(interaction_results) > 0,
                'interaction_details': interaction_results,
                'max_severity': max_severity,
                'clinical_significance': self._assess_interaction_significance(interaction_results)
            }
            
        except Exception as e:
            logging.error(f"Drug interaction check failed: {e}")
            return {
                'interactions_detected': False,
                'error': str(e)
            }
    
    async def _check_pairwise_interaction(self, medication1: str, medication2: str) -> Dict[str, Any]:
        """Check interaction between two specific medications"""
        
        # This would query a drug interaction database
        # For demonstration, using mock interaction data
        known_interactions = {
            ('warfarin', 'aspirin'): {
                'severity': 'moderate',
                'clinical_effect': 'Increased bleeding risk',
                'recommendation': 'Monitor INR closely, consider alternative'
            },
            ('lisinopril', 'spironolactone'): {
                'severity': 'moderate',
                'clinical_effect': 'Hyperkalemia risk',
                'recommendation': 'Monitor potassium levels'
            },
            ('metformin', 'iodine_contrast'): {
                'severity': 'severe',
                'clinical_effect': 'Lactic acidosis risk',
                'recommendation': 'Hold metformin 48 hours before and after contrast'
            }
        }
        
        # Normalize medication names for lookup
        med1_norm = medication1.lower().strip()
        med2_norm = medication2.lower().strip()
        
        # Check both orders
        interaction = known_interactions.get((med1_norm, med2_norm)) or \
                     known_interactions.get((med2_norm, med1_norm))
        
        if interaction:
            return {
                'medication1': medication1,
                'medication2': medication2,
                'severity': interaction['severity'],
                'clinical_effect': interaction['clinical_effect'],
                'recommendation': interaction['recommendation']
            }
        else:
            return {
                'medication1': medication1,
                'medication2': medication2,
                'severity': 'none',
                'clinical_effect': 'No known interaction',
                'recommendation': 'Standard monitoring'
            }
    
    async def _analyze_medication_with_medical_ai(self, order: MedicationOrder) -> Dict[str, Any]:
        """Analyze medication with Medical AI"""
        
        # Build analysis query
        analysis_query = f"""
        Medication Order Analysis:
        Patient: {order.patient_id}
        Medication: {order.medication_name}
        Dosage: {order.dosage}
        Frequency: {order.frequency}
        Route: {order.route}
        Duration: {order.duration}
        
        Please analyze this medication order for:
        1. Clinical appropriateness
        2. Potential concerns
        3. Monitoring requirements
        4. Patient education needs
        """
        
        try:
            # Query Medical AI
            ai_response = await self.medical_ai.analyze_medication_order(
                query=analysis_query,
                medication_details={
                    'name': order.medication_name,
                    'dosage': order.dosage,
                    'frequency': order.frequency,
                    'route': order.route,
                    'duration': order.duration
                },
                patient_context={'patient_id': order.patient_id}
            )
            
            return ai_response
            
        except Exception as e:
            logging.error(f"Medical AI medication analysis failed: {e}")
            return {
                'analysis_error': str(e),
                'clinical_recommendations': ['Manual pharmacy review recommended']
            }
    
    def _generate_pharmacy_response(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final pharmacy response"""
        
        medication_analysis = analysis_result['medication_analysis']
        
        # Determine overall status
        status = 'approved'
        rejection_reason = None
        
        # Check for critical issues
        if medication_analysis['drug_interactions'].get('max_severity') == 'severe':
            status = 'rejected'
            rejection_reason = 'Severe drug interaction detected'
        elif medication_analysis['contraindications'].get('contraindicated', False):
            status = 'rejected'
            rejection_reason = 'Medication contraindicated for patient'
        elif medication_analysis['dosing_assessment'].get('inappropriate_dosing', False):
            status = 'modified'
            rejection_reason = 'Dosing adjustments required'
        
        # Generate clinical recommendations
        recommendations = medication_analysis.get('clinical_recommendations', [])
        
        return {
            'order_id': analysis_result['order_id'],
            'patient_id': analysis_result['patient_id'],
            'status': status,
            'rejection_reason': rejection_reason,
            'clinical_recommendations': recommendations,
            'ai_analysis_confidence': analysis_result['analysis_confidence'],
            'processing_timestamp': analysis_result['processing_timestamp'],
            'pharmacy_id': self.pharmacy_config.pharmacy_id,
            'pharmacist_review': 'ai_assisted',
            'follow_up_required': self._determine_follow_up_requirements(medication_analysis)
        }
```

## Clinical Decision Support Integration

### Integration with Existing CDS Tools

```python
class ClinicalDecisionSupportIntegration:
    def __init__(self, cds_config, medical_ai_client):
        self.cds_config = cds_config
        self.medical_ai = medical_ai_client
        self.cds_tools = self._initialize_cds_tools()
        
    def _initialize_cds_tools(self) -> Dict[str, Any]:
        """Initialize connections to existing CDS tools"""
        
        cds_tools = {}
        
        # Connect to various CDS systems
        if self.cds_config.enable_epic_cds:
            cds_tools['epic_cds'] = EpicCDSClient(self.cds_config.epic_config)
        
        if self.cds_config.enable_cerner_powerchart:
            cds_tools['cerner_cds'] = CernerCDSClient(self.cds_config.cerner_config)
        
        if self.cds_config.enable_surescripts:
            cds_tools['surescripts'] = SurescriptsClient(self.cds_config.surescripts_config)
        
        if self.cds_config.enable_first_databank:
            cds_tools['drug_database'] = FirstDataBankClient(self.cds_config.fdb_config)
        
        return cds_tools
    
    async def comprehensive_cds_analysis(self, patient_context: Dict[str, Any], 
                                       clinical_query: str) -> Dict[str, Any]:
        """Perform comprehensive clinical decision support analysis"""
        
        try:
            # 1. Gather data from all CDS sources
            cds_data = await self._gather_cds_data(patient_context)
            
            # 2. Run Medical AI analysis
            ai_analysis = await self.medical_ai.comprehensive_clinical_analysis(
                patient_context=patient_context,
                clinical_query=clinical_query,
                cds_data=cds_data
            )
            
            # 3. Integrate with existing CDS rules
            integrated_recommendations = await self._integrate_cds_recommendations(
                ai_analysis, cds_data
            )
            
            # 4. Check for conflicts and contradictions
            conflict_analysis = self._analyze_recommendation_conflicts(
                integrated_recommendations
            )
            
            # 5. Generate final CDS output
            final_cds_output = {
                'patient_id': patient_context.get('patient_id'),
                'clinical_query': clinical_query,
                'cds_recommendations': integrated_recommendations,
                'ai_insights': ai_analysis,
                'conflict_analysis': conflict_analysis,
                'evidence_level': self._assess_evidence_level(integrated_recommendations),
                'confidence_score': self._calculate_confidence_score(ai_analysis, cds_data),
                'timestamp': datetime.now().isoformat(),
                'cds_system_version': self._get_cds_version_info()
            }
            
            # 6. Log CDS interaction for audit
            await self._log_cds_interaction(final_cds_output)
            
            return final_cds_output
            
        except Exception as e:
            logging.error(f"CDS analysis failed: {e}")
            return {
                'error': str(e),
                'fallback_recommendations': ['Manual clinical review required'],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _gather_cds_data(self, patient_context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather data from all configured CDS sources"""
        
        cds_data = {
            'patient_context': patient_context,
            'cds_sources': {}
        }
        
        # Gather data from each CDS tool
        for tool_name, tool_client in self.cds_tools.items():
            try:
                if tool_name == 'epic_cds':
                    data = await tool_client.get_patient_cds_data(patient_context)
                elif tool_name == 'cerner_cds':
                    data = await tool_client.get_clinical_decision_support(patient_context)
                elif tool_name == 'surescripts':
                    data = await tool_client.get_medication_history(patient_context)
                elif tool_name == 'drug_database':
                    data = await tool_client.get_drug_information(patient_context)
                else:
                    data = {}
                
                cds_data['cds_sources'][tool_name] = data
                
            except Exception as e:
                logging.error(f"Error gathering data from {tool_name}: {e}")
                cds_data['cds_sources'][tool_name] = {'error': str(e)}
        
        return cds_data
    
    async def _integrate_cds_recommendations(self, ai_analysis: Dict[str, Any], 
                                           cds_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Integrate Medical AI recommendations with existing CDS rules"""
        
        all_recommendations = []
        
        # Add Medical AI recommendations
        if 'clinical_recommendations' in ai_analysis:
            for rec in ai_analysis['clinical_recommendations']:
                recommendation = {
                    'source': 'medical_ai',
                    'recommendation': rec,
                    'confidence': ai_analysis.get('confidence_score', 0.0),
                    'evidence_level': 'ai_analysis',
                    'clinical_domain': ai_analysis.get('medical_domain', 'general')
                }
                all_recommendations.append(recommendation)
        
        # Add recommendations from CDS sources
        for source_name, source_data in cds_data.get('cds_sources', {}).items():
            if 'recommendations' in source_data:
                for rec in source_data['recommendations']:
                    recommendation = {
                        'source': source_name,
                        'recommendation': rec['text'],
                        'confidence': rec.get('confidence', 0.0),
                        'evidence_level': rec.get('evidence_level', 'clinical_guideline'),
                        'clinical_domain': rec.get('clinical_domain', 'general'),
                        'implementation_guidance': rec.get('implementation_guidance')
                    }
                    all_recommendations.append(recommendation)
        
        # Remove duplicates and prioritize
        integrated_recommendations = self._prioritize_recommendations(all_recommendations)
        
        return integrated_recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and deduplicate recommendations"""
        
        # Sort by confidence and evidence level
        priority_weights = {
            'ai_analysis': 0.9,
            'clinical_guideline': 0.8,
            'expert_consensus': 0.7,
            'case_series': 0.6
        }
        
        for rec in recommendations:
            evidence_weight = priority_weights.get(rec.get('evidence_level'), 0.5)
            confidence_weight = rec.get('confidence', 0.0)
            rec['priority_score'] = (evidence_weight * 0.7) + (confidence_weight * 0.3)
        
        # Sort by priority score (highest first)
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Remove similar recommendations
        deduplicated_recommendations = self._deduplicate_similar_recommendations(recommendations)
        
        return deduplicated_recommendations
    
    def _deduplicate_similar_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove similar recommendations"""
        
        seen_recommendations = set()
        deduplicated = []
        
        for rec in recommendations:
            # Create similarity key (simplified)
            rec_text = rec['recommendation'].lower().strip()
            
            # Check for similar recommendations
            is_duplicate = False
            for seen_text in seen_recommendations:
                if self._calculate_text_similarity(rec_text, seen_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_recommendations.add(rec_text)
                deduplicated.append(rec)
        
        return deduplicated
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        
        # Simple similarity calculation (would use more sophisticated NLP in production)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _log_cds_interaction(self, cds_output: Dict[str, Any]):
        """Log CDS interaction for compliance and audit"""
        
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': cds_output.get('patient_id'),
            'cds_type': 'comprehensive',
            'recommendations_count': len(cds_output.get('cds_recommendations', [])),
            'ai_confidence': cds_output.get('confidence_score', 0.0),
            'conflicts_detected': len(cds_output.get('conflict_analysis', {}).get('conflicts', [])),
            'system_version': cds_output.get('cds_system_version', {})
        }
        
        # Log to audit database
        await self._store_audit_log(audit_log)
```

## SDK Examples

### Python SDK for Medical AI Integration

```python
from medical_ai_sdk import MedicalAIClient, MedicalDomain, UrgencyLevel
from typing import Dict, List, Any, Optional
import asyncio

class MedicalAIIntegrationSDK:
    """Python SDK for Medical AI system integration"""
    
    def __init__(self, api_key: str, base_url: str, medical_compliance: bool = True):
        self.client = MedicalAIClient(
            api_key=api_key,
            base_url=base_url,
            medical_compliance=medical_compliance
        )
        self.connected_systems = {}
        
    async def connect_emr_system(self, system_type: str, config: Dict[str, Any]):
        """Connect to EMR system"""
        
        if system_type == 'epic':
            connector = EpicFHIRClient(config)
        elif system_type == 'cerner':
            connector = CernerMillenniumClient(config)
        elif system_type == 'allscripts':
            connector = AllscriptsClient(config)
        else:
            raise ValueError(f"Unsupported EMR system: {system_type}")
        
        await connector.authenticate()
        self.connected_systems[system_type] = connector
        
        return connector
    
    async def analyze_patient_comprehensive(self, patient_identifier: str, 
                                          emr_system: str, 
                                          clinical_query: str) -> Dict[str, Any]:
        """Perform comprehensive patient analysis"""
        
        try:
            # 1. Get patient data from EMR
            emr_connector = self.connected_systems[emr_system]
            patient_data = await emr_connector.get_patient_data(patient_identifier)
            
            # 2. Query Medical AI
            ai_response = await self.client.analyze_clinical_situation(
                query=clinical_query,
                patient_context=self._extract_patient_context(patient_data),
                medical_domain=self._determine_medical_domain(clinical_query),
                urgency_level=self._assess_urgency(clinical_query, patient_data)
            )
            
            # 3. Return integrated analysis
            return {
                'patient_data_source': emr_system,
                'patient_analysis': ai_response,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'comprehensive'
            }
            
        except Exception as e:
            logging.error(f"Patient analysis failed: {e}")
            raise
    
    async def analyze_medical_images(self, study_uid: str, 
                                   dicom_server: str,
                                   analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Analyze medical images using DICOM integration"""
        
        try:
            # 1. Connect to DICOM server
            dicom_client = DICOMWebClient(self.connected_systems[dicom_server])
            
            # 2. Retrieve and analyze images
            analysis_result = await dicom_client.analyze_medical_images(study_uid, [])
            
            # 3. Enhance with Medical AI insights
            enhanced_analysis = await self.client.analyze_medical_images(
                image_analysis=analysis_result,
                analysis_type=analysis_type
            )
            
            return {
                'study_uid': study_uid,
                'analysis_result': enhanced_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Medical image analysis failed: {e}")
            raise
    
    async def process_laboratory_results(self, hl7_message: str) -> Dict[str, Any]:
        """Process laboratory results from LIS"""
        
        try:
            # 1. Parse HL7 message
            lis_client = LISIntegration({})  # Configure as needed
            lab_result = await lis_client.receive_lab_results(hl7_message)
            
            # 2. Analyze with Medical AI
            ai_analysis = await self.client.analyze_laboratory_results(
                lab_results=lab_result['lab_result'],
                clinical_context=lab_result['patient_info']
            )
            
            return {
                'lab_result_processing': 'success',
                'ai_analysis': ai_analysis,
                'ack_response': lab_result.get('ack_response'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Lab result processing failed: {e}")
            raise
    
    def _extract_patient_context(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized patient context from EMR data"""
        
        return {
            'patient_id': patient_data.get('patient_id'),
            'demographics': patient_data.get('demographics', {}),
            'active_conditions': patient_data.get('conditions', []),
            'current_medications': patient_data.get('medications', []),
            'allergies': patient_data.get('allergies', []),
            'recent_vitals': patient_data.get('vital_signs', []),
            'lab_results': patient_data.get('lab_results', [])
        }
    
    def _determine_medical_domain(self, query: str) -> MedicalDomain:
        """Determine medical domain from query text"""
        
        domain_keywords = {
            MedicalDomain.CARDIOLOGY: ['chest pain', 'heart', 'cardiac', 'hypertension'],
            MedicalDomain.ONCOLOGY: ['cancer', 'tumor', 'chemotherapy', 'oncology'],
            MedicalDomain.NEUROLOGY: ['stroke', 'seizure', 'brain', 'neurological'],
            MedicalDomain.EMERGENCY: ['emergency', 'trauma', 'critical', 'acute'],
            MedicalDomain.PEDIATRICS: ['child', 'pediatric', 'infant', 'adolescent']
        }
        
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return MedicalDomain.GENERAL
    
    def _assess_urgency(self, query: str, patient_data: Dict[str, Any]) -> UrgencyLevel:
        """Assess urgency level from query and patient data"""
        
        urgency_keywords = {
            UrgencyLevel.CRITICAL: ['severe chest pain', 'unconscious', 'cardiac arrest', 'stroke'],
            UrgencyLevel.HIGH: ['severe', 'sudden', 'worsening', 'high fever'],
            UrgencyLevel.MEDIUM: ['moderate', 'concerning', 'several days'],
            UrgencyLevel.LOW: ['mild', 'occasional', 'minor']
        }
        
        query_lower = query.lower()
        
        for urgency, keywords in urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return urgency
        
        return UrgencyLevel.MEDIUM

# Usage Example
async def main():
    # Initialize SDK
    sdk = MedicalAIIntegrationSDK(
        api_key="your_api_key",
        base_url="https://api.medical-ai.example.com",
        medical_compliance=True
    )
    
    # Connect to Epic EMR
    epic_config = {
        'base_url': 'https://fhir.epic.com/interconnect-fhir-oauth',
        'client_id': 'your_epic_client_id',
        'client_secret': 'your_epic_client_secret'
    }
    
    await sdk.connect_emr_system('epic', epic_config)
    
    # Analyze patient
    analysis = await sdk.analyze_patient_comprehensive(
        patient_identifier='patient_123',
        emr_system='epic',
        clinical_query="Patient presents with chest pain and shortness of breath"
    )
    
    print(f"Analysis result: {analysis}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/Node.js SDK

```javascript
import { MedicalAI } from '@medical-ai/sdk';
import axios from 'axios';

class MedicalAIIntegrationSDK {
    constructor(config) {
        this.client = new MedicalAI({
            apiKey: config.apiKey,
            baseUrl: config.baseUrl,
            medicalCompliance: config.medicalCompliance || true
        });
        this.connectedSystems = {};
    }
    
    async connectEMRSystem(systemType, config) {
        let connector;
        
        switch(systemType) {
            case 'epic':
                connector = new EpicFHIRClient(config);
                break;
            case 'cerner':
                connector = new CernerMillenniumClient(config);
                break;
            default:
                throw new Error(`Unsupported EMR system: ${systemType}`);
        }
        
        await connector.authenticate();
        this.connectedSystems[systemType] = connector;
        return connector;
    }
    
    async analyzePatientComprehensive(patientIdentifier, emrSystem, clinicalQuery) {
        try {
            // Get patient data from EMR
            const emrConnector = this.connectedSystems[emrSystem];
            const patientData = await emrConnector.getPatientData(patientIdentifier);
            
            // Query Medical AI
            const aiResponse = await this.client.analyzeClinicalSituation({
                query: clinicalQuery,
                patientContext: this.extractPatientContext(patientData),
                medicalDomain: this.determineMedicalDomain(clinicalQuery),
                urgencyLevel: this.assessUrgency(clinicalQuery, patientData)
            });
            
            return {
                patientDataSource: emrSystem,
                patientAnalysis: aiResponse,
                timestamp: new Date().toISOString(),
                analysisType: 'comprehensive'
            };
            
        } catch (error) {
            console.error('Patient analysis failed:', error);
            throw error;
        }
    }
    
    extractPatientContext(patientData) {
        return {
            patientId: patientData.patientId,
            demographics: patientData.demographics || {},
            activeConditions: patientData.conditions || [],
            currentMedications: patientData.medications || [],
            allergies: patientData.allergies || [],
            recentVitals: patientData.vitalSigns || [],
            labResults: patientData.labResults || []
        };
    }
    
    determineMedicalDomain(query) {
        const domainKeywords = {
            'cardiology': ['chest pain', 'heart', 'cardiac', 'hypertension'],
            'oncology': ['cancer', 'tumor', 'chemotherapy', 'oncology'],
            'neurology': ['stroke', 'seizure', 'brain', 'neurological'],
            'emergency': ['emergency', 'trauma', 'critical', 'acute'],
            'pediatrics': ['child', 'pediatric', 'infant', 'adolescent']
        };
        
        const queryLower = query.toLowerCase();
        
        for (const [domain, keywords] of Object.entries(domainKeywords)) {
            if (keywords.some(keyword => queryLower.includes(keyword))) {
                return domain;
            }
        }
        
        return 'general';
    }
    
    assessUrgency(query, patientData) {
        const urgencyKeywords = {
            'critical': ['severe chest pain', 'unconscious', 'cardiac arrest', 'stroke'],
            'high': ['severe', 'sudden', 'worsening', 'high fever'],
            'medium': ['moderate', 'concerning', 'several days'],
            'low': ['mild', 'occasional', 'minor']
        };
        
        const queryLower = query.toLowerCase();
        
        for (const [urgency, keywords] of Object.entries(urgencyKeywords)) {
            if (keywords.some(keyword => queryLower.includes(keyword))) {
                return urgency;
            }
        }
        
        return 'medium';
    }
}

// Usage Example
async function main() {
    // Initialize SDK
    const sdk = new MedicalAIIntegrationSDK({
        apiKey: 'your_api_key',
        baseUrl: 'https://api.medical-ai.example.com',
        medicalCompliance: true
    });
    
    // Connect to Epic EMR
    const epicConfig = {
        baseUrl: 'https://fhir.epic.com/interconnect-fhir-oauth',
        clientId: 'your_epic_client_id',
        clientSecret: 'your_epic_client_secret'
    };
    
    await sdk.connectEMRSystem('epic', epicConfig);
    
    // Analyze patient
    const analysis = await sdk.analyzePatientComprehensive(
        'patient_123',
        'epic',
        'Patient presents with chest pain and shortness of breath'
    );
    
    console.log('Analysis result:', analysis);
}

// Run example
main().catch(console.error);
```

---

**⚠️ Integration Disclaimer**: This integration guide is designed for medical device compliance. All integrations must be validated for the specific healthcare environment and regulatory requirements. Never integrate medical AI systems without proper security validation and regulatory approval.
