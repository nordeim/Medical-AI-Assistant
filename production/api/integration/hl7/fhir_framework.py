# HL7 FHIR R4 Integration Framework for Healthcare API
# Production-grade FHIR-compliant EHR integration with validation and mapping

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aiofiles
import jsonschema
from jsonschema import validate, ValidationError
import pydantic
from pydantic import BaseModel, Field, validator
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FHIRResourceType(Enum):
    """FHIR Resource Types"""
    PATIENT = "Patient"
    PRACTITIONER = "Practitioner"
    ORGANIZATION = "Organization"
    ENCOUNTER = "Encounter"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    PROCEDURE = "Procedure"
    MEDICATION = "Medication"
    MEDICATION_REQUEST = "MedicationRequest"
    MEDICATION_STATEMENT = "MedicationStatement"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    CARE_PLAN = "CarePlan"
    CARE_TEAM = "CareTeam"
    DEVICE = "Device"
    DEVICE_METRIC = "DeviceMetric"
    LOCATION = "Location"
    APPOINTMENT = "Appointment"
    APPOINTMENT_RESPONSE = "AppointmentResponse"
    SCHEDULE = "Schedule"
    SLOT = "Slot"
    SERVICE_TYPE = "ServiceType"
    SERVICE_CATEGORY = "ServiceCategory"
    SPECIALTY = "Specialty"
    INVOICE = "Invoice"
    CLAIM = "Claim"
    CLAIM_RESPONSE = "ClaimResponse"

class FHIRVersion(Enum):
    """FHIR versions supported"""
    R4 = "4.0.1"
    R4B = "4.3.0"
    R5 = "5.0.0"

class InteractionType(Enum):
    """FHIR interaction types"""
    READ = "read"
    SEARCH = "search"
    CREATE = "create"
    UPDATE = "update"
    PATCH = "patch"
    DELETE = "delete"
    BUNDLE = "bundle"
    HISTORY = "history"
    PATCH = "patch"

class OperationalStatus(Enum):
    """Operational status for healthcare services"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ENTERED_IN_ERROR = "entered-in-error"

# Pydantic models for FHIR resources
class FHIRIdentifier(BaseModel):
    """FHIR Identifier type"""
    use: Optional[str] = None
    type: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    value: str
    period: Optional[Dict[str, Any]] = None
    assigner: Optional[Dict[str, Any]] = None

class FHIRHumanName(BaseModel):
    """FHIR HumanName type"""
    use: Optional[str] = None
    text: Optional[str] = None
    family: Optional[str] = None
    given: Optional[List[str]] = None
    prefix: Optional[List[str]] = None
    suffix: Optional[List[str]] = None
    period: Optional[Dict[str, Any]] = None

class FHIRContactPoint(BaseModel):
    """FHIR ContactPoint type"""
    system: Optional[str] = None
    value: Optional[str] = None
    use: Optional[str] = None
    rank: Optional[int] = None
    period: Optional[Dict[str, Any]] = None

class FHIRAddress(BaseModel):
    """FHIR Address type"""
    use: Optional[str] = None
    type: Optional[str] = None
    text: Optional[str] = None
    line: Optional[List[str]] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None
    period: Optional[Dict[str, Any]] = None

class FHIRResource(BaseModel):
    """Base FHIR Resource model"""
    resourceType: str
    id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    implicitRules: Optional[str] = None
    language: Optional[str] = "en"
    
    class Config:
        extra = "allow"

class FHIRPatient(FHIRResource):
    """FHIR Patient Resource"""
    resourceType: str = "Patient"
    identifier: Optional[List[FHIRIdentifier]] = None
    active: Optional[bool] = True
    name: Optional[List[FHIRHumanName]] = None
    telecom: Optional[List[FHIRContactPoint]] = None
    gender: Optional[str] = Field(None, regex="^(male|female|other|unknown)$")
    birthDate: Optional[str] = Field(None, regex="^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    deceased: Optional[Union[bool, str]] = None
    address: Optional[List[FHIRAddress]] = None
    maritalStatus: Optional[Dict[str, Any]] = None
    multipleBirth: Optional[Union[bool, int]] = None
    photo: Optional[List[Dict[str, Any]]] = None
    contact: Optional[List[Dict[str, Any]]] = None
    communication: Optional[List[Dict[str, Any]]] = None
    generalPractitioner: Optional[List[Dict[str, Any]]] = None
    managingOrganization: Optional[Dict[str, Any]] = None

class FHIRObservation(FHIRResource):
    """FHIR Observation Resource"""
    resourceType: str = "Observation"
    identifier: Optional[List[FHIRIdentifier]] = None
    basedOn: Optional[List[Dict[str, Any]]] = None
    partOf: Optional[List[Dict[str, Any]]] = None
    status: str = Field(..., regex="^(registered|preliminary|final|amended|corrected|cancelled|entered-in-error|unknown)$")
    category: Optional[List[Dict[str, Any]]] = None
    code: Dict[str, Any]
    subject: Optional[Dict[str, Any]] = None
    focus: Optional[List[Dict[str, Any]]] = None
    encounter: Optional[Dict[str, Any]] = None
    effective: Optional[Union[str, Dict[str, Any]]] = None
    issued: Optional[str] = None
    performer: Optional[List[Dict[str, Any]]] = None
    value: Optional[Dict[str, Any]] = None
    dataAbsentReason: Optional[Dict[str, Any]] = None
    interpretation: Optional[List[Dict[str, Any]]] = None
    note: Optional[List[Dict[str, Any]]] = None
    bodySite: Optional[Dict[str, Any]] = None
    method: Optional[Dict[str, Any]] = None
    device: Optional[Dict[str, Any]] = None
    referenceRange: Optional[List[Dict[str, Any]]] = None
    hasMember: Optional[List[Dict[str, Any]]] = None
    derivedFrom: Optional[List[Dict[str, Any]]] = None
    component: Optional[List[Dict[str, Any]]] = None

class FHIRCondition(FHIRResource):
    """FHIR Condition Resource"""
    resourceType: str = "Condition"
    identifier: Optional[List[FHIRIdentifier]] = None
    clinicalStatus: Optional[Dict[str, Any]] = None
    verificationStatus: Optional[Dict[str, Any]] = None
    category: Optional[List[Dict[str, Any]]] = None
    severity: Optional[Dict[str, Any]] = None
    code: Dict[str, Any]
    bodySite: Optional[List[Dict[str, Any]]] = None
    subject: Dict[str, Any]
    encounter: Optional[Dict[str, Any]] = None
    onset: Optional[Union[str, Dict[str, Any], Dict[str, Any]]] = None
    abatement: Optional[Union[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = None
    recordedDate: Optional[str] = None
    recorder: Optional[Dict[str, Any]] = None
    asserter: Optional[Dict[str, Any]] = None
    stage: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    note: Optional[List[Dict[str, Any]]] = None

class FHIRBundle(FHIRResource):
    """FHIR Bundle Resource for collections"""
    resourceType: str = "Bundle"
    identifier: Optional[Dict[str, Any]] = None
    type: str = Field(..., regex="^(document|message|transaction|transaction-response|batch|batch-response|history|searchset|collection)$")
    timestamp: Optional[str] = None
    total: Optional[int] = None
    link: Optional[List[Dict[str, Any]]] = None
    entry: Optional[List[Dict[str, Any]]] = None
    signature: Optional[Dict[str, Any]] = None

class FHIRCapabilityStatement(FHIRResource):
    """FHIR Capability Statement"""
    resourceType: str = "CapabilityStatement"
    url: Optional[str] = None
    version: Optional[str] = "1.0.0"
    name: str
    title: Optional[str] = None
    status: str = Field(default="active", regex="^(draft|active|retired|unknown)$")
    experimental: Optional[bool] = False
    date: str
    publisher: Optional[str] = None
    contact: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None
    useContext: Optional[List[Dict[str, Any]]] = None
    jurisdiction: Optional[List[Dict[str, Any]]] = None
    purpose: Optional[str] = None
    copyright: Optional[str] = None
    kind: str = Field(..., regex="^(instance|capability|requirements)$")
    software: Optional[Dict[str, Any]] = None
    implementation: Optional[Dict[str, Any]] = None
    fhirVersion: str = Field(default="4.0.1")
    format: List[str]
    patchFormat: Optional[List[str]] = None
    implementationGuide: Optional[List[str]] = None
    rest: Optional[List[Dict[str, Any]]] = None
    messaging: Optional[List[Dict[str, Any]]] = None
    document: Optional[List[Dict[str, Any]]] = None

class FHIRValidator:
    """FHIR R4 Resource Validator"""
    
    def __init__(self, version: FHIRVersion = FHIRVersion.R4):
        self.version = version
        self.schemas = self._load_fhir_schemas()
    
    def _load_fhir_schemas(self) -> Dict[str, Dict]:
        """Load FHIR JSON schemas for validation"""
        # In production, these would be loaded from official FHIR schema files
        # For demo purposes, returning basic schemas
        return {
            "Patient": {
                "type": "object",
                "required": ["resourceType"],
                "properties": {
                    "resourceType": {"const": "Patient"},
                    "id": {"type": "string"},
                    "name": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "use": {"type": "string"},
                                "family": {"type": "string"},
                                "given": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "gender": {"type": "string", "enum": ["male", "female", "other", "unknown"]},
                    "birthDate": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
                }
            },
            "Observation": {
                "type": "object",
                "required": ["resourceType", "status", "code"],
                "properties": {
                    "resourceType": {"const": "Observation"},
                    "id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"]
                    },
                    "code": {"type": "object"},
                    "subject": {"type": "object"}
                }
            }
        }
    
    def validate_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a FHIR resource against schema"""
        resource_type = resource.get("resourceType")
        
        if not resource_type:
            return {
                "valid": False,
                "errors": ["Missing resourceType"]
            }
        
        schema = self.schemas.get(resource_type)
        if not schema:
            return {
                "valid": False,
                "errors": [f"No schema available for resource type: {resource_type}"]
            }
        
        try:
            validate(instance=resource, schema=schema)
            return {
                "valid": True,
                "resourceType": resource_type,
                "version": self.version.value
            }
        except ValidationError as e:
            return {
                "valid": False,
                "resourceType": resource_type,
                "errors": [str(e.message)]
            }
    
    def validate_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FHIR Bundle and all contained resources"""
        validation_results = {
            "valid": True,
            "bundle_errors": [],
            "resource_validations": []
        }
        
        # Validate bundle structure
        if bundle.get("resourceType") != "Bundle":
            validation_results["valid"] = False
            validation_results["bundle_errors"].append("Invalid resourceType for bundle")
        
        # Validate entries
        entries = bundle.get("entry", [])
        if not entries:
            validation_results["valid"] = False
            validation_results["bundle_errors"].append("Bundle must contain at least one entry")
        
        for i, entry in enumerate(entries):
            try:
                resource = entry.get("resource", {})
                resource_validation = self.validate_resource(resource)
                resource_validation["entry_index"] = i
                validation_results["resource_validations"].append(resource_validation)
                
                if not resource_validation["valid"]:
                    validation_results["valid"] = False
            
            except Exception as e:
                validation_results["valid"] = False
                validation_results["bundle_errors"].append(f"Error validating entry {i}: {str(e)}")
        
        return validation_results

class EHRMapper:
    """Maps EHR system data to FHIR format"""
    
    def __init__(self):
        self.mapping_templates = self._load_mapping_templates()
    
    def _load_mapping_templates(self) -> Dict[str, Dict]:
        """Load mapping templates for different EHR systems"""
        return {
            "epic_to_fhir": {
                "patient_mapping": {
                    "mrn": {"fhir_field": "identifier[0].value", "type": "MR"},
                    "first_name": {"fhir_field": "name[0].given[0]", "type": "string"},
                    "last_name": {"fhir_field": "name[0].family", "type": "string"},
                    "birth_date": {"fhir_field": "birthDate", "type": "date"},
                    "gender": {"fhir_field": "gender", "type": "code"}
                }
            },
            "cerner_to_fhir": {
                "patient_mapping": {
                    "patient_id": {"fhir_field": "identifier[0].value", "type": "MR"},
                    "name_first": {"fhir_field": "name[0].given[0]", "type": "string"},
                    "name_last": {"fhir_field": "name[0].family", "type": "string"},
                    "birthdate": {"fhir_field": "birthDate", "type": "date"},
                    "sex": {"fhir_field": "gender", "type": "code"}
                }
            }
        }
    
    def map_epic_patient_to_fhir(self, epic_patient: Dict[str, Any]) -> FHIRPatient:
        """Map Epic patient data to FHIR Patient resource"""
        
        name = []
        if epic_patient.get("first_name") and epic_patient.get("last_name"):
            name = [FHIRHumanName(
                family=epic_patient.get("last_name"),
                given=[epic_patient.get("first_name")]
            )]
        
        identifiers = []
        if epic_patient.get("mrn"):
            identifiers = [FHIRIdentifier(
                system="urn:oid:1.2.840.114350",
                value=epic_patient.get("mrn")
            )]
        
        return FHIRPatient(
            id=epic_patient.get("patient_id"),
            active=epic_patient.get("active", True),
            name=name,
            gender=self._map_epic_gender(epic_patient.get("sex")),
            birthDate=epic_patient.get("birth_date")
        )
    
    def _map_epic_gender(self, epic_gender: str) -> Optional[str]:
        """Map Epic gender codes to FHIR"""
        gender_mapping = {
            "M": "male",
            "F": "female",
            "U": "unknown",
            "O": "other"
        }
        return gender_mapping.get(epic_gender.upper())
    
    def map_cerner_patient_to_fhir(self, cerner_patient: Dict[str, Any]) -> FHIRPatient:
        """Map Cerner patient data to FHIR Patient resource"""
        
        name = []
        if cerner_patient.get("name_first") and cerner_patient.get("name_last"):
            name = [FHIRHumanName(
                family=cerner_patient.get("name_last"),
                given=[cerner_patient.get("name_first")]
            )]
        
        return FHIRPatient(
            id=cerner_patient.get("patient_id"),
            active=cerner_patient.get("active", True),
            name=name,
            gender=self._map_cerner_gender(cerner_patient.get("sex")),
            birthDate=cerner_patient.get("birthdate")
        )
    
    def _map_cerner_gender(self, cerner_gender: str) -> Optional[str]:
        """Map Cerner gender codes to FHIR"""
        gender_mapping = {
            "Male": "male",
            "Female": "female",
            "Unknown": "unknown",
            "Other": "other"
        }
        return gender_mapping.get(cerner_gender)

class FHIRServer:
    """FHIR R4 Server implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = FHIRValidator()
        self.mapper = EHRMapper()
        self.resource_store: Dict[str, Dict[str, FHIRResource]] = {}
        self.base_url = config.get("base_url", "https://api.healthcare.org/fhir")
        
        # Initialize resource stores
        for resource_type in FHIRResourceType:
            self.resource_store[resource_type.value] = {}
    
    async def create_capability_statement(self) -> FHIRCapabilityStatement:
        """Generate FHIR Capability Statement"""
        
        rest_resources = []
        for resource_type in FHIRResourceType:
            if resource_type.value in self.resource_store:
                resource_config = {
                    "type": resource_type.value,
                    "interaction": [
                        {"code": "read"},
                        {"code": "search"},
                        {"code": "create"},
                        {"code": "update"},
                        {"code": "delete"}
                    ]
                }
                rest_resources.append(resource_config)
        
        capability_statement = FHIRCapabilityStatement(
            name="Healthcare API FHIR Server",
            status="active",
            date=datetime.now(timezone.utc).isoformat(),
            publisher="Healthcare Technology Inc",
            description="Production FHIR R4 Server for Healthcare API",
            kind="instance",
            fhirVersion="4.0.1",
            format=["json"],
            rest=[{
                "mode": "server",
                "resource": rest_resources,
                "interaction": [
                    {"code": "transaction"},
                    {"code": "transaction-batch"},
                    {"code": "history"}
                ]
            }]
        )
        
        return capability_statement
    
    async def handle_interaction(
        self,
        resource_type: str,
        interaction: InteractionType,
        resource_data: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        search_params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Handle FHIR interaction"""
        
        try:
            if interaction == InteractionType.READ:
                return await self._handle_read(resource_type, resource_id)
            elif interaction == InteractionType.SEARCH:
                return await self._handle_search(resource_type, search_params or {})
            elif interaction == InteractionType.CREATE:
                return await self._handle_create(resource_type, resource_data)
            elif interaction == InteractionType.UPDATE:
                return await self._handle_update(resource_type, resource_id, resource_data)
            elif interaction == InteractionType.DELETE:
                return await self._handle_delete(resource_type, resource_id)
            elif interaction == InteractionType.BUNDLE:
                return await self._handle_bundle(resource_data)
            else:
                raise ValueError(f"Unsupported interaction type: {interaction}")
        
        except Exception as e:
            logger.error(f"Error handling FHIR interaction: {str(e)}")
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "processing",
                    "diagnostics": str(e)
                }]
            }
    
    async def _handle_read(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """Handle FHIR READ interaction"""
        resource_store = self.resource_store.get(resource_type, {})
        
        if resource_id not in resource_store:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "not-found",
                    "diagnostics": f"Resource {resource_type}/{resource_id} not found"
                }]
            }
        
        resource = resource_store[resource_id]
        return resource.dict() if hasattr(resource, 'dict') else resource
    
    async def _handle_search(self, resource_type: str, search_params: Dict[str, str]) -> Dict[str, Any]:
        """Handle FHIR SEARCH interaction"""
        resource_store = self.resource_store.get(resource_type, {})
        matched_resources = []
        
        # Basic search implementation
        for resource_id, resource in resource_store.items():
            resource_dict = resource.dict() if hasattr(resource, 'dict') else resource
            
            # Check search parameters
            matches = True
            for param, value in search_params.items():
                if not self._matches_search_param(resource_dict, param, value):
                    matches = False
                    break
            
            if matches:
                matched_resources.append({
                    "fullUrl": f"{self.base_url}/{resource_type}/{resource_id}",
                    "resource": resource_dict
                })
        
        # Create search bundle
        bundle = FHIRBundle(
            type="searchset",
            total=len(matched_resources),
            entry=matched_resources
        )
        
        return bundle.dict()
    
    def _matches_search_param(self, resource: Dict[str, Any], param: str, value: str) -> bool:
        """Check if resource matches search parameter"""
        # Simplified search - in production would use proper FHIR search
        if param in resource:
            return str(resource[param]) == value
        return False
    
    async def _handle_create(self, resource_type: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FHIR CREATE interaction"""
        
        # Validate resource
        validation_result = self.validator.validate_resource(resource_data)
        if not validation_result["valid"]:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "invalid",
                    "diagnostics": f"Resource validation failed: {validation_result['errors']}"
                }]
            }
        
        # Generate ID if not provided
        if not resource_data.get("id"):
            resource_data["id"] = str(uuid.uuid4())
        
        # Store resource
        resource_store = self.resource_store.setdefault(resource_type, {})
        resource_store[resource_data["id"]] = resource_data
        
        # Return created resource with Location header info
        return {
            **resource_data,
            "id": resource_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.now(timezone.utc).isoformat()
            }
        }
    
    async def _handle_update(self, resource_type: str, resource_id: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FHIR UPDATE interaction"""
        
        resource_store = self.resource_store.get(resource_type, {})
        
        if resource_id not in resource_store:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "not-found",
                    "diagnostics": f"Resource {resource_type}/{resource_id} not found"
                }]
            }
        
        # Validate updated resource
        resource_data["id"] = resource_id
        validation_result = self.validator.validate_resource(resource_data)
        if not validation_result["valid"]:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "invalid",
                    "diagnostics": f"Resource validation failed: {validation_result['errors']}"
                }]
            }
        
        # Update resource
        resource_store[resource_id] = resource_data
        
        return {
            **resource_data,
            "meta": {
                "versionId": "2",
                "lastUpdated": datetime.now(timezone.utc).isoformat()
            }
        }
    
    async def _handle_delete(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """Handle FHIR DELETE interaction"""
        
        resource_store = self.resource_store.get(resource_type, {})
        
        if resource_id not in resource_store:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "not-found",
                    "diagnostics": f"Resource {resource_type}/{resource_id} not found"
                }]
            }
        
        # Delete resource
        del resource_store[resource_id]
        
        # Return 200 OK with OperationOutcome
        return {
            "resourceType": "OperationOutcome",
            "issue": [{
                "severity": "information",
                "code": "informational",
                "diagnostics": f"Resource {resource_type}/{resource_id} deleted successfully"
            }]
        }
    
    async def _handle_bundle(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FHIR Bundle transaction"""
        
        # Validate bundle
        validation_result = self.validator.validate_bundle(bundle_data)
        if not validation_result["valid"]:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{
                    "severity": "error",
                    "code": "invalid",
                    "diagnostics": f"Bundle validation failed: {validation_result['bundle_errors']}"
                }]
            }
        
        # Process bundle entries
        response_entries = []
        
        for entry in bundle_data.get("entry", []):
            try:
                resource = entry.get("resource", {})
                request = entry.get("request", {})
                
                method = request.get("method", "POST")
                resource_type = resource.get("resourceType")
                
                if method == "POST":
                    result = await self._handle_create(resource_type, resource)
                elif method == "PUT":
                    resource_id = resource.get("id")
                    result = await self._handle_update(resource_type, resource_id, resource)
                elif method == "DELETE":
                    resource_id = request.get("url", "").split("/")[-1]
                    result = await self._handle_delete(resource_type, resource_id)
                else:
                    result = {
                        "resourceType": "OperationOutcome",
                        "issue": [{
                            "severity": "error",
                            "code": "not-supported",
                            "diagnostics": f"HTTP method {method} not supported in bundle"
                        }]
                    }
                
                response_entries.append({
                    "response": {
                        "status": "200" if isinstance(result, dict) and "OperationOutcome" not in result.get("resourceType", "") else "400",
                        "location": f"{self.base_url}/{resource_type}/{result.get('id', '')}" if result.get('id') else None,
                        "outcome": result
                    }
                })
            
            except Exception as e:
                response_entries.append({
                    "response": {
                        "status": "500",
                        "outcome": {
                            "resourceType": "OperationOutcome",
                            "issue": [{
                                "severity": "error",
                                "code": "exception",
                                "diagnostics": str(e)
                            }]
                        }
                    }
                })
        
        # Return transaction response bundle
        response_bundle = FHIRBundle(
            type="transaction-response",
            entry=response_entries
        )
        
        return response_bundle.dict()

# Example usage
if __name__ == "__main__":
    # Initialize FHIR server
    fhir_config = {
        "base_url": "https://api.healthcare.org/fhir",
        "version": "4.0.1"
    }
    
    fhir_server = FHIRServer(fhir_config)
    
    # Create capability statement
    capability_statement = asyncio.run(fhir_server.create_capability_statement())
    print(f"FHIR Server Capability Statement: {capability_statement.name}")
    print(f"Supported FHIR Version: {capability_statement.fhirVersion}")
    print(f"Server Kind: {capability_statement.kind}")