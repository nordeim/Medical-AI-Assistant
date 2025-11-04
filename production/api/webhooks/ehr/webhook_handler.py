# Production Webhook System for EHR/EMR Integration
# HIPAA-compliant webhook handlers for third-party healthcare systems

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import base64
import os
from cryptography.fernet import Fernet
import aiohttp
import aiofiles
from contextlib import asynccontextmanager

# Configure logging for HIPAA compliance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Healthcare webhook event types"""
    PATIENT_CREATED = "patient.created"
    PATIENT_UPDATED = "patient.updated"
    PATIENT_DELETED = "patient.deleted"
    OBSERVATION_CREATED = "observation.created"
    OBSERVATION_UPDATED = "observation.updated"
    MEDICATION_ORDERED = "medication.ordered"
    MEDICATION_ADMINISTERED = "medication.administered"
    APPOINTMENT_SCHEDULED = "appointment.scheduled"
    APPOINTMENT_CANCELLED = "appointment.cancelled"
    CARE_PLAN_CREATED = "careplan.created"
    CARE_PLAN_UPDATED = "careplan.updated"
    CLINICAL_NOTE_CREATED = "note.created"
    LAB_RESULT_AVAILABLE = "lab.result.available"
    DIAGNOSIS_ASSIGNED = "diagnosis.assigned"
    ALLERGY_RECORDED = "allergy.recorded"

class EHRSystem(Enum):
    """Supported EHR/EMR systems"""
    EPIC = "epic"
    CERNER = "cerner"
    ALLSCRIPTS = "allscripts"
    ATHENAHEALTH = "athenahealth"
    MEDITECH = "meditech"
    eCLINICALWORKS = "eclinicalworks"
    NEXTGEN = "nextgen"
    AMAZON_HEALTHLAKE = "healthlake"

class SecurityLevel(Enum):
    """Security levels for webhooks"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PHI = "phi"

@dataclass
class WebhookPayload:
    """Structured webhook payload for healthcare data"""
    event_type: EventType
    event_id: str
    timestamp: datetime
    source_system: EHRSystem
    resource_type: str
    resource_id: str
    resource_data: Dict[str, Any]
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    provider_id: Optional[str] = None
    facility_id: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['source_system'] = self.source_system.value
        data['security_level'] = self.security_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookPayload':
        """Create payload from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        data['source_system'] = EHRSystem(data['source_system'])
        data['security_level'] = SecurityLevel(data['security_level'])
        return cls(**data)

class HIPAAWebhookHandler:
    """HIPAA-compliant webhook handler for EHR integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.active_webhooks: Dict[str, Dict] = {}
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds
        self.timeout = 30  # seconds
        
    def _get_encryption_key(self) -> bytes:
        """Get encryption key for sensitive data"""
        key = os.getenv('WEBHOOK_ENCRYPTION_KEY')
        if not key:
            # Generate key for development
            key = base64.urlsafe_b64encode(b'development-key-for-hipaa-webhooks')
        return key
    
    async def register_webhook(
        self,
        webhook_id: str,
        url: str,
        event_types: List[EventType],
        secret: str,
        security_level: SecurityLevel,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Register a new webhook endpoint"""
        
        webhook_config = {
            'id': webhook_id,
            'url': url,
            'event_types': [event.value for event in event_types],
            'secret': secret,
            'security_level': security_level.value,
            'metadata': metadata or {},
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_triggered': None,
            'trigger_count': 0,
            'failure_count': 0
        }
        
        self.active_webhooks[webhook_id] = webhook_config
        await self._persist_webhook_config(webhook_config)
        
        logger.info(f"Registered webhook {webhook_id} for events: {[event.value for event in event_types]}")
        return {'webhook_id': webhook_id, 'status': 'registered'}
    
    async def unregister_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Unregister a webhook endpoint"""
        if webhook_id in self.active_webhooks:
            del self.active_webhooks[webhook_id]
            await self._remove_webhook_config(webhook_id)
            logger.info(f"Unregistered webhook {webhook_id}")
            return {'webhook_id': webhook_id, 'status': 'unregistered'}
        raise ValueError(f"Webhook {webhook_id} not found")
    
    async def trigger_webhooks(
        self,
        payload: WebhookPayload,
        source_system: EHRSystem,
        organization_id: str = None
    ) -> Dict[str, Any]:
        """Trigger webhooks for a specific event"""
        
        triggered_webhooks = []
        failed_webhooks = []
        
        # Find relevant webhooks
        relevant_webhooks = self._find_relevant_webhooks(payload.event_type)
        
        for webhook_id, webhook_config in relevant_webhooks.items():
            try:
                # Security check
                if not self._check_webhook_security(payload, webhook_config):
                    logger.warning(f"Security check failed for webhook {webhook_id}")
                    continue
                
                # Prepare and send webhook
                result = await self._send_webhook(webhook_config, payload)
                if result['success']:
                    triggered_webhooks.append({
                        'webhook_id': webhook_id,
                        'status': 'triggered',
                        'response_status': result['status_code']
                    })
                else:
                    failed_webhooks.append({
                        'webhook_id': webhook_id,
                        'status': 'failed',
                        'error': result['error']
                    })
            
            except Exception as e:
                logger.error(f"Failed to trigger webhook {webhook_id}: {str(e)}")
                failed_webhooks.append({
                    'webhook_id': webhook_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        return {
            'event_type': payload.event_type.value,
            'event_id': payload.event_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source_system': source_system.value,
            'triggered_count': len(triggered_webhooks),
            'failed_count': len(failed_webhooks),
            'triggered_webhooks': triggered_webhooks,
            'failed_webhooks': failed_webhooks
        }
    
    def _find_relevant_webhooks(self, event_type: EventType) -> Dict[str, Dict]:
        """Find webhooks that should receive this event"""
        relevant = {}
        
        for webhook_id, config in self.active_webhooks.items():
            if config['status'] == 'active':
                webhook_events = [EventType(event) for event in config['event_types']]
                if event_type in webhook_events:
                    relevant[webhook_id] = config
        
        return relevant
    
    def _check_webhook_security(self, payload: WebhookPayload, webhook_config: Dict) -> bool:
        """Perform security checks for webhook delivery"""
        
        # Check if webhook is allowed to receive this security level of data
        webhook_security_level = SecurityLevel(webhook_config['security_level'])
        
        # PHII data can only be sent to PHI-capable webhooks
        if payload.security_level == SecurityLevel.PHI and webhook_security_level != SecurityLevel.PHI:
            return False
        
        # Check organization access if specified
        webhook_orgs = webhook_config.get('metadata', {}).get('allowed_organizations', [])
        if webhook_orgs and payload.resource_data.get('organization_id') not in webhook_orgs:
            return False
        
        return True
    
    async def _send_webhook(
        self,
        webhook_config: Dict,
        payload: WebhookPayload
    ) -> Dict[str, Any]:
        """Send webhook payload to endpoint"""
        
        url = webhook_config['url']
        secret = webhook_config['secret']
        
        # Prepare payload
        webhook_data = self._prepare_webhook_data(payload, webhook_config)
        
        # Generate signature
        signature = self._generate_signature(webhook_data, secret)
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Healthcare-Webhook-System/1.0',
            'X-Webhook-ID': webhook_config['id'],
            'X-Event-Type': payload.event_type.value,
            'X-Timestamp': datetime.now(timezone.utc).isoformat(),
            'X-Signature': signature,
            'X-Security-Level': payload.security_level.value
        }
        
        # Send webhook with retry logic
        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=webhook_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response_data = {
                            'success': True,
                            'status_code': response.status,
                            'response_body': await response.text()
                        }
                        
                        # Update webhook statistics
                        await self._update_webhook_stats(webhook_config['id'], True, response.status)
                        
                        return response_data
            
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.timeout} seconds"
                if attempt == self.retry_attempts - 1:
                    await self._update_webhook_stats(webhook_config['id'], False, None, error_msg)
                    return {'success': False, 'error': error_msg}
            
            except Exception as e:
                error_msg = f"HTTP error: {str(e)}"
                if attempt == self.retry_attempts - 1:
                    await self._update_webhook_stats(webhook_config['id'], False, None, error_msg)
                    return {'success': False, 'error': error_msg}
            
            # Wait before retry
            await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return {'success': False, 'error': 'Max retry attempts reached'}
    
    def _prepare_webhook_data(self, payload: WebhookPayload, webhook_config: Dict) -> Dict[str, Any]:
        """Prepare webhook data for delivery"""
        
        # Base webhook data
        webhook_data = {
            'webhook_id': webhook_config['id'],
            'event': payload.to_dict(),
            'environment': self.config.get('environment', 'production'),
            'delivery_timestamp': datetime.now(timezone.utc).isoformat(),
            'request_id': payload.event_id
        }
        
        # Encrypt sensitive data if needed
        if payload.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.PHI]:
            webhook_data['sensitive_data'] = self._encrypt_sensitive_data(payload.resource_data)
        else:
            webhook_data['resource_data'] = payload.resource_data
        
        # Add webhook-specific metadata
        if webhook_config.get('metadata', {}).get('include_audit_trail'):
            webhook_data['audit_trail'] = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source_system': payload.source_system.value,
                'security_level': payload.security_level.value,
                'event_id': payload.event_id
            }
        
        return webhook_data
    
    def _encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive healthcare data"""
        json_data = json.dumps(data)
        encrypted_data = self.cipher_suite.encrypt(json_data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def _generate_signature(self, data: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        payload = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def _update_webhook_stats(
        self,
        webhook_id: str,
        success: bool,
        status_code: int = None,
        error: str = None
    ):
        """Update webhook delivery statistics"""
        
        webhook_config = self.active_webhooks[webhook_id]
        webhook_config['trigger_count'] += 1
        webhook_config['last_triggered'] = datetime.now(timezone.utc).isoformat()
        
        if not success:
            webhook_config['failure_count'] += 1
            webhook_config.get('errors', []).append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': error,
                'status_code': status_code
            })
        
        await self._persist_webhook_config(webhook_config)
    
    async def _persist_webhook_config(self, config: Dict[str, Any]):
        """Persist webhook configuration"""
        config_path = f"/workspace/production/api/webhooks/{config['id']}_config.json"
        async with aiofiles.open(config_path, 'w') as f:
            await f.write(json.dumps(config, indent=2))
    
    async def _remove_webhook_config(self, webhook_id: str):
        """Remove webhook configuration file"""
        config_path = f"/workspace/production/api/webhooks/{webhook_id}_config.json"
        if os.path.exists(config_path):
            os.remove(config_path)
    
    async def get_webhook_statistics(self, webhook_id: str = None) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        
        if webhook_id:
            if webhook_id in self.active_webhooks:
                config = self.active_webhooks[webhook_id]
                success_rate = (
                    (config['trigger_count'] - config['failure_count']) / 
                    max(config['trigger_count'], 1) * 100
                )
                
                return {
                    'webhook_id': webhook_id,
                    'status': config['status'],
                    'trigger_count': config['trigger_count'],
                    'failure_count': config['failure_count'],
                    'success_rate_percent': round(success_rate, 2),
                    'last_triggered': config['last_triggered'],
                    'security_level': config['security_level']
                }
            else:
                raise ValueError(f"Webhook {webhook_id} not found")
        else:
            # Return statistics for all webhooks
            all_stats = []
            for wid in self.active_webhooks:
                stats = await self.get_webhook_statistics(wid)
                all_stats.append(stats)
            
            return {
                'total_webhooks': len(all_stats),
                'active_webhooks': len([s for s in all_stats if s['status'] == 'active']),
                'webhook_statistics': all_stats
            }

# Epic EHR Webhook Handler
class EpicWebhookHandler(HIPAAWebhookHandler):
    """Specialized handler for Epic EHR system webhooks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.epic_api_url = config.get('epic_api_url')
        self.epic_client_id = config.get('epic_client_id')
        self.epic_client_secret = config.get('epic_client_secret')
    
    async def handle_epic_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Epic-specific webhook format"""
        
        # Convert Epic webhook format to standard payload
        event_mapping = {
            'patient_update': EventType.PATIENT_UPDATED,
            'lab_result_update': EventType.LAB_RESULT_AVAILABLE,
            'order_update': EventType.MEDICATION_ORDERED
        }
        
        epic_event = webhook_data.get('eventType', '').lower()
        event_type = event_mapping.get(epic_event, EventType.OBSERVATION_CREATED)
        
        # Create standard payload
        payload = WebhookPayload(
            event_type=event_type,
            event_id=webhook_data.get('eventId'),
            timestamp=datetime.fromisoformat(webhook_data.get('timestamp')),
            source_system=EHRSystem.EPIC,
            resource_type=webhook_data.get('resourceType'),
            resource_id=webhook_data.get('resourceId'),
            resource_data=webhook_data.get('data', {}),
            patient_id=webhook_data.get('patientId'),
            security_level=SecurityLevel.PHI
        )
        
        # Trigger webhooks
        return await self.trigger_webhooks(payload, EHRSystem.EPIC)

# Cerner EHR Webhook Handler
class CernerWebhookHandler(HIPAAWebhookHandler):
    """Specialized handler for Cerner EHR system webhooks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cerner_api_url = config.get('cerner_api_url')
        self.cerner_tenant_id = config.get('cerner_tenant_id')
    
    async def handle_cerner_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Cerner-specific webhook format"""
        
        # Map Cerner event types to standard events
        event_mapping = {
            'patient.demographic.change': EventType.PATIENT_UPDATED,
            'encounter.start': EventType.PATIENT_UPDATED,
            'encounter.end': EventType.PATIENT_UPDATED,
            'order.new': EventType.MEDICATION_ORDERED,
            'result.notification': EventType.LAB_RESULT_AVAILABLE
        }
        
        cerner_event = webhook_data.get('topic', '').lower()
        event_type = event_mapping.get(cerner_event, EventType.OBSERVATION_CREATED)
        
        # Create standard payload
        payload = WebhookPayload(
            event_type=event_type,
            event_id=webhook_data.get('id'),
            timestamp=datetime.fromisoformat(webhook_data.get('publishedAt')),
            source_system=EHRSystem.CERNER,
            resource_type=webhook_data.get('resourceType'),
            resource_id=webhook_data.get('resource'),
            resource_data=webhook_data.get('data', {}),
            patient_id=webhook_data.get('subject', {}).get('reference', {}).split('/')[-1] if webhook_data.get('subject') else None,
            security_level=SecurityLevel.PHI
        )
        
        # Trigger webhooks
        return await self.trigger_webhooks(payload, EHRSystem.CERNER)

# Webhook Management API
class WebhookAPI:
    """API for managing healthcare webhooks"""
    
    def __init__(self, handlers: Dict[EHRSystem, HIPAAWebhookHandler]):
        self.handlers = handlers
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for webhook system"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'active_handlers': len(self.handlers),
            'handler_statuses': {
                system.value: 'active' for system in self.handlers.keys()
            }
        }

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    webhook_config = {
        'environment': 'production',
        'epic_api_url': 'https://fhir.epic.com/interconnect-fhir-oauth',
        'cerner_api_url': 'https://fhir-open.cerner.com/r4',
        'epic_client_id': 'your-epic-client-id',
        'epic_client_secret': 'your-epic-client-secret',
        'cerner_tenant_id': 'your-cerner-tenant-id'
    }
    
    # Initialize webhook handlers
    epic_handler = EpicWebhookHandler(webhook_config)
    cerner_handler = CernerWebhookHandler(webhook_config)
    
    handlers = {
        EHRSystem.EPIC: epic_handler,
        EHRSystem.CERNER: cerner_handler
    }
    
    webhook_api = WebhookAPI(handlers)
    
    print("Healthcare Webhook System initialized successfully")
    print("Handlers for systems:", [system.value for system in handlers.keys()])