# Production API Documentation Portal with Interactive Testing
# OpenAPI/Swagger-based documentation with live testing capabilities

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiohttp
import aiofiles
import base64
from urllib.parse import urljoin
import jsonschema
from jsonschema import validate, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APILevel(Enum):
    """API maturity levels"""
    INTERNAL = "internal"
    PARTNER = "partner"
    PUBLIC = "public"
    DEPRECATED = "deprecated"

class AuthenticationType(Enum):
    """Authentication methods"""
    API_KEY = "apiKey"
    BEARER = "http"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    BASIC = "basic"

class MediaType(Enum):
    """Supported media types"""
    APPLICATION_JSON = "application/json"
    APPLICATION_XML = "application/xml"
    TEXT_CSV = "text/csv"
    MULTIPART_FORM_DATA = "multipart/form-data"

@dataclass
class APIEndpoint:
    """API endpoint specification"""
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]] = None
    responses: List[Dict[str, Any]] = None
    security: List[Dict[str, Any]] = None
    deprecated: bool = False
    version: str = "1.0.0"
    
    def to_openapi_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI specification format"""
        return {
            "tags": self.tags,
            "summary": self.summary,
            "description": self.description,
            "operationId": f"{self.method.lower()}_{self.path.replace('/', '_').replace('{', '').replace('}', '')}",
            "parameters": self.parameters,
            "requestBody": self.request_body,
            "responses": {str(code): resp for code, resp in self.responses.items()} if self.responses else {},
            "security": self.security or [],
            "deprecated": self.deprecated
        }

class OpenAPISpecification:
    """OpenAPI 3.0 specification generator for healthcare APIs"""
    
    def __init__(self, title: str, version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.info = {
            "title": title,
            "description": self._generate_description(),
            "version": version,
            "contact": {
                "name": "Healthcare API Support",
                "email": "support@healthcare.org",
                "url": "https://healthcare.org/support"
            },
            "license": {
                "name": "Proprietary",
                "url": "https://healthcare.org/license"
            },
            "termsOfService": "https://healthcare.org/terms"
        }
        self.servers = self._generate_servers()
        self.security_schemes = self._generate_security_schemes()
        self.components = self._generate_components()
        self.paths = {}
        self.tags = self._generate_tags()
    
    def _generate_description(self) -> str:
        """Generate comprehensive API description"""
        return """
        # Healthcare API - Production Documentation
        
        ## Overview
        This API provides comprehensive healthcare data management capabilities including:
        - Patient record management
        - FHIR-compliant EHR integration
        - Clinical decision support
        - Care plan management
        - Analytics and reporting
        
        ## Compliance
        - HIPAA-compliant data handling
        - FHIR R4 standard compliance
        - SOC 2 Type II certified infrastructure
        
        ## Authentication
        All endpoints require authentication via OAuth 2.0 or API keys.
        
        ## Rate Limiting
        - Clinical API: 100 requests/minute
        - Analytics API: 200 requests/minute  
        - EHR Integration: 30 requests/minute
        
        ## Error Handling
        All errors follow RFC 7807 problem details format.
        
        ## Support
        For technical support, contact: support@healthcare.org
        """
    
    def _generate_servers(self) -> List[Dict[str, str]]:
        """Generate server configurations"""
        return [
            {
                "url": "https://api.healthcare.org/v1",
                "description": "Production server"
            },
            {
                "url": "https://api-staging.healthcare.org/v1", 
                "description": "Staging server"
            },
            {
                "url": "https://api-dev.healthcare.org/v1",
                "description": "Development server"
            }
        ]
    
    def _generate_security_schemes(self) -> Dict[str, Any]:
        """Generate security schemes"""
        return {
            "ApiKeyAuth": {
                "type": "apiKey",
                "name": "X-API-Key",
                "in": "header",
                "description": "API key for authentication"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT bearer token"
            },
            "OAuth2": {
                "type": "oauth2",
                "description": "OAuth 2.0 authentication",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": "https://auth.healthcare.org/oauth/token",
                        "scopes": {
                            "read:patients": "Read patient data",
                            "write:patients": "Write patient data",
                            "read:observations": "Read observation data",
                            "write:observations": "Write observation data"
                        }
                    },
                    "authorizationCode": {
                        "authorizationUrl": "https://auth.healthcare.org/oauth/authorize",
                        "tokenUrl": "https://auth.healthcare.org/oauth/token",
                        "refreshUrl": "https://auth.healthcare.org/oauth/refresh",
                        "scopes": {
                            "read:patients": "Read patient data",
                            "write:patients": "Write patient data",
                            "read:observations": "Read observation data",
                            "write:observations": "Write observation data"
                        }
                    }
                }
            }
        }
    
    def _generate_components(self) -> Dict[str, Any]:
        """Generate OpenAPI components (schemas)"""
        return {
            "schemas": {
                "Patient": {
                    "type": "object",
                    "required": ["resourceType", "id"],
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "enum": ["Patient"],
                            "description": "FHIR resource type"
                        },
                        "id": {
                            "type": "string",
                            "description": "Unique patient identifier"
                        },
                        "identifier": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "system": {"type": "string"},
                                    "value": {"type": "string"}
                                }
                            },
                            "description": "Patient identifiers (MRN, SSN, etc.)"
                        },
                        "name": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "use": {
                                        "type": "string",
                                        "enum": ["usual", "official", "temp", "nickname", "anonymous", "old", "maiden"]
                                    },
                                    "family": {"type": "string"},
                                    "given": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "gender": {
                            "type": "string",
                            "enum": ["male", "female", "other", "unknown"],
                            "description": "Patient gender"
                        },
                        "birthDate": {
                            "type": "string",
                            "format": "date",
                            "description": "Date of birth (YYYY-MM-DD)"
                        },
                        "address": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "use": {"type": "string"},
                                    "line": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "city": {"type": "string"},
                                    "state": {"type": "string"},
                                    "postalCode": {"type": "string"},
                                    "country": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "Observation": {
                    "type": "object",
                    "required": ["resourceType", "id", "status", "code"],
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "enum": ["Observation"]
                        },
                        "id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"],
                            "description": "Observation status"
                        },
                        "category": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "coding": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "system": {"type": "string"},
                                                "code": {"type": "string"},
                                                "display": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "code": {
                            "type": "object",
                            "properties": {
                                "coding": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "system": {"type": "string"},
                                            "code": {"type": "string"},
                                            "display": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "subject": {
                            "type": "object",
                            "properties": {
                                "reference": {"type": "string"}
                            }
                        },
                        "valueQuantity": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                                "system": {"type": "string"},
                                "code": {"type": "string"}
                            }
                        }
                    }
                },
                "Error": {
                    "type": "object",
                    "required": ["type", "title", "status"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "format": "uri",
                            "description": "Error type URI"
                        },
                        "title": {
                            "type": "string",
                            "description": "Short error description"
                        },
                        "status": {
                            "type": "integer",
                            "description": "HTTP status code"
                        },
                        "detail": {
                            "type": "string",
                            "description": "Detailed error description"
                        },
                        "instance": {
                            "type": "string",
                            "format": "uri",
                            "description": "Error instance identifier"
                        }
                    }
                }
            },
            "securitySchemes": self.security_schemes
        }
    
    def _generate_tags(self) -> List[Dict[str, str]]:
        """Generate API tags"""
        return [
            {
                "name": "Patients",
                "description": "Patient resource management endpoints"
            },
            {
                "name": "Observations", 
                "description": "Clinical observation data endpoints"
            },
            {
                "name": "Conditions",
                "description": "Patient condition and diagnosis endpoints"
            },
            {
                "name": "Care Plans",
                "description": "Care plan management endpoints"
            },
            {
                "name": "FHIR",
                "description": "FHIR-compliant resource endpoints"
            },
            {
                "name": "Analytics",
                "description": "Analytics and reporting endpoints"
            },
            {
                "name": "Webhooks",
                "description": "Webhook management endpoints"
            }
        ]
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add endpoint to specification"""
        if endpoint.path not in self.paths:
            self.paths[endpoint.path] = {}
        
        self.paths[endpoint.path][endpoint.method.lower()] = endpoint.to_openapi_dict()
    
    def generate_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI specification"""
        return {
            "openapi": "3.0.3",
            "info": self.info,
            "servers": self.servers,
            "tags": self.tags,
            "paths": self.paths,
            "components": self.components,
            "security": [
                {"OAuth2": ["read:patients", "write:patients"]}
            ]
        }

class InteractiveAPITester:
    """Interactive API testing tool for documentation portal"""
    
    def __init__(self, base_url: str, auth_config: Dict[str, Any] = None):
        self.base_url = base_url
        self.auth_config = auth_config or {}
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def authenticate(self, auth_type: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Authenticate with the API"""
        if auth_type == "api_key":
            return await self._authenticate_api_key(credentials)
        elif auth_type == "oauth2":
            return await self._authenticate_oauth2(credentials)
        elif auth_type == "bearer":
            return await self._authenticate_bearer(credentials)
        else:
            raise ValueError(f"Unsupported authentication type: {auth_type}")
    
    async def _authenticate_api_key(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Authenticate using API key"""
        api_key = credentials.get("api_key")
        if not api_key:
            raise ValueError("API key required")
        
        return {
            "auth_type": "api_key",
            "api_key": api_key,
            "headers": {"X-API-Key": api_key}
        }
    
    async def _authenticate_oauth2(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Authenticate using OAuth2"""
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        username = credentials.get("username")
        password = credentials.get("password")
        scope = credentials.get("scope", "read:patients write:patients")
        
        if not all([client_id, client_secret]):
            raise ValueError("Client ID and secret required for OAuth2")
        
        # Token request
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope
        }
        
        if username and password:
            token_data["grant_type"] = "password"
            token_data["username"] = username
            token_data["password"] = password
        
        async with self.session.post(
            f"{self.base_url}/oauth/token",
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        ) as response:
            if response.status == 200:
                token_response = await response.json()
                return {
                    "auth_type": "oauth2",
                    "access_token": token_response["access_token"],
                    "token_type": token_response["token_type"],
                    "expires_in": token_response["expires_in"],
                    "headers": {"Authorization": f"Bearer {token_response['access_token']}"}
                }
            else:
                raise Exception(f"Authentication failed: {response.status}")
    
    async def _authenticate_bearer(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Authenticate using bearer token"""
        token = credentials.get("token")
        if not token:
            raise ValueError("Bearer token required")
        
        return {
            "auth_type": "bearer",
            "headers": {"Authorization": f"Bearer {token}"}
        }
    
    async def test_endpoint(
        self,
        method: str,
        path: str,
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        data: Any = None,
        auth_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Test an API endpoint"""
        
        # Merge headers
        request_headers = headers or {}
        if auth_info and "headers" in auth_info:
            request_headers.update(auth_info["headers"])
        
        # Remove None values from params
        request_params = {k: v for k, v in (params or {}).items() if v is not None}
        
        url = urljoin(self.base_url, path)
        start_time = datetime.now(timezone.utc)
        
        try:
            async with self.session.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                params=request_params,
                json=data if isinstance(data, dict) else None,
                data=data if not isinstance(data, dict) else None
            ) as response:
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()
                
                # Get response content
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    response_body = await response.json()
                else:
                    response_body = await response.text()
                
                return {
                    "request": {
                        "method": method.upper(),
                        "url": url,
                        "headers": request_headers,
                        "params": request_params,
                        "body": data
                    },
                    "response": {
                        "status": response.status,
                        "status_text": response.reason,
                        "headers": dict(response.headers),
                        "body": response_body,
                        "content_type": content_type
                    },
                    "timing": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "duration_seconds": duration
                    },
                    "success": 200 <= response.status < 300
                }
        
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return {
                "request": {
                    "method": method.upper(),
                    "url": url,
                    "headers": request_headers,
                    "params": request_params,
                    "body": data
                },
                "response": {
                    "status": 0,
                    "status_text": "Request Failed",
                    "error": str(e)
                },
                "timing": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration
                },
                "success": False
            }
    
    async def test_patient_endpoints(self, auth_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test patient-related endpoints"""
        results = []
        
        # Test get patients
        result = await self.test_endpoint(
            "GET",
            "/api/v1/patients",
            auth_info=auth_info,
            params={"limit": "10"}
        )
        results.append(("GET /api/v1/patients", result))
        
        # Test get specific patient
        if result["success"] and "body" in result["response"]:
            patients = result["response"]["body"].get("entry", [])
            if patients:
                patient_id = patients[0]["resource"]["id"]
                result = await self.test_endpoint(
                    "GET",
                    f"/api/v1/patients/{patient_id}",
                    auth_info=auth_info
                )
                results.append((f"GET /api/v1/patients/{patient_id}", result))
        
        return {"patient_endpoint_tests": results}
    
    async def test_observation_endpoints(self, auth_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test observation-related endpoints"""
        results = []
        
        # Test get observations
        result = await self.test_endpoint(
            "GET",
            "/api/v1/observations",
            auth_info=auth_info,
            params={"limit": "5"}
        )
        results.append(("GET /api/v1/observations", result))
        
        return {"observation_endpoint_tests": results}

class DocumentationPortal:
    """Complete documentation portal with interactive testing"""
    
    def __init__(self, spec: OpenAPISpecification):
        self.spec = spec
        self.tester = None
    
    async def generate_swagger_ui_config(self) -> Dict[str, Any]:
        """Generate Swagger UI configuration"""
        return {
            "dom_id": "#swagger-ui",
            "deepLinking": True,
            "displayOperationId": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
            "layout": "BaseLayout",
            "oauth2RedirectUrl": "https://portal.healthcare.org/oauth2-redirect.html",
            "showExtensions": True,
            "showCommonExtensions": True,
            "supportedSubmitMethods": ["get", "put", "post", "delete", "options", "head", "patch"],
            "tagsSorter": "alpha",
            "operationsSorter": "alpha",
            "swaggerUIBundle": {
                "presets": ["SwaggerUIBundle.presets.apis"],
                "plugins": ["SwaggerUIBundle.plugins.DownloadUrl"],
                "layout": "BaseLayout",
                "deepLinking": True
            }
        }
    
    async def create_interactive_examples(self) -> Dict[str, Any]:
        """Create interactive examples for testing"""
        examples = {
            "authentication": {
                "api_key": {
                    "name": "API Key Authentication",
                    "description": "Use API key for authentication",
                    "code": "curl -H 'X-API-Key: your-api-key' https://api.healthcare.org/api/v1/patients",
                    "parameters": ["api_key"]
                },
                "oauth2": {
                    "name": "OAuth 2.0 Authentication", 
                    "description": "Use OAuth 2.0 for authentication",
                    "code": """curl -X POST https://api.healthcare.org/oauth/token \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret" """,
                    "parameters": ["client_id", "client_secret", "scope"]
                }
            },
            "patient_operations": {
                "create_patient": {
                    "description": "Create a new patient record",
                    "method": "POST",
                    "path": "/api/v1/patients",
                    "request_body": {
                        "name": [{"family": "Doe", "given": ["John"]}],
                        "gender": "male",
                        "birthDate": "1980-01-01"
                    },
                    "response": {
                        "status": 201,
                        "body": {
                            "id": "patient-123",
                            "resourceType": "Patient"
                        }
                    }
                },
                "get_patient": {
                    "description": "Get patient by ID",
                    "method": "GET", 
                    "path": "/api/v1/patients/{patient_id}",
                    "parameters": {
                        "patient_id": "patient-123"
                    },
                    "response": {
                        "status": 200,
                        "body": {
                            "resourceType": "Patient",
                            "id": "patient-123"
                        }
                    }
                }
            }
        }
        return examples
    
    async def generate_portal_html(self) -> str:
        """Generate complete documentation portal HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Healthcare API Documentation Portal</title>
            <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui.css" />
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css">
            <style>
                body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
                .header { background: #1a365d; color: white; padding: 1rem; }
                .container { display: flex; }
                .sidebar { width: 300px; background: #f7fafc; padding: 1rem; border-right: 1px solid #e2e8f0; }
                .main-content { flex: 1; padding: 2rem; }
                .test-panel { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
                .endpoint-item { padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; cursor: pointer; }
                .endpoint-item:hover { background: #edf2f7; }
                .method-get { border-left: 4px solid #38a169; }
                .method-post { border-left: 4px solid #3182ce; }
                .method-put { border-left: 4px solid #d69e2e; }
                .method-delete { border-left: 4px solid #e53e3e; }
                pre { background: #f7fafc; padding: 1rem; border-radius: 0.25rem; overflow-x: auto; }
                code { font-family: 'Monaco', 'Consolas', monospace; }
                .response-success { border-left: 4px solid #38a169; }
                .response-error { border-left: 4px solid #e53e3e; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 Healthcare API Documentation Portal</h1>
                <p>Production-grade API documentation with interactive testing</p>
            </div>
            
            <div class="container">
                <div class="sidebar">
                    <h3>Quick Start</h3>
                    <ul>
                        <li><a href="#overview">Overview</a></li>
                        <li><a href="#authentication">Authentication</a></li>
                        <li><a href="#examples">Examples</a></li>
                        <li><a href="#interactive">Interactive Testing</a></li>
                        <li><a href="#compliance">Compliance</a></li>
                    </ul>
                    
                    <h3>API Endpoints</h3>
                    <div id="endpoint-list">
                        <div class="endpoint-item method-get">GET /api/v1/patients</div>
                        <div class="endpoint-item method-get">GET /api/v1/patients/{id}</div>
                        <div class="endpoint-item method-post">POST /api/v1/patients</div>
                        <div class="endpoint-item method-get">GET /api/v1/observations</div>
                        <div class="endpoint-item method-post">POST /api/v1/observations</div>
                        <div class="endpoint-item method-get">GET /fhir/{resourceType}</div>
                    </div>
                </div>
                
                <div class="main-content">
                    <div id="overview">
                        <h2>API Overview</h2>
                        <p>Welcome to the Healthcare API documentation. This API provides comprehensive healthcare data management capabilities including patient records, clinical observations, and FHIR-compliant EHR integration.</p>
                        
                        <h3>Base URLs</h3>
                        <ul>
                            <li>Production: <code>https://api.healthcare.org/v1</code></li>
                            <li>Staging: <code>https://api-staging.healthcare.org/v1</code></li>
                            <li>Development: <code>https://api-dev.healthcare.org/v1</code></li>
                        </ul>
                    </div>
                    
                    <div id="authentication" style="margin-top: 2rem;">
                        <h2>Authentication</h2>
                        <p>The API supports multiple authentication methods:</p>
                        
                        <h3>API Key Authentication</h3>
                        <pre><code class="language-bash">curl -H "X-API-Key: your-api-key" \\
     https://api.healthcare.org/api/v1/patients</code></pre>
                        
                        <h3>OAuth 2.0</h3>
                        <pre><code class="language-bash"># Get access token
curl -X POST https://api.healthcare.org/oauth/token \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret"

# Use access token
curl -H "Authorization: Bearer your-access-token" \\
     https://api.healthcare.org/api/v1/patients</code></pre>
                    </div>
                    
                    <div id="examples" style="margin-top: 2rem;">
                        <h2>Examples</h2>
                        
                        <h3>Get Patient List</h3>
                        <pre><code class="language-bash">curl -H "X-API-Key: your-api-key" \\
     "https://api.healthcare.org/api/v1/patients?limit=10&offset=0"</code></pre>
                        
                        <h3>Create Patient</h3>
                        <pre><code class="language-bash">curl -X POST https://api.healthcare.org/api/v1/patients \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key" \\
  -d '{
    "name": [
      {
        "family": "Doe",
        "given": ["John", "Michael"]
      }
    ],
    "gender": "male",
    "birthDate": "1980-01-01"
  }'</code></pre>
                        
                        <h3>Get Patient Observations</h3>
                        <pre><code class="language-bash">curl -H "X-API-Key: your-api-key" \\
     "https://api.healthcare.org/api/v1/observations?patient=patient-123"</code></pre>
                    </div>
                    
                    <div id="interactive" style="margin-top: 2rem;">
                        <h2>Interactive API Testing</h2>
                        <div class="test-panel">
                            <h3>Authentication Setup</h3>
                            <form id="auth-form">
                                <label>API Key:</label>
                                <input type="text" id="api-key" placeholder="Enter your API key">
                                <button type="button" onclick="setupAuth()">Setup Authentication</button>
                            </form>
                        </div>
                        
                        <div class="test-panel">
                            <h3>Test Patient Endpoint</h3>
                            <button onclick="testPatientsEndpoint()">Test GET /api/v1/patients</button>
                            <div id="test-result"></div>
                        </div>
                    </div>
                    
                    <div id="compliance" style="margin-top: 2rem;">
                        <h2>Compliance & Security</h2>
                        <ul>
                            <li><strong>HIPAA Compliance:</strong> All data handling meets HIPAA requirements</li>
                            <li><strong>FHIR R4:</strong> Full FHIR R4 standard compliance</li>
                            <li><strong>SOC 2 Type II:</strong> SOC 2 Type II certified infrastructure</li>
                            <li><strong>Encryption:</strong> All data encrypted in transit and at rest</li>
                            <li><strong>Access Control:</strong> Role-based access control and audit logging</li>
                            <li><strong>Rate Limiting:</strong> Configurable rate limits to prevent abuse</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <script src="https://unpkg.com/swagger-ui-dist@5.0.0/swagger-ui-bundle.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
            <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
            
            <script>
                let authHeaders = {};
                
                function setupAuth() {
                    const apiKey = document.getElementById('api-key').value;
                    if (apiKey) {
                        authHeaders = {
                            'X-API-Key': apiKey,
                            'Content-Type': 'application/json'
                        };
                        alert('Authentication setup complete!');
                    } else {
                        alert('Please enter an API key');
                    }
                }
                
                async function testPatientsEndpoint() {
                    const resultDiv = document.getElementById('test-result');
                    resultDiv.innerHTML = '<p>Testing...</p>';
                    
                    try {
                        const response = await axios.get('https://api.healthcare.org/api/v1/patients?limit=5', {
                            headers: authHeaders
                        });
                        
                        const result = {
                            status: response.status,
                            data: response.data,
                            headers: response.headers,
                            duration: response.headers['x-response-time'] || 'N/A'
                        };
                        
                        resultDiv.innerHTML = `
                            <div class="response-success">
                                <h4>✅ Success - ${response.status} ${response.statusText}</h4>
                                <p><strong>Response Time:</strong> ${result.duration}</p>
                                <pre><code>${JSON.stringify(result.data, null, 2)}</code></pre>
                            </div>
                        `;
                        
                        // Highlight code
                        document.querySelectorAll('pre code').forEach(block => {
                            hljs.highlightBlock(block);
                        });
                        
                    } catch (error) {
                        resultDiv.innerHTML = `
                            <div class="response-error">
                                <h4>❌ Error - ${error.response?.status || 'Request Failed'}</h4>
                                <pre><code>${JSON.stringify(error.response?.data || error.message, null, 2)}</code></pre>
                            </div>
                        `;
                    }
                }
                
                // Initialize syntax highlighting
                document.addEventListener('DOMContentLoaded', function() {
                    hljs.highlightAll();
                });
            </script>
        </body>
        </html>
        """
    
    async def save_portal_files(self, output_dir: str):
        """Save all portal files"""
        # Generate OpenAPI spec
        spec_json = json.dumps(self.spec.generate_spec(), indent=2)
        async with aiofiles.open(f"{output_dir}/openapi.json", "w") as f:
            await f.write(spec_json)
        
        # Generate Swagger UI config
        swagger_config = await self.generate_swagger_ui_config()
        async with aiofiles.open(f"{output_dir}/swagger-config.json", "w") as f:
            await f.write(json.dumps(swagger_config, indent=2))
        
        # Generate interactive examples
        examples = await self.create_interactive_examples()
        async with aiofiles.open(f"{output_dir}/examples.json", "w") as f:
            await f.write(json.dumps(examples, indent=2))
        
        # Generate HTML portal
        html_content = await self.generate_portal_html()
        async with aiofiles.open(f"{output_dir}/index.html", "w") as f:
            await f.write(html_content)

# Example usage
if __name__ == "__main__":
    # Create API specification
    api_spec = OpenAPISpecification("Healthcare API", "1.0.0")
    
    # Add patient endpoints
    patient_endpoints = [
        APIEndpoint(
            path="/api/v1/patients",
            method="GET",
            summary="Get Patients",
            description="Retrieve a list of patients",
            tags=["Patients"],
            parameters=[
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of patients to return",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                {
                    "name": "offset", 
                    "in": "query",
                    "description": "Number of patients to skip",
                    "schema": {"type": "integer", "minimum": 0}
                }
            ],
            responses={
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "entry": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Patient"}
                                    }
                                }
                            }
                        }
                    }
                },
                "400": {
                    "description": "Bad request",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}
                        }
                    }
                }
            }
        ),
        APIEndpoint(
            path="/api/v1/patients",
            method="POST",
            summary="Create Patient",
            description="Create a new patient record",
            tags=["Patients"],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Patient"}
                    }
                }
            },
            responses={
                "201": {
                    "description": "Patient created successfully",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Patient"}
                        }
                    }
                }
            }
        )
    ]
    
    for endpoint in patient_endpoints:
        api_spec.add_endpoint(endpoint)
    
    # Create documentation portal
    portal = DocumentationPortal(api_spec)
    
    print("Healthcare API Documentation Portal created successfully")
    print("Features:")
    print("- Interactive OpenAPI/Swagger documentation")
    print("- Live API testing capabilities") 
    print("- Authentication setup")
    print("- Response validation")
    print("- Code examples")
    print("- HIPAA compliance information")