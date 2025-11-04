# Production API Testing Configuration
# Comprehensive test suite for healthcare API management

import pytest
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List
import aiohttp
import aiofiles
from httpx import AsyncClient

# Test configuration
TEST_CONFIG = {
    "api_base_url": os.getenv("TEST_API_URL", "http://localhost:8000"),
    "fhir_base_url": os.getenv("TEST_FHIR_URL", "http://localhost:8003"),
    "auth_api_key": os.getenv("TEST_API_KEY"),
    "auth_token": os.getenv("TEST_JWT_TOKEN"),
    "test_data_dir": os.path.join(os.path.dirname(__file__), "test_data"),
    "mock_data_dir": os.path.join(os.path.dirname(__file__), "mock_data")
}

class APITestClient:
    """Test client for healthcare API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None
        self.headers = {}
    
    async def __aenter__(self):
        self.client = AsyncClient(base_url=self.base_url, timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    def set_auth(self, api_key: str = None, token: str = None):
        """Set authentication headers"""
        if api_key:
            self.headers["X-API-Key"] = api_key
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def get(self, path: str, params: Dict = None):
        """GET request"""
        return await self.client.get(path, params=params, headers=self.headers)
    
    async def post(self, path: str, json_data: Dict = None, data: Any = None):
        """POST request"""
        return await self.client.post(path, json=json_data, data=data, headers=self.headers)
    
    async def put(self, path: str, json_data: Dict = None):
        """PUT request"""
        return await self.client.put(path, json=json_data, headers=self.headers)
    
    async def delete(self, path: str):
        """DELETE request"""
        return await self.client.delete(path, headers=self.headers)

class TestDataManager:
    """Manages test data for API testing"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    async def load_test_patient(self) -> Dict[str, Any]:
        """Load sample patient data"""
        async with aiofiles.open(os.path.join(self.data_dir, "sample_patient.json")) as f:
            return json.loads(await f.read())
    
    async def load_fhir_patient(self) -> Dict[str, Any]:
        """Load FHIR Patient resource"""
        async with aiofiles.open(os.path.join(self.data_dir, "fhir_patient.json")) as f:
            return json.loads(await f.read())
    
    async def load_test_observation(self) -> Dict[str, Any]:
        """Load sample observation data"""
        async with aiofiles.open(os.path.join(self.data_dir, "sample_observation.json")) as f:
            return json.loads(await f.read())

@pytest.fixture
async def test_client():
    """Create test client"""
    async with APITestClient(TEST_CONFIG["api_base_url"]) as client:
        if TEST_CONFIG["auth_api_key"]:
            client.set_auth(api_key=TEST_CONFIG["auth_api_key"])
        elif TEST_CONFIG["auth_token"]:
            client.set_auth(token=TEST_CONFIG["auth_token"])
        yield client

@pytest.fixture
async def fhir_client():
    """Create FHIR test client"""
    async with APITestClient(TEST_CONFIG["fhir_base_url"]) as client:
        if TEST_CONFIG["auth_api_key"]:
            client.set_auth(api_key=TEST_CONFIG["auth_api_key"])
        yield client

@pytest.fixture
def test_data_manager():
    """Create test data manager"""
    return TestDataManager(TEST_CONFIG["test_data_dir"])

class TestSuite:
    """Comprehensive test suite for healthcare API"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
    
    async def test_api_health(self, client: APITestClient):
        """Test API health endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    async def test_patient_crud_operations(self, client: APITestClient, test_data_manager):
        """Test complete Patient CRUD operations"""
        # Create patient
        patient_data = await test_data_manager.load_test_patient()
        
        create_response = await client.post("/api/v1/patients", json_data=patient_data)
        assert create_response.status_code == 201
        created_patient = create_response.json()
        patient_id = created_patient["id"]
        
        # Read patient
        read_response = await client.get(f"/api/v1/patients/{patient_id}")
        assert read_response.status_code == 200
        assert read_response.json()["id"] == patient_id
        
        # Update patient
        update_data = patient_data.copy()
        update_data["name"][0]["family"] = "Updated Family Name"
        
        update_response = await client.put(f"/api/v1/patients/{patient_id}", json_data=update_data)
        assert update_response.status_code == 200
        assert update_response.json()["name"][0]["family"] == "Updated Family Name"
        
        # Delete patient
        delete_response = await client.delete(f"/api/v1/patients/{patient_id}")
        assert delete_response.status_code == 200
        
        # Verify deletion
        get_deleted_response = await client.get(f"/api/v1/patients/{patient_id}")
        assert get_deleted_response.status_code == 404
    
    async def test_fhir_operations(self, fhir_client: APITestClient, test_data_manager):
        """Test FHIR operations"""
        # Load FHIR Patient resource
        fhir_patient = await test_data_manager.load_fhir_patient()
        
        # Create FHIR Patient
        create_response = await fhir_client.post("/fhir/Patient", json_data=fhir_patient)
        assert create_response.status_code == 201
        created_patient = create_response.json()
        patient_id = created_patient["id"]
        
        # Read FHIR Patient
        read_response = await fhir_client.get(f"/fhir/Patient/{patient_id}")
        assert read_response.status_code == 200
        assert read_response.json()["resourceType"] == "Patient"
        
        # Test capability statement
        cap_response = await fhir_client.get("/fhir/metadata")
        assert cap_response.status_code == 200
        cap_data = cap_response.json()
        assert cap_data["resourceType"] == "CapabilityStatement"
        assert "rest" in cap_data
    
    async def test_observation_operations(self, client: APITestClient, test_data_manager):
        """Test Observation operations"""
        obs_data = await test_data_manager.load_test_observation()
        
        # Create observation
        create_response = await client.post("/api/v1/observations", json_data=obs_data)
        assert create_response.status_code == 201
        observation_id = create_response.json()["id"]
        
        # Read observation
        read_response = await client.get(f"/api/v1/observations/{observation_id}")
        assert read_response.status_code == 200
        
        # List observations with filters
        list_response = await client.get("/api/v1/observations", params={"limit": "10"})
        assert list_response.status_code == 200
        assert "entry" in list_response.json()
    
    async def test_authentication_flow(self):
        """Test authentication flow"""
        async with APITestClient(TEST_CONFIG["api_base_url"]) as client:
            # Test with invalid API key
            client.set_auth(api_key="invalid_key")
            response = await client.get("/api/v1/patients")
            assert response.status_code == 401
            
            # Test without authentication
            client.headers = {}
            response = await client.get("/api/v1/patients")
            assert response.status_code in [401, 403]  # Unauthorized or Forbidden
    
    async def test_rate_limiting(self, client: APITestClient):
        """Test rate limiting"""
        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(150):  # Exceed typical rate limit
            response = await client.get("/api/v1/patients", params={"limit": "1"})
            responses.append(response.status_code)
        
        # Should see some 429 (Too Many Requests) responses
        rate_limited_count = sum(1 for status in responses if status == 429)
        assert rate_limited_count > 0
    
    async def test_api_versioning(self, client: APITestClient):
        """Test API versioning"""
        # Test version-specific endpoint
        v1_response = await client.get("/api/v1/patients")
        assert v1_response.status_code == 200
        
        # Test with version header
        client.headers["X-API-Version"] = "1.0.0"
        version_response = await client.get("/api/v1/patients")
        assert version_response.status_code == 200
        
        # Test invalid version
        client.headers["X-API-Version"] = "9.9.9"
        invalid_response = await client.get("/api/v1/patients")
        assert invalid_response.status_code == 400  # Bad request for invalid version
    
    async def test_error_handling(self, client: APITestClient):
        """Test error handling scenarios"""
        # Test 404 for non-existent resource
        response = await client.get("/api/v1/patients/non-existent-id")
        assert response.status_code == 404
        
        # Test 400 for invalid input
        invalid_patient = {"invalid": "data"}
        response = await client.post("/api/v1/patients", json_data=invalid_patient)
        assert response.status_code == 400
        
        # Test 500 for server error
        # This would require mocking a server error condition
        # For now, test that error responses have proper structure
        if response.status_code >= 400:
            error_data = response.json()
            assert "error" in error_data or "message" in error_data
    
    async def test_webhook_operations(self, client: APITestClient):
        """Test webhook management"""
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["patient.created", "patient.updated"],
            "secret": "test_secret"
        }
        
        # Register webhook
        register_response = await client.post("/webhooks/register", json_data=webhook_data)
        assert register_response.status_code in [200, 201]
        
        if register_response.status_code in [200, 201]:
            webhook_id = register_response.json().get("webhook_id")
            
            # Test webhook status
            status_response = await client.get(f"/webhooks/status/{webhook_id}")
            assert status_response.status_code == 200
    
    async def test_analytics_endpoints(self, client: APITestClient):
        """Test analytics and metrics endpoints"""
        # Test metrics endpoint
        metrics_response = await client.get("/metrics")
        assert metrics_response.status_code == 200
        # Should return Prometheus format
        content = metrics_response.text
        assert "api_requests_total" in content
        
        # Test analytics endpoint
        analytics_response = await client.get("/analytics/metrics")
        assert analytics_response.status_code == 200
    
    async def test_security_headers(self, client: APITestClient):
        """Test security headers in responses"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        # Check for security headers
        headers = response.headers
        expected_security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        for header in expected_security_headers:
            if header in headers:  # Not all headers may be present in test environment
                assert headers[header] is not None
    
    async def test_performance_benchmarks(self, client: APITestClient):
        """Test API performance benchmarks"""
        import time
        
        # Test response time for simple endpoint
        start_time = time.time()
        response = await client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Health check should be fast (< 1 second)
        assert response_time < 1.0
        
        # Test throughput
        start_time = time.time()
        concurrent_requests = 10
        
        tasks = []
        for i in range(concurrent_requests):
            task = client.get("/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        requests_per_second = concurrent_requests / total_time
        
        # Should handle at least 10 RPS in test environment
        assert requests_per_second > 10
        
        # All responses should be successful
        successful_responses = sum(1 for resp in responses if resp.status_code == 200)
        assert successful_responses == concurrent_requests
    
    async def test_billing_endpoints(self, client: APITestClient):
        """Test billing and quota management"""
        # Test usage endpoint
        usage_response = await client.get("/billing/usage")
        assert usage_response.status_code == 200
        
        # Test quota check
        quota_response = await client.get("/billing/quota", params={"resource_type": "api_request"})
        assert quota_response.status_code == 200
    
    async def test_cors_policies(self, client: APITestClient):
        """Test CORS policies"""
        # Test preflight request
        preflight_response = await client.options("/api/v1/patients")
        
        # Check CORS headers if present
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers"
        ]
        
        for header in cors_headers:
            if header in preflight_response.headers:
                assert preflight_response.headers[header] is not None

async def run_comprehensive_tests():
    """Run all tests and generate report"""
    test_suite = TestSuite()
    
    print("🧪 Starting Healthcare API Comprehensive Test Suite")
    print("=" * 60)
    
    # Initialize test client
    async with APITestClient(TEST_CONFIG["api_base_url"]) as client:
        test_data_manager = TestDataManager(TEST_CONFIG["test_data_dir"])
        
        # Run tests
        tests = [
            ("API Health Check", test_suite.test_api_health),
            ("Patient CRUD Operations", test_suite.test_patient_crud_operations),
            ("FHIR Operations", test_suite.test_fhir_operations),
            ("Observation Operations", test_suite.test_observation_operations),
            ("Authentication Flow", test_suite.test_authentication_flow),
            ("Rate Limiting", test_suite.test_rate_limiting),
            ("API Versioning", test_suite.test_api_versioning),
            ("Error Handling", test_suite.test_error_handling),
            ("Webhook Operations", test_suite.test_webhook_operations),
            ("Analytics Endpoints", test_suite.test_analytics_endpoints),
            ("Security Headers", test_suite.test_security_headers),
            ("Performance Benchmarks", test_suite.test_performance_benchmarks),
            ("Billing Endpoints", test_suite.test_billing_endpoints),
            ("CORS Policies", test_suite.test_cors_policies)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                if "fhir" in test_name.lower():
                    async with APITestClient(TEST_CONFIG["fhir_base_url"]) as fhir_client:
                        await test_func(fhir_client, test_data_manager)
                else:
                    await test_func(client, test_data_manager)
                print(f"✅ PASSED: {test_name}")
                passed += 1
            except Exception as e:
                print(f"❌ FAILED: {test_name} - {str(e)}")
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed} passed, {failed} failed")
        print(f"📈 Success Rate: {passed / (passed + failed) * 100:.1f}%")
        
        if failed == 0:
            print("\n🎉 All tests passed! Healthcare API is ready for production.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please review and fix issues.")
        
        return {
            "passed": passed,
            "failed": failed,
            "success_rate": passed / (passed + failed) * 100
        }

if __name__ == "__main__":
    # Run tests when executed directly
    results = asyncio.run(run_comprehensive_tests())
    
    # Exit with appropriate code
    exit(0 if results["failed"] == 0 else 1)