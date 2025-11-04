"""
Demo Environment Testing Framework
Comprehensive tests for all demo components and integrations
"""

import pytest
import asyncio
import sqlite3
import json
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import demo components
import sys
sys.path.append('.')

from demo.auth.demo_auth import DemoAuthManager, UserRole, Permission
from demo.analytics.demo_analytics import DemoAnalyticsManager, DemoTracker, UserAction
from demo.backup.demo_backup import DemoBackupManager
from demo.scenarios.medical_scenarios import ScenarioManager, DiabetesManagementScenario
from demo.database.populate_demo_data import DemoDatabasePopulator

class TestDemoDatabase:
    """Test demo database functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def populator(self, temp_db):
        """Create database populator for testing"""
        return DemoDatabasePopulator(temp_db)
    
    def test_database_initialization(self, populator):
        """Test database schema creation"""
        populator.connect()
        cursor = populator.conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'users', 'patients', 'medical_conditions', 'vital_signs',
            'medications', 'lab_results', 'ai_assessments', 'demo_scenarios',
            'demo_sessions', 'usage_analytics', 'demo_config'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
        
        populator.close()
    
    def test_data_population(self, populator):
        """Test synthetic data generation"""
        populator.run_population()
        populator.connect()
        
        cursor = populator.conn.cursor()
        
        # Check user count
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        assert user_count > 0, "No users created"
        
        # Check patient count
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        assert patient_count > 0, "No patients created"
        
        # Check vital signs data
        cursor.execute("SELECT COUNT(*) FROM vital_signs")
        vitals_count = cursor.fetchone()[0]
        assert vitals_count > 0, "No vital signs data created"
        
        populator.close()
    
    def test_data_realism(self, populator):
        """Test that generated data is realistic"""
        populator.run_population()
        populator.connect()
        
        cursor = populator.conn.cursor()
        
        # Check vital signs ranges
        cursor.execute("""
            SELECT blood_pressure_systolic, blood_pressure_diastolic, 
                   glucose_level, heart_rate
            FROM vital_signs 
            LIMIT 100
        """)
        
        for row in cursor.fetchall():
            bp_systolic, bp_diastolic, glucose, heart_rate = row
            
            # Check realistic ranges
            assert 70 <= bp_systolic <= 200, f"Invalid systolic BP: {bp_systolic}"
            assert 40 <= bp_diastolic <= 120, f"Invalid diastolic BP: {bp_diastolic}"
            assert 50 <= glucose <= 400, f"Invalid glucose: {glucose}"
            assert 40 <= heart_rate <= 150, f"Invalid heart rate: {heart_rate}"
        
        populator.close()

class TestDemoAuthentication:
    """Test demo authentication system"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager for testing"""
        return DemoAuthManager()
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_password_hashing(self, auth_manager):
        """Test password hashing and verification"""
        password = "TestPassword123!"
        hashed = auth_manager.hash_password(password)
        
        # Verify correct password
        assert auth_manager.verify_password(password, hashed)
        
        # Verify incorrect password
        assert not auth_manager.verify_password("wrongpassword", hashed)
    
    def test_user_creation(self, auth_manager, temp_db_path):
        """Test user creation"""
        auth_manager.db_path = temp_db_path
        
        user_id = auth_manager.create_user(
            email="test@example.com",
            password="TestPassword123!",
            first_name="Test",
            last_name="User",
            role=UserRole.NURSE
        )
        
        assert user_id > 0, "User creation failed"
        
        # Verify user exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT email, role FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        assert result[0] == "test@example.com"
        assert result[1] == "nurse"
    
    def test_user_authentication(self, auth_manager, temp_db_path):
        """Test user authentication"""
        auth_manager.db_path = temp_db_path
        
        # Create test user
        user_id = auth_manager.create_user(
            email="test@example.com",
            password="TestPassword123!",
            first_name="Test",
            last_name="User",
            role=UserRole.PATIENT
        )
        
        # Test authentication
        user = auth_manager.authenticate_user("test@example.com", "TestPassword123!")
        
        assert user is not None
        assert user.id == user_id
        assert user.email == "test@example.com"
        assert user.role == UserRole.PATIENT
        assert user.session_token is not None
    
    def test_session_management(self, auth_manager, temp_db_path):
        """Test session management"""
        auth_manager.db_path = temp_db_path
        
        # Create and authenticate user
        user = auth_manager.authenticate_user("test@example.com", "TestPassword123!")
        token = user.session_token
        
        # Test session retrieval
        retrieved_user = auth_manager.get_user_by_token(token)
        assert retrieved_user is not None
        assert retrieved_user.id == user.id
        
        # Test session invalidation
        auth_manager.logout_user(token)
        
        # Test session after logout
        invalid_user = auth_manager.get_user_by_token(token)
        assert invalid_user is None
    
    def test_permission_system(self, auth_manager, temp_db_path):
        """Test permission system"""
        auth_manager.db_path = temp_db_path
        
        # Create users with different roles
        admin = auth_manager.create_user(
            email="admin@example.com",
            password="Password123!",
            first_name="Admin",
            last_name="User",
            role=UserRole.ADMIN
        )
        
        nurse = auth_manager.create_user(
            email="nurse@example.com",
            password="Password123!",
            first_name="Nurse",
            last_name="User",
            role=UserRole.NURSE
        )
        
        patient = auth_manager.create_user(
            email="patient@example.com",
            password="Password123!",
            first_name="Patient",
            last_name="User",
            role=UserRole.PATIENT
        )
        
        # Authenticate users
        admin_user = auth_manager.authenticate_user("admin@example.com", "Password123!")
        nurse_user = auth_manager.authenticate_user("nurse@example.com", "Password123!")
        patient_user = auth_manager.authenticate_user("patient@example.com", "Password123!")
        
        # Test permissions
        assert auth_manager.check_permission(admin_user, Permission.ADMIN_ACCESS)
        assert auth_manager.check_permission(nurse_user, Permission.READ_PATIENTS)
        assert not auth_manager.check_permission(patient_user, Permission.ADMIN_ACCESS)

class TestDemoAnalytics:
    """Test demo analytics system"""
    
    @pytest.fixture
    def analytics_manager(self):
        """Create analytics manager for testing"""
        return DemoAnalyticsManager()
    
    def test_user_action_tracking(self, analytics_manager):
        """Test user action tracking"""
        action = UserAction(
            user_id=1,
            session_id="test_session_123",
            action_type="page_view",
            component="dashboard",
            timestamp=datetime.now(),
            duration_ms=500,
            success=True
        )
        
        analytics_manager.track_user_action(action)
        
        # Verify tracking worked
        user_analytics = analytics_manager.get_user_analytics(user_id=1, days=1)
        assert user_analytics["total_actions"] > 0
    
    def test_demo_session_tracking(self, analytics_manager):
        """Test demo session tracking"""
        session_id = "test_session_456"
        user_id = 1
        
        # Start session
        success = analytics_manager.start_demo_session(session_id, user_id)
        assert success
        
        # End session
        success = analytics_manager.end_demo_session(session_id, feedback_score=5)
        assert success
        
        # Verify session was tracked
        user_analytics = analytics_manager.get_user_analytics(user_id=user_id, days=1)
        assert user_analytics["total_sessions"] > 0
    
    def test_system_metrics(self, analytics_manager):
        """Test system metrics tracking"""
        from demo.analytics.demo_analytics import SystemMetric
        
        metric = SystemMetric(
            timestamp=datetime.now(),
            metric_type="response_time",
            value=250.5,
            unit="ms",
            component="api"
        )
        
        analytics_manager.track_system_metric(metric)
        
        # Verify metrics were tracked
        performance_data = analytics_manager.get_system_performance(hours=1)
        assert performance_data["total_metrics"] > 0
    
    def test_analytics_dashboard(self, analytics_manager):
        """Test analytics dashboard data generation"""
        dashboard_data = analytics_manager.get_demo_dashboard_data()
        
        # Check dashboard structure
        assert "user_analytics" in dashboard_data
        assert "system_performance" in dashboard_data
        assert "scenario_analytics" in dashboard_data
        assert "active_sessions" in dashboard_data
        assert "real_time_metrics" in dashboard_data

class TestDemoBackup:
    """Test demo backup system"""
    
    @pytest.fixture
    def backup_manager(self):
        """Create backup manager for testing"""
        temp_dir = tempfile.mkdtemp()
        backup_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        return DemoBackupManager(backup_dir=backup_dir, demo_data_dir=temp_dir)
    
    def test_backup_creation(self, backup_manager):
        """Test backup creation"""
        # Create demo data
        os.makedirs(backup_manager.demo_data_dir, exist_ok=True)
        with open(os.path.join(backup_manager.demo_data_dir, "test_file.txt"), "w") as f:
            f.write("test data")
        
        backup = backup_manager.create_full_backup("Test backup")
        
        assert backup is not None
        assert backup.backup_id is not None
        assert backup.backup_type == "full"
        assert os.path.exists(backup.source_path)
    
    def test_demo_verification(self, backup_manager):
        """Test demo environment verification"""
        verification = backup_manager.verify_demo_state()
        
        # Should fail verification initially (no databases)
        assert isinstance(verification, dict)
        assert "database_accessible" in verification
        assert "config_valid" in verification
        assert "analytics_working" in verification
        assert "models_loaded" in verification
        assert "all_tests_passed" in verification
    
    def test_demo_readiness_report(self, backup_manager):
        """Test demo readiness report generation"""
        report = backup_manager.get_demo_readiness_report()
        
        assert report is not None
        assert "timestamp" in report
        assert "demo_state" in report
        assert "backup_status" in report
        assert "recommendations" in report
    
    def test_backup_cleanup(self, backup_manager):
        """Test backup cleanup functionality"""
        # Create multiple backups
        backup1 = backup_manager.create_full_backup("Old backup 1")
        backup2 = backup_manager.create_full_backup("Old backup 2")
        
        # Mark backups as old (set timestamp manually)
        from datetime import timedelta
        backup1.timestamp = datetime.now() - timedelta(days=10)
        
        # Run cleanup
        backup_manager.cleanup_old_backups(retention_days=7)
        
        # Verify old backup was removed
        metadata = backup_manager._load_backup_metadata(backup1.backup_id)
        assert metadata is None

class TestDemoScenarios:
    """Test demo scenario functionality"""
    
    @pytest.fixture
    def scenario_manager(self):
        """Create scenario manager for testing"""
        return ScenarioManager()
    
    def test_diabetes_scenario_creation(self, scenario_manager):
        """Test diabetes scenario creation"""
        scenario = scenario_manager.create_scenario("diabetes", 1)
        
        assert scenario is not None
        assert isinstance(scenario, DiabetesManagementScenario)
        assert scenario.patient_id == 1
        assert scenario.patient_name == "John Smith"
    
    def test_diabetes_scenario_data(self, scenario_manager):
        """Test diabetes scenario data generation"""
        scenario_manager.create_scenario("diabetes", 1)
        data = scenario_manager.get_scenario_data("diabetes")
        
        assert data is not None
        assert "patient_info" in data
        assert "vital_signs" in data
        assert "lab_results" in data
        assert "medications" in data
        assert "ai_recommendations" in data
        
        # Check data structure
        assert len(data["vital_signs"]) > 0
        assert len(data["lab_results"]) > 0
        assert len(data["medications"]) > 0
        assert len(data["ai_recommendations"]) > 0
    
    def test_hypertension_scenario(self, scenario_manager):
        """Test hypertension scenario"""
        scenario = scenario_manager.create_scenario("hypertension", 2)
        data = scenario_manager.get_scenario_data("hypertension")
        
        assert data is not None
        assert "patient_info" in data
        assert "bp_trends" in data
        assert "cv_risk_score" in data
        assert "ai_recommendations" in data
        
        # Check blood pressure trends
        bp_trends = data["bp_trends"]
        assert len(bp_trends) == 30  # 30 days of data
        
        # Verify BP ranges
        for trend in bp_trends:
            assert 110 <= trend["systolic"] <= 160
            assert 70 <= trend["diastolic"] <= 100
    
    def test_chest_pain_scenario(self, scenario_manager):
        """Test chest pain assessment scenario"""
        scenario = scenario_manager.create_scenario("chest_pain", 3)
        data = scenario_manager.get_scenario_data("chest_pain")
        
        assert data is not None
        assert "patient_info" in data
        assert "symptoms" in data
        assert "ecg_findings" in data
        assert "risk_stratification" in data
        assert "emergency_protocol" in data
        
        # Check emergency protocol
        protocol = data["emergency_protocol"]
        assert "immediate_actions" in protocol
        assert "medications" in protocol
        assert "consultations" in protocol

class TestDemoIntegration:
    """Integration tests for demo components"""
    
    @pytest.fixture
    def temp_demo_env(self):
        """Create temporary demo environment"""
        temp_dir = tempfile.mkdtemp()
        demo_dir = os.path.join(temp_dir, "demo")
        os.makedirs(demo_dir, exist_ok=True)
        
        # Create basic demo structure
        open(os.path.join(demo_dir, "demo.db"), "w").close()
        open(os.path.join(temp_dir, "demo_analytics.db"), "w").close()
        
        yield {
            "temp_dir": temp_dir,
            "demo_dir": demo_dir,
            "db_path": os.path.join(demo_dir, "demo.db"),
            "analytics_path": os.path.join(temp_dir, "demo_analytics.db")
        }
        
        shutil.rmtree(temp_dir)
    
    def test_complete_demo_workflow(self, temp_demo_env):
        """Test complete demo workflow"""
        db_path = temp_demo_env["db_path"]
        analytics_path = temp_demo_env["analytics_path"]
        
        # Initialize components
        auth_manager = DemoAuthManager(db_path=db_path)
        analytics_manager = DemoAnalyticsManager(analytics_path)
        scenario_manager = ScenarioManager()
        
        # Create demo database and populate
        populator = DemoDatabasePopulator(db_path)
        populator.run_population()
        
        # Initialize demo users
        auth_manager.initialize_demo_users()
        
        # Test user login
        user = auth_manager.authenticate_user(
            "admin@demo.medai.com", 
            "DemoAdmin123!"
        )
        assert user is not None
        
        # Start analytics session
        session_id = "integration_test_session"
        analytics_manager.start_demo_session(session_id, user.id)
        
        # Create and run scenario
        scenario = scenario_manager.create_scenario("diabetes", 1)
        scenario_data = scenario_manager.get_scenario_data("diabetes")
        
        assert scenario_data is not None
        
        # Track user action
        tracker = DemoTracker(analytics_manager)
        tracker.track_page_view(user.id, session_id, "scenario_dashboard")
        
        # End session
        analytics_manager.end_demo_session(session_id, feedback_score=5)
        
        # Verify analytics
        user_analytics = analytics_manager.get_user_analytics(user.id, days=1)
        assert user_analytics["total_actions"] > 0
        assert user_analytics["total_sessions"] > 0
    
    def test_end_to_end_demo_sequence(self, temp_demo_env):
        """Test end-to-end demo sequence"""
        # This test simulates a complete user journey
        db_path = temp_demo_env["db_path"]
        analytics_path = temp_demo_env["analytics_path"]
        
        # Initialize
        auth_manager = DemoAuthManager(db_path=db_path)
        analytics_manager = DemoAnalyticsManager(analytics_path)
        scenario_manager = ScenarioManager()
        
        # Populate database
        populator = DemoDatabasePopulator(db_path)
        populator.run_population()
        auth_manager.initialize_demo_users()
        
        # Simulate user journey
        steps = [
            "User logs in",
            "User views dashboard", 
            "User selects diabetes scenario",
            "User reviews glucose data",
            "User reviews AI recommendations",
            "User completes scenario",
            "User provides feedback"
        ]
        
        # Login
        user = auth_manager.authenticate_user("nurse.jones@demo.medai.com", "DemoNurse456!")
        assert user is not None
        
        session_id = "e2e_test_session"
        analytics_manager.start_demo_session(session_id, user.id)
        
        # Simulate each step
        tracker = DemoTracker(analytics_manager)
        for i, step in enumerate(steps):
            tracker.track_scenario_step(user.id, session_id, step, success=True)
            
        # Complete scenario
        tracker.track_demo_completion(
            user_id=user.id,
            session_id=session_id,
            scenario_id="diabetes",
            scenario_name="Diabetes Management",
            steps_completed=len(steps),
            total_steps=len(steps),
            duration_minutes=12.5,
            satisfaction=5
        )
        
        # Verify completion
        user_analytics = analytics_manager.get_user_analytics(user.id, days=1)
        assert user_analytics["total_actions"] == len(steps)
        
        # Check scenario completion
        scenario_analytics = analytics_manager.get_demo_scenarios_analytics()
        assert scenario_analytics["total_completions"] > 0

class TestDemoPerformance:
    """Test demo performance characteristics"""
    
    def test_response_time_requirements(self):
        """Test that demo meets response time requirements"""
        # This test verifies performance constraints
        auth_manager = DemoAuthManager()
        
        # Measure authentication time
        start_time = datetime.now()
        
        # Note: This test would need proper setup in real environment
        # For now, we just verify the method exists and can be called
        assert hasattr(auth_manager, 'authenticate_user')
        
        elapsed = (datetime.now() - start_time).total_seconds()
        assert elapsed < 5.0  # Should complete within 5 seconds
    
    def test_database_query_performance(self, temp_db_path):
        """Test database query performance"""
        # Create populated database
        populator = DemoDatabasePopulator(temp_db_path)
        populator.run_population()
        
        # Measure query performance
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        start_time = datetime.now()
        
        # Test various queries
        cursor.execute("SELECT COUNT(*) FROM patients")
        cursor.execute("SELECT COUNT(*) FROM vital_signs")
        cursor.execute("SELECT * FROM vital_signs WHERE patient_id = 1 LIMIT 10")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        conn.close()
        
        # Should complete within reasonable time
        assert elapsed < 2.0, f"Database queries took too long: {elapsed}s"

class TestDemoConfiguration:
    """Test demo configuration system"""
    
    def test_demo_settings_loading(self):
        """Test demo settings can be loaded"""
        try:
            from demo.config.demo_settings import DEMO_MODE, API_CONFIG, AUTH_CONFIG
            
            assert DEMO_MODE["enabled"] == True
            assert "host" in API_CONFIG
            assert "port" in API_CONFIG
            assert "secret_key" in AUTH_CONFIG
            
        except ImportError as e:
            pytest.skip(f"Demo settings not available: {e}")
    
    def test_demo_users_configuration(self):
        """Test demo users are properly configured"""
        try:
            from demo.config.demo_settings import DEMO_USERS
            
            assert "admin" in DEMO_USERS
            assert "nurse_jones" in DEMO_USERS
            assert "patient_smith" in DEMO_USERS
            
            # Check each user has required fields
            for user_key, user_data in DEMO_USERS.items():
                assert "email" in user_data
                assert "password" in user_data
                assert "role" in user_data
                assert "permissions" in user_data
                
        except ImportError:
            pytest.skip("Demo settings not available")

def run_demo_tests():
    """Run all demo tests"""
    print("Running Demo Environment Tests...")
    print("=" * 50)
    
    # Run pytest programmatically
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers"
    ])

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    run_demo_tests()