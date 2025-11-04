"""
Demo Authentication and Role-Based Access Control System
Provides secure demo authentication with role-based permissions
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import jwt
import sqlite3

class UserRole(Enum):
    """User roles in the demo system"""
    PATIENT = "patient"
    NURSE = "nurse" 
    ADMIN = "admin"

class Permission(Enum):
    """System permissions"""
    # Patient data permissions
    READ_PATIENTS = "read_patients"
    WRITE_PATIENTS = "write_patients"
    DELETE_PATIENTS = "delete_patients"
    
    # Vital signs permissions
    READ_VITALS = "read_vitals"
    WRITE_VITALS = "write_vitals"
    
    # Medication permissions
    READ_MEDICATIONS = "read_medications"
    WRITE_MEDICATIONS = "write_medications"
    
    # Lab results permissions
    READ_LAB_RESULTS = "read_lab_results"
    WRITE_LAB_RESULTS = "write_lab_results"
    
    # AI assessment permissions
    READ_ASSESSMENTS = "read_assessments"
    WRITE_ASSESSMENTS = "write_assessments"
    EXECUTE_AI = "execute_ai"
    
    # Admin permissions
    ADMIN_ACCESS = "admin_access"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    CONFIGURE_SYSTEM = "configure_system"
    
    # Demo-specific permissions
    DEMO_ADMIN = "demo_admin"
    SCENARIO_CONTROL = "scenario_control"

@dataclass
class User:
    """Demo user representation"""
    id: int
    email: str
    first_name: str
    last_name: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    last_login: Optional[datetime] = None
    session_token: Optional[str] = None
    session_expires: Optional[datetime] = None

@dataclass
class Session:
    """User session data"""
    user_id: int
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str

class RolePermissions:
    """Role-based permission definitions"""
    
    ROLE_PERMISSIONS = {
        UserRole.PATIENT: {
            Permission.READ_PATIENTS,  # Only own data
            Permission.READ_VITALS,     # Only own vitals
            Permission.READ_MEDICATIONS, # Only own medications
            Permission.READ_LAB_RESULTS, # Only own lab results
            Permission.READ_ASSESSMENTS, # Only own assessments
        },
        UserRole.NURSE: {
            Permission.READ_PATIENTS,
            Permission.WRITE_PATIENTS,
            Permission.READ_VITALS,
            Permission.WRITE_VITALS,
            Permission.READ_MEDICATIONS,
            Permission.WRITE_MEDICATIONS,
            Permission.READ_LAB_RESULTS,
            Permission.WRITE_LAB_RESULTS,
            Permission.READ_ASSESSMENTS,
            Permission.WRITE_ASSESSMENTS,
            Permission.EXECUTE_AI,
        },
        UserRole.ADMIN: {
            Permission.READ_PATIENTS,
            Permission.WRITE_PATIENTS,
            Permission.DELETE_PATIENTS,
            Permission.READ_VITALS,
            Permission.WRITE_VITALS,
            Permission.READ_MEDICATIONS,
            Permission.WRITE_MEDICATIONS,
            Permission.READ_LAB_RESULTS,
            Permission.WRITE_LAB_RESULTS,
            Permission.READ_ASSESSMENTS,
            Permission.WRITE_ASSESSMENTS,
            Permission.EXECUTE_AI,
            Permission.ADMIN_ACCESS,
            Permission.MANAGE_USERS,
            Permission.VIEW_ANALYTICS,
            Permission.CONFIGURE_SYSTEM,
            Permission.DEMO_ADMIN,
            Permission.SCENARIO_CONTROL,
        }
    }

class DemoAuthManager:
    """Demo authentication and authorization manager"""
    
    def __init__(self, db_path: str = "demo.db", secret_key: str = "demo-secret-key"):
        self.db_path = db_path
        self.secret_key = secret_key
        self.sessions: Dict[str, Session] = {}
        self.users_cache: Dict[int, User] = {}
        
    def hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        salt = secrets.token_bytes(32)
        iterations = 100000
        hash_value = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
        return f"{salt.hex()}:{hash_value.hex()}:{iterations}"
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex, iterations = hashed.split(':')
            salt_bytes = bytes.fromhex(salt)
            iterations = int(iterations)
            hash_bytes = bytes.fromhex(hash_hex)
            
            test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt_bytes, iterations)
            return test_hash == hash_bytes
        except:
            return False
            
    def create_user(self, email: str, password: str, first_name: str, 
                   last_name: str, role: UserRole) -> int:
        """Create a new demo user"""
        password_hash = self.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (email, password_hash, first_name, last_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, (email, password_hash, first_name, last_name, role.value))
        
        user_id = cursor.lastrowid
        
        # Create User object with permissions
        user = User(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            role=role,
            permissions=RolePermissions.ROLE_PERMISSIONS[role].copy()
        )
        
        self.users_cache[user_id] = user
        conn.commit()
        conn.close()
        
        return user_id
        
    def authenticate_user(self, email: str, password: str, 
                         ip_address: str = "127.0.0.1", 
                         user_agent: str = "demo-browser") -> Optional[User]:
        """Authenticate user and create session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, password_hash, first_name, last_name, role, is_active, last_login
            FROM users WHERE email = ? AND is_active = 1
        """, (email,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        user_id, password_hash, first_name, last_name, role, is_active, last_login = result
        
        if not is_active:
            return None
            
        if not self.verify_password(password, password_hash):
            return None
            
        # Create user object
        user = User(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            role=UserRole(role),
            permissions=RolePermissions.ROLE_PERMISSIONS[UserRole(role)].copy(),
            last_login=datetime.now()
        )
        
        # Create session
        session_token = self._generate_session_token()
        expires_at = datetime.now() + timedelta(hours=8)  # 8-hour demo sessions
        
        session = Session(
            user_id=user_id,
            token=session_token,
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_token] = session
        self.users_cache[user_id] = user
        
        # Update last login in database
        self._update_last_login(user_id)
        
        user.session_token = session_token
        user.session_expires = expires_at
        
        return user
        
    def get_user_by_token(self, token: str) -> Optional[User]:
        """Get user by session token"""
        session = self.sessions.get(token)
        
        if not session:
            return None
            
        if datetime.now() > session.expires_at:
            del self.sessions[token]
            return None
            
        user = self.users_cache.get(session.user_id)
        if user and user.session_token == token:
            return user
            
        return None
        
    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating session"""
        if token in self.sessions:
            del self.sessions[token]
            
        # Update user cache
        for user in self.users_cache.values():
            if user.session_token == token:
                user.session_token = None
                user.session_expires = None
                break
                
        return True
        
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions
        
    def check_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions"""
        return all(permission in user.permissions for permission in permissions)
        
    def filter_patient_data_by_access(self, user: User, patients: List[Dict]) -> List[Dict]:
        """Filter patient data based on user role and permissions"""
        if UserRole.ADMIN in user.role:
            return patients  # Admin sees all patients
            
        if UserRole.NURSE in user.role:
            return patients  # Nurse sees all patients (demo environment)
            
        if UserRole.PATIENT in user.role:
            # Patient only sees their own data
            return [p for p in patients if p.get('user_id') == user.id]
            
        return []
        
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract user from kwargs (assuming user is passed as 'user' parameter)
                user = kwargs.get('user')
                if not user or not self.check_permission(user, permission):
                    raise PermissionError(f"Permission {permission.value} required")
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def require_role(self, *roles: UserRole):
        """Decorator to require specific role(s)"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                user = kwargs.get('user')
                if not user or user.role not in roles:
                    raise PermissionError(f"One of roles {roles} required")
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
        
    def _update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.now(), user_id))
        conn.commit()
        conn.close()
        
    def get_demo_users(self) -> Dict[str, Dict]:
        """Get predefined demo users"""
        return {
            "admin": {
                "email": "admin@demo.medai.com",
                "password": "DemoAdmin123!",
                "role": UserRole.ADMIN,
                "name": "Demo Administrator",
                "permissions": list(RolePermissions.ROLE_PERMISSIONS[UserRole.ADMIN])
            },
            "nurse_jones": {
                "email": "nurse.jones@demo.medai.com",
                "password": "DemoNurse456!",
                "role": UserRole.NURSE,
                "name": "Sarah Jones, RN",
                "permissions": list(RolePermissions.ROLE_PERMISSIONS[UserRole.NURSE])
            },
            "patient_smith": {
                "email": "patient.smith@demo.medai.com",
                "password": "DemoPatient789!",
                "role": UserRole.PATIENT,
                "name": "John Smith",
                "permissions": list(RolePermissions.ROLE_PERMISSIONS[UserRole.PATIENT])
            }
        }
        
    def initialize_demo_users(self):
        """Initialize demo users in database"""
        demo_users = self.get_demo_users()
        
        for user_key, user_data in demo_users.items():
            try:
                self.create_user(
                    email=user_data["email"],
                    password=user_data["password"],
                    first_name=user_data["name"].split()[0],
                    last_name=user_data["name"].split()[-1],
                    role=user_data["role"]
                )
                print(f"✓ Created demo user: {user_data['name']}")
            except Exception as e:
                print(f"✗ Failed to create demo user {user_key}: {e}")
                
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_tokens = [
            token for token, session in self.sessions.items()
            if now > session.expires_at
        ]
        
        for token in expired_tokens:
            del self.sessions[token]
            
        return len(expired_tokens)

# Demo-specific auth utilities

class DemoSecurityManager:
    """Security manager for demo environment"""
    
    def __init__(self, auth_manager: DemoAuthManager):
        self.auth_manager = auth_manager
        
    def log_security_event(self, event_type: str, user_id: int, details: str):
        """Log security events for demo monitoring"""
        # In a real system, this would write to audit logs
        print(f"SECURITY EVENT: {event_type} - User: {user_id} - Details: {details}")
        
    def validate_demo_access(self, user: User, resource: str) -> bool:
        """Validate demo environment access"""
        # Demo-specific validation rules
        if user.role == UserRole.PATIENT:
            return resource in ["own_data", "assessments", "medications"]
        elif user.role == UserRole.NURSE:
            return resource in ["patients", "vitals", "assessments", "medications"]
        elif user.role == UserRole.ADMIN:
            return True  # Admin has full access in demo
            
        return False
        
    def sanitize_demo_data(self, data: Dict, user_role: UserRole) -> Dict:
        """Sanitize data based on user role"""
        # Remove sensitive fields based on role
        sensitive_fields = [
            "real_ssn", "real_phone", "real_address", 
            "insurance_details", "payment_info"
        ]
        
        if user_role == UserRole.PATIENT:
            # Patients only see their own sanitized data
            pass  # Already filtered by access control
            
        elif user_role == UserRole.NURSE:
            # Nurses see clinical data but not administrative details
            for field in sensitive_fields:
                if field in data:
                    del data[field]
                    
        elif user_role == UserRole.ADMIN:
            # Admin sees all data but with demo warnings
            data["_demo_note"] = "This is synthetic demo data"
            
        return data

if __name__ == "__main__":
    # Test authentication system
    auth_manager = DemoAuthManager()
    
    print("Demo Authentication System")
    print("=" * 40)
    
    # Create demo users
    auth_manager.initialize_demo_users()
    
    # Test authentication
    demo_users = auth_manager.get_demo_users()
    
    for user_key, user_data in demo_users.items():
        print(f"\nTesting {user_data['name']}:")
        
        user = auth_manager.authenticate_user(
            user_data["email"], 
            user_data["password"]
        )
        
        if user:
            print(f"✓ Authentication successful")
            print(f"  Role: {user.role.value}")
            print(f"  Permissions: {len(user.permissions)}")
            print(f"  Session token: {user.session_token[:16]}...")
        else:
            print("✗ Authentication failed")