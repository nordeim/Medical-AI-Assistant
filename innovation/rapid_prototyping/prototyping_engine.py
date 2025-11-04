"""
Rapid Prototyping and Development Methodologies System
Agile, DevOps, CI/CD pipelines for fast feature development and deployment
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess
import os
import yaml
import tempfile

class DevelopmentPhase(Enum):
    PLANNING = "planning"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

class DevOpsStage(Enum):
    SOURCE_CONTROL = "source_control"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    ROLLBACK = "rollback"

class PrototypeStatus(Enum):
    CONCEPT = "concept"
    WIREFRAME = "wireframe"
    FUNCTIONAL = "functional"
    TESTED = "tested"
    DEPLOYED = "deployed"
    VALIDATED = "validated"

class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"

@dataclass
class DevelopmentTask:
    """Individual development task"""
    task_id: str
    title: str
    description: str
    phase: DevelopmentPhase
    assigned_to: str
    estimated_hours: float
    actual_hours: float = 0.0
    status: str = "pending"
    dependencies: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Prototype:
    """Prototype artifact"""
    prototype_id: str
    feature_id: str
    name: str
    description: str
    prototype_type: str  # "wireframe", "functional", "api", "ml_model"
    status: PrototypeStatus
    created_at: datetime
    technologies: List[str]
    files: List[str]
    test_results: Dict[str, Any]
    deployment_info: Dict[str, Any]
    
@dataclass
class DevOpsPipeline:
    """DevOps pipeline configuration"""
    pipeline_id: str
    name: str
    stages: Dict[DevOpsStage, Dict[str, Any]]
    triggers: List[str]
    environment: str
    last_run: datetime
    status: str
    metrics: Dict[str, Any]

class RapidPrototypingEngine:
    """Rapid prototyping and development engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('RapidPrototypingEngine')
        
        # Development tools and configurations
        self.agile_manager = AgileProjectManager(config.get('agile', {}))
        self.devops_engine = DevOpsEngine(config.get('devops', {}))
        self.ci_cd_manager = CICDManager(config.get('cicd', {}))
        self.testing_framework = TestingFramework(config.get('testing', {}))
        
        # Prototyping tools
        self.prototype_creator = PrototypeCreator()
        self.deployment_manager = DeploymentManager()
        
        # Storage
        self.active_prototypes: Dict[str, Prototype] = {}
        self.development_tasks: Dict[str, DevelopmentTask] = {}
        self.pipelines: Dict[str, DevOpsPipeline] = {}
        
        # Metrics
        self.prototype_metrics = []
        self.deployment_metrics = []
        
    async def initialize(self):
        """Initialize rapid prototyping engine"""
        self.logger.info("Initializing Rapid Prototyping Engine...")
        
        # Initialize all subsystems
        await self.agile_manager.initialize()
        await self.devops_engine.initialize()
        await self.ci_cd_manager.initialize()
        await self.testing_framework.initialize()
        
        # Setup default pipelines
        await self._setup_default_pipelines()
        
        # Start background monitoring
        asyncio.create_task(self._monitor_prototypes())
        
        return {"status": "prototyping_engine_initialized"}
    
    async def create_prototypes(self, features: List[Dict[str, Any]]) -> List[Prototype]:
        """Create rapid prototypes for features"""
        prototypes = []
        
        try:
            for feature in features[:5]:  # Limit to top 5 features
                # Create prototype
                prototype = await self._create_feature_prototype(feature)
                
                # Setup development pipeline
                await self._setup_development_pipeline(prototype)
                
                # Create initial tasks
                tasks = await self.agile_manager.create_sprint_tasks(prototype)
                
                # Setup CI/CD pipeline
                pipeline = await self.ci_cd_manager.create_pipeline(prototype)
                
                prototypes.append(prototype)
                self.active_prototypes[prototype.prototype_id] = prototype
                
                self.logger.info(f"Created prototype: {prototype.name}")
            
            return prototypes
            
        except Exception as e:
            self.logger.error(f"Prototype creation failed: {str(e)}")
            raise
    
    async def _create_feature_prototype(self, feature: Dict[str, Any]) -> Prototype:
        """Create prototype for a specific feature"""
        prototype_id = str(uuid.uuid4())
        
        # Determine prototype type based on feature
        if 'ai' in feature.get('category', '').lower() or 'ml' in feature.get('title', '').lower():
            prototype_type = 'ml_model'
        elif 'api' in feature.get('title', '').lower() or 'integration' in feature.get('category', '').lower():
            prototype_type = 'api'
        elif 'dashboard' in feature.get('title', '').lower() or 'analytics' in feature.get('category', '').lower():
            prototype_type = 'functional'
        else:
            prototype_type = 'wireframe'
        
        # Create prototype files
        files = await self.prototype_creator.create_prototype_files(
            prototype_id, feature, prototype_type
        )
        
        # Create test cases
        test_cases = await self.testing_framework.generate_test_cases(feature, prototype_type)
        
        prototype = Prototype(
            prototype_id=prototype_id,
            feature_id=feature.get('feature_id', ''),
            name=f"Prototype: {feature.get('title', 'Unknown Feature')}",
            description=feature.get('description', ''),
            prototype_type=prototype_type,
            status=PrototypeStatus.CONCEPT,
            created_at=datetime.now(),
            technologies=self._determine_technologies(prototype_type),
            files=files,
            test_results={'generated': len(test_cases)},
            deployment_info={}
        )
        
        return prototype
    
    async def _setup_development_pipeline(self, prototype: Prototype):
        """Setup development pipeline for prototype"""
        pipeline_config = {
            'prototype_id': prototype.prototype_id,
            'source_repo': f"prototypes/{prototype.prototype_id}",
            'build_commands': self._get_build_commands(prototype),
            'test_commands': self._get_test_commands(prototype),
            'deploy_commands': self._get_deploy_commands(prototype)
        }
        
        # Initialize repository and basic structure
        await self.devops_engine.initialize_repository(pipeline_config)
        
        # Setup automated testing
        await self.testing_framework.setup_test_suite(prototype)
        
        # Create deployment configuration
        await self.deployment_manager.create_deployment_config(prototype)
    
    async def _setup_default_pipelines(self):
        """Setup default DevOps pipelines"""
        # Python ML Pipeline
        ml_pipeline = DevOpsPipeline(
            pipeline_id=str(uuid.uuid4()),
            name="ML Feature Pipeline",
            stages={
                DevOpsStage.SOURCE_CONTROL: {
                    'trigger': 'push',
                    'repository': 'feature-repository',
                    'branch': 'main'
                },
                DevOpsStage.BUILD: {
                    'steps': ['pip install -r requirements.txt', 'python setup.py build'],
                    'artifacts': ['dist/', '*.whl']
                },
                DevOpsStage.TEST: {
                    'steps': ['pytest tests/', 'coverage run -m pytest'],
                    'coverage_threshold': 80
                },
                DevOpsStage.SECURITY_SCAN: {
                    'tools': ['bandit', 'safety'],
                    'fail_on_issues': True
                },
                DevOpsStage.DEPLOY: {
                    'target': 'docker',
                    'registry': 'internal-registry'
                }
            },
            triggers=['push', 'pull_request'],
            environment='development',
            last_run=datetime.now(),
            status='active',
            metrics={'success_rate': 95.5}
        )
        
        # API Development Pipeline
        api_pipeline = DevOpsPipeline(
            pipeline_id=str(uuid.uuid4()),
            name="API Development Pipeline",
            stages={
                DevOpsStage.SOURCE_CONTROL: {
                    'trigger': 'commit',
                    'repository': 'api-repository'
                },
                DevOpsStage.BUILD: {
                    'steps': ['npm install', 'npm run build'],
                    'artifacts': ['build/']
                },
                DevOpsStage.TEST: {
                    'steps': ['npm test', 'jest --coverage'],
                    'test_types': ['unit', 'integration']
                },
                DevOpsStage.DEPLOY: {
                    'target': 'kubernetes',
                    'namespace': 'development'
                }
            },
            triggers=['push'],
            environment='development',
            last_run=datetime.now(),
            status='active',
            metrics={'deployment_frequency': 12.3}
        )
        
        self.pipelines[ml_pipeline.pipeline_id] = ml_pipeline
        self.pipelines[api_pipeline.pipeline_id] = api_pipeline
    
    def _determine_technologies(self, prototype_type: str) -> List[str]:
        """Determine technologies based on prototype type"""
        tech_map = {
            'ml_model': ['python', 'tensorflow', 'scikit-learn', 'pandas', 'numpy'],
            'api': ['nodejs', 'express', 'mongodb', 'docker'],
            'functional': ['react', 'typescript', 'css', 'html'],
            'wireframe': ['figma', 'sketch', 'html', 'css']
        }
        return tech_map.get(prototype_type, ['python', 'javascript'])
    
    def _get_build_commands(self, prototype: Prototype) -> List[str]:
        """Get build commands for prototype"""
        if 'python' in prototype.technologies:
            return ['pip install -r requirements.txt', 'python -m pytest']
        elif 'nodejs' in prototype.technologies:
            return ['npm install', 'npm run build']
        else:
            return ['make build']
    
    def _get_test_commands(self, prototype: Prototype) -> List[str]:
        """Get test commands for prototype"""
        if 'python' in prototype.technologies:
            return ['pytest tests/ -v', 'coverage run -m pytest']
        elif 'nodejs' in prototype.technologies:
            return ['npm test', 'jest --coverage']
        else:
            return ['make test']
    
    def _get_deploy_commands(self, prototype: Prototype) -> List[str]:
        """Get deploy commands for prototype"""
        return [
            'docker build -t prototype:latest .',
            'docker push prototype:latest',
            'kubectl apply -f k8s/'
        ]
    
    async def get_prototype_status(self, prototype_id: str) -> Dict[str, Any]:
        """Get detailed prototype status"""
        if prototype_id not in self.active_prototypes:
            raise ValueError(f"Prototype {prototype_id} not found")
        
        prototype = self.active_prototypes[prototype_id]
        
        # Get pipeline status
        pipeline_status = await self.devops_engine.get_pipeline_status(prototype_id)
        
        # Get test results
        test_results = await self.testing_framework.get_test_results(prototype_id)
        
        # Get deployment status
        deployment_status = await self.deployment_manager.get_deployment_status(prototype_id)
        
        return {
            'prototype': asdict(prototype),
            'pipeline_status': pipeline_status,
            'test_results': test_results,
            'deployment_status': deployment_status,
            'progress_percentage': self._calculate_progress(prototype),
            'estimated_completion': self._estimate_completion(prototype)
        }
    
    def _calculate_progress(self, prototype: Prototype) -> float:
        """Calculate prototype completion percentage"""
        status_weights = {
            PrototypeStatus.CONCEPT: 10,
            PrototypeStatus.WIREFRAME: 30,
            PrototypeStatus.FUNCTIONAL: 60,
            PrototypeStatus.TESTED: 80,
            PrototypeStatus.DEPLOYED: 95,
            PrototypeStatus.VALIDATED: 100
        }
        
        return status_weights.get(prototype.status, 0)
    
    def _estimate_completion(self, prototype: Prototype) -> datetime:
        """Estimate prototype completion date"""
        progress = self._calculate_progress(prototype)
        
        if progress >= 100:
            return datetime.now()
        
        days_remaining = max(1, (100 - progress) / 10)  # Estimate 10% completion per day
        return datetime.now() + timedelta(days=days_remaining)
    
    async def deploy_prototype(self, prototype_id: str, environment: str = 'development') -> Dict[str, Any]:
        """Deploy prototype to specified environment"""
        if prototype_id not in self.active_prototypes:
            raise ValueError(f"Prototype {prototype_id} not found")
        
        prototype = self.active_prototypes[prototype_id]
        
        try:
            # Update status
            prototype.status = PrototypeStatus.DEPLOYED
            prototype.deployment_info = {
                'environment': environment,
                'deployed_at': datetime.now().isoformat(),
                'deployment_id': str(uuid.uuid4())
            }
            
            # Execute deployment
            deployment_result = await self.deployment_manager.deploy(
                prototype, environment
            )
            
            # Update pipeline
            await self.devops_engine.trigger_deployment_pipeline(
                prototype_id, environment
            )
            
            self.logger.info(f"Deployed prototype {prototype_id} to {environment}")
            
            return {
                'status': 'success',
                'prototype_id': prototype_id,
                'environment': environment,
                'deployment_info': deployment_result,
                'deployed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            raise
    
    async def get_prototyping_metrics(self) -> Dict[str, Any]:
        """Get prototyping and development metrics"""
        total_prototypes = len(self.active_prototypes)
        completed_prototypes = len([p for p in self.active_prototypes.values() 
                                  if p.status == PrototypeStatus.VALIDATED])
        
        status_distribution = {}
        for status in PrototypeStatus:
            count = len([p for p in self.active_prototypes.values() if p.status == status])
            status_distribution[status.value] = count
        
        # Calculate average development time
        development_times = []
        for prototype in self.active_prototypes.values():
            if prototype.status == PrototypeStatus.VALIDATED:
                duration = (datetime.now() - prototype.created_at).days
                development_times.append(duration)
        
        avg_dev_time = sum(development_times) / len(development_times) if development_times else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_prototypes': total_prototypes,
            'completed_prototypes': completed_prototypes,
            'completion_rate': (completed_prototypes / total_prototypes * 100) if total_prototypes > 0 else 0,
            'status_distribution': status_distribution,
            'average_development_time_days': round(avg_dev_time, 2),
            'active_pipelines': len([p for p in self.pipelines.values() if p.status == 'active']),
            'deployment_frequency': self._calculate_deployment_frequency(),
            'quality_metrics': await self._get_quality_metrics()
        }
    
    def _calculate_deployment_frequency(self) -> float:
        """Calculate deployment frequency per week"""
        if not self.deployment_metrics:
            return 0.0
        
        recent_deployments = [m for m in self.deployment_metrics 
                            if (datetime.now() - m['timestamp']).days <= 7]
        
        return len(recent_deployments) / 7.0  # Deployments per day
    
    async def _get_quality_metrics(self) -> Dict[str, float]:
        """Get development quality metrics"""
        # Simulate quality metrics
        return {
            'test_coverage': 87.5,
            'code_quality_score': 92.3,
            'security_score': 88.7,
            'performance_score': 91.2,
            'maintainability_index': 85.6
        }

# Supporting classes
class AgileProjectManager:
    """Agile project management for rapid development"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('AgileProjectManager')
        self.sprints = []
        self.user_stories = []
    
    async def initialize(self):
        """Initialize agile project manager"""
        self.logger.info("Initializing Agile Project Manager...")
        return {"status": "agile_manager_initialized"}
    
    async def create_sprint_tasks(self, prototype: Prototype) -> List[DevelopmentTask]:
        """Create sprint tasks for prototype"""
        tasks = []
        
        # Generate tasks based on prototype type
        if prototype.prototype_type == 'ml_model':
            tasks = self._create_ml_tasks(prototype)
        elif prototype.prototype_type == 'api':
            tasks = self._create_api_tasks(prototype)
        elif prototype.prototype_type == 'functional':
            tasks = self._create_functional_tasks(prototype)
        else:
            tasks = self._create_wireframe_tasks(prototype)
        
        return tasks
    
    def _create_ml_tasks(self, prototype: Prototype) -> List[DevelopmentTask]:
        """Create tasks for ML prototype"""
        return [
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="Data Collection and Preprocessing",
                description="Collect and preprocess healthcare data",
                phase=DevelopmentPhase.DEVELOPMENT,
                assigned_to="ml_engineer_1",
                estimated_hours=16.0
            ),
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="Model Development",
                description="Develop and train ML model",
                phase=DevelopmentPhase.DEVELOPMENT,
                assigned_to="ml_engineer_1",
                estimated_hours=24.0
            ),
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="Model Validation",
                description="Validate model performance",
                phase=DevelopmentPhase.TESTING,
                assigned_to="qa_engineer_1",
                estimated_hours=8.0
            )
        ]
    
    def _create_api_tasks(self, prototype: Prototype) -> List[DevelopmentTask]:
        """Create tasks for API prototype"""
        return [
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="API Design and Documentation",
                description="Design RESTful API endpoints",
                phase=DevelopmentPhase.DESIGN,
                assigned_to="backend_developer_1",
                estimated_hours=12.0
            ),
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="Backend Implementation",
                description="Implement API endpoints",
                phase=DevelopmentPhase.DEVELOPMENT,
                assigned_to="backend_developer_1",
                estimated_hours=20.0
            ),
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="API Testing",
                description="Write and execute API tests",
                phase=DevelopmentPhase.TESTING,
                assigned_to="qa_engineer_1",
                estimated_hours=10.0
            )
        ]
    
    def _create_functional_tasks(self, prototype: Prototype) -> List[DevelopmentTask]:
        """Create tasks for functional prototype"""
        return [
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="UI Component Development",
                description="Develop UI components",
                phase=DevelopmentPhase.DEVELOPMENT,
                assigned_to="frontend_developer_1",
                estimated_hours=18.0
            ),
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="Integration Testing",
                description="Test component integration",
                phase=DevelopmentPhase.TESTING,
                assigned_to="qa_engineer_1",
                estimated_hours=12.0
            )
        ]
    
    def _create_wireframe_tasks(self, prototype: Prototype) -> List[DevelopmentTask]:
        """Create tasks for wireframe prototype"""
        return [
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="Wireframe Creation",
                description="Create low-fidelity wireframes",
                phase=DevelopmentPhase.DESIGN,
                assigned_to="ux_designer_1",
                estimated_hours=8.0
            ),
            DevelopmentTask(
                task_id=str(uuid.uuid4()),
                title="User Testing",
                description="Conduct wireframe testing",
                phase=DevelopmentPhase.TESTING,
                assigned_to="ux_researcher_1",
                estimated_hours=6.0
            )
        ]

class DevOpsEngine:
    """DevOps pipeline management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('DevOpsEngine')
    
    async def initialize(self):
        """Initialize DevOps engine"""
        self.logger.info("Initializing DevOps Engine...")
        return {"status": "devops_initialized"}
    
    async def initialize_repository(self, config: Dict[str, Any]):
        """Initialize code repository"""
        # Simulate repository initialization
        await asyncio.sleep(0.1)
        self.logger.info(f"Repository initialized: {config['source_repo']}")
    
    async def get_pipeline_status(self, prototype_id: str) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'status': 'running',
            'current_stage': 'build',
            'progress': 65,
            'last_run': datetime.now().isoformat()
        }
    
    async def trigger_deployment_pipeline(self, prototype_id: str, environment: str):
        """Trigger deployment pipeline"""
        self.logger.info(f"Triggering deployment pipeline for {prototype_id} to {environment}")

class CICDManager:
    """CI/CD pipeline management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('CICDManager')
    
    async def initialize(self):
        """Initialize CI/CD manager"""
        self.logger.info("Initializing CI/CD Manager...")
        return {"status": "cicd_initialized"}
    
    async def create_pipeline(self, prototype: Prototype) -> DevOpsPipeline:
        """Create CI/CD pipeline for prototype"""
        pipeline = DevOpsPipeline(
            pipeline_id=str(uuid.uuid4()),
            name=f"Pipeline for {prototype.name}",
            stages={},
            triggers=['push'],
            environment='development',
            last_run=datetime.now(),
            status='active',
            metrics={}
        )
        
        self.logger.info(f"Created pipeline: {pipeline.name}")
        return pipeline

class TestingFramework:
    """Automated testing framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('TestingFramework')
    
    async def initialize(self):
        """Initialize testing framework"""
        self.logger.info("Initializing Testing Framework...")
        return {"status": "testing_initialized"}
    
    async def generate_test_cases(self, feature: Dict[str, Any], prototype_type: str) -> List[str]:
        """Generate test cases for feature"""
        if prototype_type == 'ml_model':
            return [
                'test_model_accuracy',
                'test_data_validation',
                'test_prediction_consistency'
            ]
        elif prototype_type == 'api':
            return [
                'test_api_endpoints',
                'test_request_validation',
                'test_error_handling'
            ]
        else:
            return [
                'test_basic_functionality',
                'test_user_interface',
                'test_performance'
            ]
    
    async def setup_test_suite(self, prototype: Prototype):
        """Setup automated test suite"""
        self.logger.info(f"Setting up test suite for {prototype.prototype_id}")
    
    async def get_test_results(self, prototype_id: str) -> Dict[str, Any]:
        """Get test results"""
        return {
            'total_tests': 25,
            'passed': 23,
            'failed': 2,
            'coverage': 87.5,
            'status': 'passing'
        }

class PrototypeCreator:
    """Prototype file creation"""
    
    def __init__(self):
        self.logger = logging.getLogger('PrototypeCreator')
    
    async def create_prototype_files(self, prototype_id: str, feature: Dict[str, Any], 
                                   prototype_type: str) -> List[str]:
        """Create prototype files"""
        files = []
        
        # Create README
        readme_content = f"""# {feature.get('title', 'Prototype')}

## Description
{feature.get('description', 'AI-generated prototype')}

## Type
{prototype_type}

## Technologies
- Python
- FastAPI
- React
- Docker

## Setup
```bash
pip install -r requirements.txt
npm install
```

## Development
```bash
python main.py
npm run dev
```
"""
        
        files.append('README.md')
        
        # Create requirements.txt
        if 'python' in feature.get('technologies', []):
            files.append('requirements.txt')
        
        # Create package.json
        if 'javascript' in feature.get('technologies', []):
            files.append('package.json')
        
        # Create main implementation file
        if prototype_type == 'ml_model':
            files.append('model.py')
        elif prototype_type == 'api':
            files.append('api.py')
        else:
            files.append('app.py')
        
        return files

class DeploymentManager:
    """Deployment management"""
    
    def __init__(self):
        self.logger = logging.getLogger('DeploymentManager')
    
    async def create_deployment_config(self, prototype: Prototype):
        """Create deployment configuration"""
        config = {
            'image': f'prototype-{prototype.prototype_id}',
            'replicas': 1,
            'resources': {
                'cpu': '500m',
                'memory': '1Gi'
            }
        }
        
        self.logger.info(f"Created deployment config for {prototype.prototype_id}")
    
    async def deploy(self, prototype: Prototype, environment: str) -> Dict[str, Any]:
        """Deploy prototype"""
        # Simulate deployment
        await asyncio.sleep(2)
        
        return {
            'deployment_url': f'https://{prototype.prototype_id}.{environment}.example.com',
            'status': 'deployed',
            'version': '1.0.0'
        }
    
    async def get_deployment_status(self, prototype_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            'status': 'running',
            'url': f'https://{prototype_id}.development.example.com',
            'last_deployment': datetime.now().isoformat()
        }

async def _monitor_prototypes(self):
    """Background task to monitor prototypes"""
    while True:
        try:
            # Simulate monitoring
            await asyncio.sleep(300)  # Check every 5 minutes
            self.logger.debug("Monitoring prototypes...")
            
        except Exception as e:
            self.logger.error(f"Prototype monitoring error: {str(e)}")
            await asyncio.sleep(60)  # Retry in 1 minute

# Add the method to the RapidPrototypingEngine class
setattr(RapidPrototypingEngine, '_monitor_prototypes', 
        lambda self: _monitor_prototypes(self))

async def main():
    """Main function to demonstrate rapid prototyping engine"""
    config = {
        'agile': {'sprint_duration': '2 weeks'},
        'devops': {'platform': 'kubernetes'},
        'cicd': {'provider': 'github_actions'},
        'testing': {'framework': 'pytest'}
    }
    
    engine = RapidPrototypingEngine(config)
    
    # Initialize engine
    init_result = await engine.initialize()
    print(f"Prototyping engine initialized: {init_result}")
    
    # Sample features to prototype
    features = [
        {
            'feature_id': 'feat_001',
            'title': 'AI Diagnostic Assistant',
            'description': 'AI-powered diagnostic tool for medical images',
            'category': 'ai_diagnostics',
            'priority': 9
        },
        {
            'feature_id': 'feat_002',
            'title': 'Patient Dashboard API',
            'description': 'RESTful API for patient data management',
            'category': 'api',
            'priority': 8
        }
    ]
    
    # Create prototypes
    prototypes = await engine.create_prototypes(features)
    print(f"Created {len(prototypes)} prototypes")
    
    # Get metrics
    metrics = await engine.get_prototyping_metrics()
    print(f"Prototyping metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())