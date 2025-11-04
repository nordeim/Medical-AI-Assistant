"""
Kubernetes Auto-scaling Configuration for Healthcare AI Workloads
Implements HPA/VPA with workload prediction for medical AI applications
"""

import yaml
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class WorkloadMetrics:
    """Workload metrics for scaling decisions"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    active_sessions: int

class HealthcareWorkloadPredictor:
    """
    Predicts healthcare workload patterns for intelligent auto-scaling
    """
    
    def __init__(self):
        self.prediction_window = timedelta(hours=24)
        self.history_retention = timedelta(days=7)
        self.workload_patterns = {}
        self.prediction_models = {}
    
    def add_workload_data(self, metrics: WorkloadMetrics):
        """Add workload metrics for pattern analysis"""
        hour = metrics.timestamp.hour
        day_of_week = metrics.timestamp.weekday()
        
        if hour not in self.workload_patterns:
            self.workload_patterns[hour] = {}
        if day_of_week not in self.workload_patterns[hour]:
            self.workload_patterns[hour][day_of_week] = []
        
        self.workload_patterns[hour][day_of_week].append(metrics)
        
        # Maintain only recent history
        self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff = datetime.now() - self.history_retention
        for hour in self.workload_patterns:
            for day in self.workload_patterns[hour]:
                self.workload_patterns[hour][day] = [
                    m for m in self.workload_patterns[hour][day] 
                    if m.timestamp > cutoff
                ]
    
    def predict_workload(self, target_time: datetime) -> WorkloadMetrics:
        """Predict workload for a specific time"""
        hour = target_time.hour
        day_of_week = target_time.weekday()
        
        if hour not in self.workload_patterns:
            hour = 12  # Default to noon if no pattern
        
        # Get historical data for similar times
        similar_metrics = []
        
        # Look for same hour across different days
        if hour in self.workload_patterns:
            for day in self.workload_patterns[hour]:
                similar_metrics.extend(self.workload_patterns[hour][day])
        
        # Look for similar hours (+/- 1 hour)
        for hour_offset in [-1, 1]:
            check_hour = (hour + hour_offset) % 24
            if check_hour in self.workload_patterns:
                for day in self.workload_patterns[check_hour]:
                    similar_metrics.extend(self.workload_patterns[check_hour][day])
        
        if not similar_metrics:
            # Default predictions based on general healthcare patterns
            return self._get_default_prediction(target_time)
        
        # Calculate average metrics
        avg_cpu = np.mean([m.cpu_usage for m in similar_metrics])
        avg_memory = np.mean([m.memory_usage for m in similar_metrics])
        avg_requests = np.mean([m.request_rate for m in similar_metrics])
        avg_response_time = np.mean([m.response_time for m in similar_metrics])
        avg_sessions = np.mean([m.active_sessions for m in similar_metrics])
        
        return WorkloadMetrics(
            timestamp=target_time,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            request_rate=avg_requests,
            response_time=avg_response_time,
            active_sessions=avg_sessions
        )
    
    def _get_default_prediction(self, target_time: datetime) -> WorkloadMetrics:
        """Get default predictions based on healthcare patterns"""
        hour = target_time.hour
        
        # Healthcare has different patterns
        # Peak hours: 6-9 AM (morning rounds), 2-4 PM (afternoon rounds)
        # Low hours: 10 PM - 6 AM (night hours)
        
        if 6 <= hour <= 9:  # Morning peak
            cpu_base, memory_base, request_base = 70, 60, 100
        elif 14 <= hour <= 16:  # Afternoon peak
            cpu_base, memory_base, request_base = 60, 55, 80
        elif 22 <= hour or hour <= 6:  # Night hours
            cpu_base, memory_base, request_base = 20, 30, 10
        else:  # Regular hours
            cpu_base, memory_base, request_base = 40, 40, 40
        
        return WorkloadMetrics(
            timestamp=target_time,
            cpu_usage=cpu_base,
            memory_usage=memory_base,
            request_rate=request_base,
            response_time=1.0,
            active_sessions=int(request_base / 10)
        )


class KubernetesAutoscalingManager:
    """
    Manages Kubernetes HPA and VPA configurations for medical AI workloads
    """
    
    def __init__(self, namespace: str = "medical-ai"):
        self.namespace = namespace
        self.workload_predictor = HealthcareWorkloadPredictor()
    
    def generate_hpa_config(self, service_name: str, 
                           min_replicas: int = 2,
                           max_replicas: int = 20,
                           cpu_target: float = 70.0,
                           memory_target: float = 80.0) -> Dict:
        """
        Generate HPA configuration with medical AI optimizations
        """
        hpa_config = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{service_name}-hpa",
                "namespace": self.namespace,
                "labels": {
                    "app": service_name,
                    "component": "medical-ai",
                    "optimization": "healthcare"
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": service_name
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": cpu_target
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": memory_target
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,  # Faster scale-up for medical emergencies
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,  # Scale up 100% quickly
                                "periodSeconds": 60
                            }
                        ],
                        "selectPolicy": "Max"
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,  # Slower scale-down for stability
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 10,  # Scale down only 10% at a time
                                "periodSeconds": 60
                            }
                        ],
                        "selectPolicy": "Min"
                    }
                }
            }
        }
        
        # Add custom metrics for medical AI
        hpa_config["spec"]["metrics"].append({
            "type": "Pods",
            "pods": {
                "metric": {
                    "name": "medical_ai_request_rate"
                },
                "target": {
                    "type": "AverageValue",
                    "averageValue": "100"
                }
            }
        })
        
        return hpa_config
    
    def generate_vpa_config(self, service_name: str,
                          min_memory: str = "512Mi",
                          max_memory: str = "8Gi",
                          min_cpu: str = "500m",
                          max_cpu: str = "4000m") -> Dict:
        """
        Generate VPA configuration for resource optimization
        """
        vpa_config = {
            "apiVersion": "autoscaling.k8s.io/v1",
            "kind": "VerticalPodAutoscaler",
            "metadata": {
                "name": f"{service_name}-vpa",
                "namespace": self.namespace,
                "labels": {
                    "app": service_name,
                    "component": "medical-ai",
                    "optimization": "healthcare"
                }
            },
            "spec": {
                "targetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": service_name
                },
                "updatePolicy": {
                    "updateMode": "Auto"  # Automatically update pod resources
                },
                "resourcePolicy": {
                    "containerPolicies": [
                        {
                            "containerName": service_name,
                            "minAllowed": {
                                "cpu": min_cpu,
                                "memory": min_memory
                            },
                            "maxAllowed": {
                                "cpu": max_cpu,
                                "memory": max_memory
                            },
                            "controlledResources": ["cpu", "memory"],
                            "controlledValues": "RequestsAndLimits"
                        }
                    ]
                }
            }
        }
        
        return vpa_config
    
    def generate_predictive_hpa_config(self, service_name: str,
                                     prediction_enabled: bool = True) -> Dict:
        """
        Generate HPA with predictive scaling based on healthcare patterns
        """
        hpa_config = self.generate_hpa_config(service_name)
        
        if prediction_enabled:
            # Add custom metric for predictive scaling
            hpa_config["spec"]["metrics"].append({
                "type": "Object",
                "object": {
                    "metric": {
                        "name": "predicted_workload_score"
                    },
                    "target": {
                        "type": "Value",
                        "value": "80"
                    },
                    "describedObject": {
                        "apiVersion": "v1",
                        "kind": "ConfigMap",
                        "name": f"{service_name}-predictions"
                    }
                }
            })
        
        return hpa_config
    
    def create_deployment_with_annotations(self, service_name: str,
                                          image: str,
                                          replicas: int = 3) -> Dict:
        """
        Create deployment with auto-scaling annotations
        """
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service_name,
                "namespace": self.namespace,
                "labels": {
                    "app": service_name,
                    "component": "medical-ai"
                },
                "annotations": {
                    "medical-ai/workload-pattern": "healthcare",
                    "medical-ai/critical-service": "true",
                    "medical-ai/scale-priority": "high"
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": service_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service_name
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8080",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": service_name,
                                "image": image,
                                "ports": [
                                    {
                                        "containerPort": 8080,
                                        "name": "http"
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "1000m",
                                        "memory": "1Gi"
                                    },
                                    "limits": {
                                        "cpu": "2000m",
                                        "memory": "4Gi"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                },
                                "env": [
                                    {
                                        "name": "SERVICE_NAME",
                                        "value": service_name
                                    },
                                    {
                                        "name": "NAMESPACE",
                                        "value": self.namespace
                                    }
                                ]
                            }
                        ],
                        "affinity": {
                            "podAntiAffinity": {
                                "preferredDuringSchedulingIgnoredDuringExecution": [
                                    {
                                        "weight": 100,
                                        "podAffinityTerm": {
                                            "labelSelector": {
                                                "matchExpressions": [
                                                    {
                                                        "key": "app",
                                                        "operator": "In",
                                                        "values": [service_name]
                                                    }
                                                ]
                                            },
                                            "topologyKey": "kubernetes.io/hostname"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        
        return deployment


class MedicalAIServiceAutoscalingConfig:
    """
    Complete auto-scaling configuration for medical AI services
    """
    
    def __init__(self, namespace: str = "medical-ai"):
        self.namespace = namespace
        self.k8s_manager = KubernetesAutoscalingManager(namespace)
    
    def generate_complete_config(self, service_config: Dict) -> Dict[str, str]:
        """
        Generate complete auto-scaling configuration for a medical AI service
        """
        service_name = service_config['name']
        image = service_config['image']
        service_type = service_config.get('type', 'api')
        
        # Generate all configuration files
        configs = {}
        
        # 1. Deployment with annotations
        deployment = self.k8s_manager.create_deployment_with_annotations(
            service_name, image, replicas=service_config.get('replicas', 3)
        )
        configs[f"{service_name}-deployment.yaml"] = yaml.dump(deployment, default_flow_style=False)
        
        # 2. HPA configuration
        if service_type == 'api':
            hpa = self.k8s_manager.generate_hpa_config(
                service_name,
                min_replicas=3,
                max_replicas=20,
                cpu_target=70.0,
                memory_target=80.0
            )
        elif service_type == 'model-serving':
            hpa = self.k8s_manager.generate_hpa_config(
                service_name,
                min_replicas=1,
                max_replicas=10,
                cpu_target=80.0,
                memory_target=85.0
            )
        else:
            hpa = self.k8s_manager.generate_hpa_config(service_name)
        
        configs[f"{service_name}-hpa.yaml"] = yaml.dump(hpa, default_flow_style=False)
        
        # 3. VPA configuration
        vpa = self.k8s_manager.generate_vpa_config(service_name)
        configs[f"{service_name}-vpa.yaml"] = yaml.dump(vpa, default_flow_style=False)
        
        # 4. Predictive HPA (optional)
        if service_config.get('predictive_scaling', False):
            predictive_hpa = self.k8s_manager.generate_predictive_hpa_config(service_name)
            configs[f"{service_name}-predictive-hpa.yaml"] = yaml.dump(predictive_hpa, default_flow_style=False)
        
        # 5. Service definition
        service = self._generate_service_config(service_name, service_config.get('ports', [8080]))
        configs[f"{service_name}-service.yaml"] = yaml.dump(service, default_flow_style=False)
        
        return configs
    
    def _generate_service_config(self, service_name: str, ports: List[int]) -> Dict:
        """Generate Kubernetes Service configuration"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service_name,
                "namespace": self.namespace,
                "labels": {
                    "app": service_name,
                    "component": "medical-ai"
                }
            },
            "spec": {
                "selector": {
                    "app": service_name
                },
                "ports": [
                    {
                        "name": "http",
                        "port": port,
                        "targetPort": port,
                        "protocol": "TCP"
                    } for port in ports
                ],
                "type": "ClusterIP"
            }
        }


async def main():
    """Example usage of auto-scaling configuration"""
    
    # Medical AI service configurations
    services = [
        {
            'name': 'medical-ai-api',
            'image': 'medical-ai/api:latest',
            'type': 'api',
            'replicas': 3,
            'predictive_scaling': True,
            'ports': [8080]
        },
        {
            'name': 'model-serving',
            'image': 'medical-ai/model-server:latest',
            'type': 'model-serving',
            'replicas': 2,
            'predictive_scaling': True,
            'ports': [8080, 8081]
        },
        {
            'name': 'patient-data-service',
            'image': 'medical-ai/patient-service:latest',
            'type': 'api',
            'replicas': 2,
            'predictive_scaling': False,
            'ports': [8080]
        }
    ]
    
    # Generate configurations
    autoscaler = MedicalAIServiceAutoscalingConfig()
    
    for service_config in services:
        configs = autoscaler.generate_complete_config(service_config)
        
        print(f"\n=== Configuration for {service_config['name']} ===")
        for filename, config_yaml in configs.items():
            print(f"\n--- {filename} ---")
            print(config_yaml[:500] + "..." if len(config_yaml) > 500 else config_yaml)


if __name__ == "__main__":
    asyncio.run(main())