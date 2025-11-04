# Medical AI Assistant - Comprehensive Deployment Guide

**Version:** 1.0.0  
**Last Updated:** November 4, 2025  
**Compliance:** HIPAA, FDA, ISO 27001  
**Healthcare Grade:** Production Ready 🏥

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Deployment](#local-development-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Production Configuration](#production-configuration)
7. [Scaling Considerations](#scaling-considerations)
8. [Monitoring Setup](#monitoring-setup)
9. [Security Configuration](#security-configuration)
10. [Troubleshooting](#troubleshooting)
11. [Compliance](#compliance)

---

## Overview

The Medical AI Assistant is a healthcare-compliant AI platform requiring specialized deployment procedures for Protected Health Information (PHI) handling. This guide covers deployment across multiple environments while maintaining HIPAA, FDA, and ISO 27001 compliance.

### Architecture Overview

```mermaid
graph TB
    subgraph "External Layer"
        A[Load Balancer<br/>SSL/TLS Termination]
        B[WAF<br/>Web Application Firewall]
    end
    
    subgraph "Kubernetes Cluster"
        C[Nginx Ingress<br/>SSL & Routing]
        D[Frontend<br/>React/TypeScript]
        E[Backend<br/>FastAPI/Python]
        F[Model Serving<br/>PyTorch/CUDA]
        G[Database<br/>PostgreSQL]
        H[Cache<br/>Redis Cluster]
    end
    
    subgraph "Data Layer"
        I[Backup System<br/>Encrypted Storage]
        J[Audit Logs<br/>Compliance Storage]
        K[Monitoring<br/>Prometheus/Grafana]
    end
    
    A --> C
    B --> C
    C --> D
    C --> E
    E --> F
    E --> G
    E --> H
    F --> H
    D --> E
    I --> G
    J --> K
    K --> E
```

---

## Prerequisites

### System Requirements

#### Development Environment
- **Operating System:** Linux, macOS, or Windows 10+
- **Memory:** 16GB RAM minimum, 32GB recommended
- **Storage:** 100GB available space
- **CPU:** 8 cores minimum, 16 cores recommended
- **GPU:** Optional (for local model testing)

#### Production Environment
- **Cloud Provider:** AWS, Google Cloud, or Azure
- **Kubernetes:** v1.25+ with multi-AZ support
- **Database:** PostgreSQL 15+ with encryption
- **Cache:** Redis 7+ with authentication
- **Load Balancer:** Application load balancer with SSL termination
- **Monitoring:** Prometheus, Grafana, AlertManager

### Required Tools

```bash
# Core tools
kubectl >= 1.25              # Kubernetes CLI
helm >= 3.10                 # Kubernetes package manager
terraform >= 1.0             # Infrastructure as code
docker >= 20.0               # Container runtime
docker-compose >= 2.0        # Multi-container orchestration

# Cloud provider CLI (choose appropriate)
aws-cli                      # AWS command line
gcloud                       # Google Cloud SDK
az                           # Azure CLI

# Security and monitoring tools
trivy                       # Container security scanning
helmfile                    # Helm charts management
prometheus                  # Metrics collection
grafana                     # Monitoring dashboards
```

### Environment Variables

Create environment configuration file:

```bash
# .env.production
DOMAIN_NAME=medical-ai.example.com
ENVIRONMENT=production
DATABASE_URL=postgresql://user:password@postgres:5432/medical_ai
REDIS_URL=redis://redis:6379/0
SECRET_KEY=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)
ALLOWED_HOSTS=medical-ai.example.com,api.medical-ai.example.com
CORS_ORIGINS=https://medical-ai.example.com
HIPAA_COMPLIANCE=true
AUDIT_LOGGING=true
```

---

## Local Development Deployment

### Quick Start with Docker Compose

1. **Clone and Setup**
```bash
git clone https://github.com/your-org/medical-ai-assistant.git
cd medical-ai-assistant

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

2. **Start Development Environment**
```bash
# Start all services
docker-compose -f docker/docker-compose.dev.yml up -d

# Check service status
docker-compose -f docker/docker-compose.dev.yml ps
```

3. **Access Applications**
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000

### Development Docker Compose Configuration

```yaml
# docker/docker-compose.dev.yml
version: '3.8'

services:
  # Frontend Development
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
      target: development
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000
      - VITE_ENVIRONMENT=development
    depends_on:
      - backend
    networks:
      - medical-ai-dev

  # Backend Development
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
      target: development
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./models:/app/models
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/medical_ai_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - medical-ai-dev
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=medical_ai_dev
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - medical-ai-dev

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - medical-ai-dev
    command: redis-server --appendonly yes

  # Model Serving (Development)
  serving:
    build:
      context: .
      dockerfile: docker/Dockerfile.serving
      target: development
    ports:
      - "8081:8081"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_CACHE_SIZE=10
    networks:
      - medical-ai-dev
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m serving.main --host 0.0.0.0 --port 8081 --reload

  # MinIO for Local File Storage
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin123
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    networks:
      - medical-ai-dev

volumes:
  postgres_data:
  minio_data:

networks:
  medical-ai-dev:
    driver: bridge
```

### Development-Specific Configurations

#### Frontend Development
```bash
# Install dependencies
cd frontend && npm install

# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Run tests
npm run test
npm run test:e2e
```

#### Backend Development
```bash
# Install Python dependencies
pip install -r requirements-dev.txt

# Initialize database
python scripts/init_db.py

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v
```

---

## Cloud Deployment

### AWS EKS Deployment

#### 1. Terraform Infrastructure Setup

```bash
# Navigate to cloud configuration
cd deployment/cloud

# Initialize Terraform
terraform init

# Create variables file
cat > aws-production.tfvars << EOF
aws_region = "us-east-1"
environment = "production"
cluster_name = "medical-ai-prod"
vpc_cidr = "10.0.0.0/16"
private_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
public_subnet_cidrs = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
database_password = "$(openssl rand -base64 32)"
redis_password = "$(openssl rand -base64 32)"
domain_name = "medical-ai.example.com"
certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
EOF

# Plan infrastructure
terraform plan -var-file=aws-production.tfvars

# Apply infrastructure
terraform apply -var-file=aws-production.tfvars

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name medical-ai-prod
```

#### 2. EKS Terraform Configuration

```hcl
# deployment/cloud/aws-eks-terraform.tf (excerpt)
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = concat(aws_subnet.private_subnets[*].id, aws_subnet.public_subnets[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
    
    # Security groups for healthcare compliance
    security_group_ids = [
      aws_security_group.cluster_sg.id,
      aws_security_group.cluster_endpoint_sg.id
    ]
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  tags = {
    Environment  = var.environment
    Project      = "medical-ai"
    Compliance   = "HIPAA"
    DataClass    = "PHI"
  }
}

resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "medical-ai-workers"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private_subnets[*].id

  # GPU node group for model serving
  instance_types = ["p3.2xlarge", "p4d.xlarge"]

  capacity_type  = "ON_DEMAND"
  disk_size      = 100
  
  # Enable container insights for monitoring
  container_images {
    # Custom images with healthcare compliance
    ami_type = "CUSTOM"
    ami_id   = data.aws_ami.eks_ubuntu.id
  }

  labels = {
    workload-type = "medical-ai"
    tier          = "production"
  }

  taint {
    key    = "medical-workload"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  # Security settings for healthcare
  remote_access {
    ec2_ssh_key = aws_key_pair.eks.key_name
    source_security_groups = [aws_security_group.node_sg.id]
  }

  tags = {
    Environment  = var.environment
    Project      = "medical-ai"
  }
}

# HIPAA-compliant KMS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS encryption key for Medical AI"
  deletion_window_in_days = 30
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow EKS Service"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Environment  = var.environment
    Project      = "medical-ai"
    Compliance   = "HIPAA"
  }
}
```

#### 3. Deploy to EKS

```bash
# Set up environment variables
export KUBECONFIG=~/.kube/config
export NAMESPACE=medical-ai-prod

# Create namespace and RBAC
kubectl apply -f deployment/kubernetes/00-namespace-rbac.yaml

# Deploy application using Helm
helm upgrade --install medical-ai ./helm-chart \
  --namespace $NAMESPACE \
  --values ./helm-chart/values-production.yaml \
  --set image.tag=latest \
  --set environment=production

# Verify deployment
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE
```

### Google Cloud GKE Deployment

#### 1. GKE Infrastructure Setup

```bash
# Set up gcloud CLI
gcloud init
gcloud auth configure-docker

# Create variables file
cat > gcp-production.tfvars << EOF
project_id = "medical-ai-production"
region     = "us-central1"
zone       = "us-central1-a"
cluster_name = "medical-ai-prod"
network    = "medical-ai-vpc"
subnetwork = "medical-ai-subnet"
database_password = "$(openssl rand -base64 32)"
redis_password = "$(openssl rand -base64 32)"
domain_name = "medical-ai.example.com"
certificate_id = "projects/medical-ai-production/global/sslCertificates/medical-ai-cert"
EOF

# Initialize and deploy Terraform
terraform init
terraform plan -var-file=gcp-production.tfvars
terraform apply -var-file=gcp-production.tfvars
```

#### 2. GKE Terraform Configuration

```hcl
# deployment/cloud/gcp-gke-terraform.tf (excerpt)
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # Enable GKE Autopilot for simplified management
  enable_autopilot = true

  # Network policy for security
  network_policy {
    enabled = true
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Security configurations
  enable_legacy_abac      = false
  enable_shielded_nodes   = true
  enable_intranode_visibility = true

  # HIPAA compliance settings
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  addons_config {
    network_policy_config {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    vertical_pod_autoscaling {
      enabled = true
    }
    dns_cache_config {
      enabled = true
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }

  network_policy {
    enabled = true
  }

  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "API_SERVER",
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
    ]
    
    managed_prometheus {
      enabled = true
    }
  }

  tags = [
    "medical-ai",
    "production",
    "hipaa-compliant",
  ]

  resource_labels = {
    environment  = var.environment
    project      = "medical-ai"
    compliance   = "HIPAA"
    data-class   = "PHI"
  }

  removal_policy = "ABANDON"
}

# GPU node pool for model serving
resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 1

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    service_account = google_service_account.node.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    labels = {
      accelerator = "nvidia-tesla-v100"
      tier        = "gpu"
    }

    tags = ["gpu-node"]
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 5
    location_policy = "BALANCED"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
```

### Azure AKS Deployment

#### 1. AKS Infrastructure Setup

```bash
# Login to Azure
az login
az account set --subscription "Medical AI Production"

# Create variables file
cat > azure-production.tfvars << EOF
location = "East US"
resource_group_name = "rg-medical-ai-prod"
cluster_name = "aks-medical-ai-prod"
dns_prefix = "medical-ai"
database_password = "$(openssl rand -base64 32)"
redis_password = "$(openssl rand -base64 32)"
domain_name = "medical-ai.example.com"
certificate_secret = "medical-ai-cert"
EOF

# Initialize and deploy Terraform
terraform init
terraform plan -var-file=azure-production.tfvars
terraform apply -var-file=azure-production.tfvars
```

#### 2. AKS Terraform Configuration

```hcl
# deployment/cloud/azure-aks-terraform.tf (excerpt)
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${var.cluster_name}"
  address_space       = ["10.0.0.0/8"]
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

resource "azurerm_subnet" "main" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.240.0.0/16"]
  service_endpoints    = ["Microsoft.Sql", "Microsoft.Storage"]
}

resource "azurerm_container_registry" "main" {
  name                = "acrmedicalaiprod"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "Premium"
  admin_enabled       = false

  georeplications {
    location = var.location
    zone_redundancy {
      enabled = true
    }
  }

  network_rule_set {
    default_action = "Deny"
    ip_rule {
      action   = "Allow"
      ip_range = "your-ip-range/32"
    }
  }

  trust_policy {
    enabled = true
  }

  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

resource "azurerm_kubernetes_cluster" "main" {
  name                = var.cluster_name
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.dns_prefix
  node_resource_group = "rg-${var.cluster_name}-nodes"

  default_node_pool {
    name                = "systempool"
    vm_size             = "Standard_D4s_v3"
    node_count          = 3
    min_count           = 3
    max_count           = 10
    enable_auto_scaling = true
    vnet_subnet_id      = azurerm_subnet.main.id
    os_disk_size_gb     = 100
    os_disk_type        = "Ephemeral"
    type                = "VirtualMachineScaleSets"
    node_labels = {
      "nodepool-type"    = "system"
      "application"      = "medical-ai"
      "compliance"       = "hipaa"
    }
    node_taints = [
      "medical-workload=true:NoSchedule"
    ]
  }

  identity {
    type = "SystemAssigned"
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  azure_policy {
    enabled = true
  }

  # Network configuration
  network_profile {
    network_plugin     = "azure"
    network_policy     = "azure"
    service_cidr       = "10.2.0.0/16"
    dns_service_ip     = "10.2.0.10"
    docker_bridge_cidr = "172.17.0.1/16"
  }

  # RBAC and Azure AD integration
  role_based_access_control {
    enabled = true
    azure_active_directory {
      managed                = true
      admin_group_object_ids = ["your-admin-group-id"]
    }
  }

  # Monitor add-on
  oms_agent {
    enabled = true
  }

  # Private cluster for enhanced security
  private_cluster_enabled = true
  private_dns_zone_id     = "/subscriptions/${data.azurerm_subscription.current.subscription_id}/resourceGroups/${var.resource_group_name}/providers/Microsoft.ContainerService/managedClusters/${var.cluster_name}/managedPrivateDNS"

  # API server access
  api_server_authorized_ip_ranges = ["your-ip-range/32"]

  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

# GPU node pool for model serving
resource "azurerm_kubernetes_cluster_node_pool" "gpu_nodes" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = "Standard_NC6s_v3"
  node_count            = 0
  min_count             = 0
  max_count             = 5
  enable_auto_scaling   = true
  node_taints           = ["sku=gpu:NoSchedule"]
  labels = {
    "accelerator" = "nvidia-tesla-v100"
    "tier"        = "gpu"
  }
}
```

---

## Kubernetes Deployment

### Production Kubernetes Manifests

#### 1. Namespace and RBAC Configuration

```yaml
# deployment/kubernetes/00-namespace-rbac.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: medical-ai-prod
  labels:
    name: medical-ai-prod
    environment: production
    compliance: hipaa
    data-classification: phi
    healthcare-system: "true"
    audit-enabled: "true"
    encryption-required: "true"
  annotations:
    description: "Production namespace for Medical AI Assistant with HIPAA compliance"
    contact: "devops@medical-ai.example.com"
    compliance-requirements: "HIPAA,FDA,ISO27001"
    data-retention-policy: "7-years"
    backup-frequency: "daily"
    monitoring-enabled: "true"

---
# Network Policies for security
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: medical-ai-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Resource quotas
apiVersion: v1
kind: ResourceQuota
metadata:
  name: medical-ai-quota
  namespace: medical-ai-prod
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 200Gi
    limits.cpu: "100"
    limits.memory: 400Gi
    persistentvolumeclaims: "50"
    requests.storage: 2Ti
    pods: "100"
    services: "20"
    secrets: "100"
    configmaps: "50"

---
# Limit ranges
apiVersion: v1
kind: LimitRange
metadata:
  name: medical-ai-limits
  namespace: medical-ai-prod
spec:
  limits:
  - type: Container
    default:
      cpu: "4"
      memory: 8Gi
    defaultRequest:
      cpu: "200m"
      memory: 1Gi
    max:
      cpu: "16"
      memory: 32Gi
    min:
      cpu: "100m"
      memory: 256Mi
  - type: Pod
    max:
      cpu: "32"
      memory: 64Gi
```

#### 2. Frontend Deployment

```yaml
# deployment/kubernetes/01-frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-ai-frontend
  namespace: medical-ai-prod
  labels:
    app: frontend
    tier: frontend
    environment: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  selector:
    matchLabels:
      app: frontend
      tier: frontend
  template:
    metadata:
      labels:
        app: frontend
        tier: frontend
        environment: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: medical-ai-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: frontend
        image: medical-ai/frontend:v1.0.0
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        env:
        - name: NODE_ENV
          value: "production"
        - name: API_URL
          value: "https://api.medical-ai.example.com"
        - name: ENABLE_ANALYTICS
          value: "true"
        resources:
          requests:
            cpu: "200m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: [ALL]
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
        - name: var-run-volume
          mountPath: /var/run
      volumes:
      - name: tmp-volume
        emptyDir: {}
      - name: var-run-volume
        emptyDir: {}
      tolerations:
      - key: "medical-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      nodeSelector:
        workload-type: frontend

---
apiVersion: v1
kind: Service
metadata:
  name: medical-ai-frontend
  namespace: medical-ai-prod
  labels:
    app: frontend
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 443
    targetPort: 8080
    protocol: TCP
    name: https

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medical-ai-frontend-hpa
  namespace: medical-ai-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-ai-frontend
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30
```

#### 3. Backend API Deployment

```yaml
# deployment/kubernetes/02-backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-ai-backend
  namespace: medical-ai-prod
  labels:
    app: backend
    tier: api
    environment: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: backend
      tier: api
  template:
    metadata:
      labels:
        app: backend
        tier: api
        environment: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: medical-ai-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: backend
        image: medical-ai/backend:v1.0.0
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: medical-ai-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: medical-ai-secrets
              key: redis-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: medical-ai-secrets
              key: secret-key
        - name: HIPAA_COMPLIANCE
          value: "true"
        - name: AUDIT_LOGGING
          value: "true"
        - name: MODEL_CACHE_SIZE
          value: "1000"
        - name: MAX_CONCURRENT_REQUESTS
          value: "50"
        - name: REQUEST_TIMEOUT
          value: "60"
        resources:
          requests:
            cpu: "500m"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: "/health/ready"
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: "/health"
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 10
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: [ALL]
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: tmp-volume
          mountPath: /tmp
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: tmp-volume
        emptyDir: {}
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
      tolerations:
      - key: "medical-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

---
apiVersion: v1
kind: Service
metadata:
  name: medical-ai-backend
  namespace: medical-ai-prod
  labels:
    app: backend
spec:
  type: ClusterIP
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medical-ai-backend-hpa
  namespace: medical-ai-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-ai-backend
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: active_sessions
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30
```

#### 4. Model Serving Deployment

```yaml
# deployment/kubernetes/03-serving-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-ai-serving
  namespace: medical-ai-prod
  labels:
    app: serving
    tier: ml-inference
    environment: production
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  selector:
    matchLabels:
      app: serving
      tier: ml-inference
  template:
    metadata:
      labels:
        app: serving
        tier: ml-inference
        environment: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8081"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: medical-ai-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: serving
        image: medical-ai/serving:v1.0.0
        ports:
        - containerPort: 8081
          name: http
          protocol: TCP
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: OMP_NUM_THREADS
          value: "1"
        - name: TOKENIZERS_PARALLELISM
          value: "false"
        - name: CUDA_MEMORY_FRACTION
          value: "0.8"
        - name: MODEL_CACHE_SIZE
          value: "50"
        - name: MAX_BATCH_SIZE
          value: "32"
        - name: MAX_SEQUENCE_LENGTH
          value: "512"
        - name: INFERENCE_TIMEOUT
          value: "300"
        - name: ENABLE_METRICS
          value: "true"
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "8"
            memory: "32Gi"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: "/health"
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 60
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: "/ready"
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: [ALL]
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: cache-volume
          mountPath: /app/cache
        - name: tmp-volume
          mountPath: /tmp
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: "medical-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      - key: "sku"
        operator: "Equal"
        value: "gpu"
        effect: "NoSchedule"
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: cache-volume
        persistentVolumeClaim:
          claimName: cache-pvc
      - name: tmp-volume
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: medical-ai-serving
  namespace: medical-ai-prod
  labels:
    app: serving
spec:
  type: ClusterIP
  selector:
    app: serving
  ports:
  - port: 8081
    targetPort: 8081
    protocol: TCP
    name: http
```

#### 5. Database and Cache Deployment

```yaml
# deployment/kubernetes/04-database-cache-deployment.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: medical-ai-prod
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: ssd

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: medical-ai-postgres
  namespace: medical-ai-prod
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9187"
    spec:
      serviceAccountName: medical-ai-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 70
        fsGroup: 70
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          value: medical_ai
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: medical-ai-secrets
              key: db-user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: medical-ai-secrets
              key: db-password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        - name: POSTGRES_INITDB_ARGS
          value: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
            - -d
            - $(POSTGRES_DB)
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - $(POSTGRES_USER)
            - -d
            - $(POSTGRES_DB)
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: config-volume
          mountPath: /etc/postgresql
      volumes:
      - name: config-volume
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Gi
      storageClassName: ssd

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: medical-ai-prod
  labels:
    app: postgres
spec:
  clusterIP: None
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
    name: postgres

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-ai-redis
  namespace: medical-ai-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: redis
        command:
        - redis-server
        - --requirepass
        - "$(REDIS_PASSWORD)"
        - --appendonly
        - "yes"
        - --appendfsync
        - "everysec"
        - --maxmemory
        - "1gb"
        - --maxmemory-policy
        - "allkeys-lru"
        resources:
          requests:
            cpu: "200m"
            memory: "1Gi"
          limits:
            cpu: "1"
            memory: "2Gi"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: medical-ai-prod
  labels:
    app: redis
spec:
  type: ClusterIP
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
    name: redis
```

### Deploy All Kubernetes Resources

```bash
#!/bin/bash
# deployment/kubernetes/deploy.sh

set -euo pipefail

NAMESPACE="medical-ai-prod"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply resources in order
echo "Applying namespace and RBAC..."
kubectl apply -f 00-namespace-rbac.yaml

echo "Waiting for admission controllers..."
kubectl wait --for=condition=admissionwebhookconfigurations/mutating-webhook-configs --timeout=30s || true

echo "Deploying frontend..."
kubectl apply -f 01-frontend-deployment.yaml

echo "Deploying backend..."
kubectl apply -f 02-backend-deployment.yaml

echo "Deploying model serving..."
kubectl apply -f 03-serving-deployment.yaml

echo "Deploying database and cache..."
kubectl apply -f 04-database-cache-deployment.yaml

echo "Deploying monitoring..."
kubectl apply -f 05-monitoring-deployment.yaml

echo "Setting up load balancing and autoscaling..."
kubectl apply -f 06-load-balancing-autoscaling.yaml

echo "Waiting for all pods to be ready..."
kubectl wait --for=condition=Ready pods --all -n $NAMESPACE --timeout=600s

echo "Deployment completed successfully!"
echo "Timestamp: $TIMESTAMP"

# Show status
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE
```

---

## Production Configuration

### Environment Configuration

```yaml
# production-config.env
# Environment Configuration for Medical AI Assistant
# HIPAA-compliant production settings

# === BASIC CONFIGURATION ===
ENVIRONMENT=production
APP_NAME=MedicalAIAssistant
APP_VERSION=1.0.0
DOMAIN_NAME=medical-ai.example.com
API_DOMAIN=api.medical-ai.example.com

# === DATABASE CONFIGURATION ===
DB_ENGINE=postgresql
DB_HOST=postgres-service.medical-ai-prod.svc.cluster.local
DB_PORT=5432
DB_NAME=medical_ai
DB_USER=medical_ai_user
DB_PASSWORD=<secure-password>
DB_SSL_MODE=require
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=0
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# === REDIS CONFIGURATION ===
REDIS_HOST=redis-service.medical-ai-prod.svc.cluster.local
REDIS_PORT=6379
REDIS_PASSWORD=<secure-password>
REDIS_SSL=false
REDIS_MAX_CONNECTIONS=100
REDIS_TIMEOUT=5
REDIS_DB=0

# === SECURITY CONFIGURATION ===
SECRET_KEY=<32-char-secret-key>
ENCRYPTION_KEY=<base64-encoded-key>
JWT_SECRET_KEY=<jwt-secret>
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
SESSION_TIMEOUT_MINUTES=60
CORS_ORIGINS=https://medical-ai.example.com,https://www.medical-ai.example.com

# === HIPAA COMPLIANCE ===
HIPAA_COMPLIANCE=true
AUDIT_LOGGING=true
ENCRYPTION_AT_REST=true
DATA_RETENTION_DAYS=2555  # 7 years
BACKUP_ENCRYPTION=true
AUDIT_LOG_RETENTION_DAYS=2555

# === API CONFIGURATION ===
API_VERSION=v1
API_PREFIX=/api/v1
API_TITLE=Medical AI Assistant API
API_DESCRIPTION=Production API for Medical AI Assistant
API_CONTACT_EMAIL=support@medical-ai.example.com
API_LICENSE_NAME=Proprietary
MAX_REQUEST_SIZE=10MB
REQUEST_TIMEOUT=60
CONCURRENT_REQUESTS=100

# === MODEL CONFIGURATION ===
MODEL_CACHE_SIZE=1000
MAX_CONCURRENT_INFERENCES=50
MODEL_TIMEOUT=300
GPU_MEMORY_FRACTION=0.8
BATCH_SIZE=32
SEQUENCE_LENGTH=512
PRECISION=float16

# === MONITORING CONFIGURATION ===
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
METRICS_PATH=/metrics
HEALTH_CHECK_PATH=/health
READY_CHECK_PATH=/ready
ALERT_MANAGER_URL=https://alertmanager.medical-ai.example.com

# === LOGGING CONFIGURATION ===
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=/app/logs/application.log
MAX_LOG_SIZE=100MB
MAX_LOG_FILES=30
AUDIT_LOG_LEVEL=INFO
AUDIT_LOG_FILE=/app/logs/audit.log

# === RATE LIMITING ===
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10
API_RATE_LIMIT_REQUESTS_PER_HOUR=1000

# === CACHE CONFIGURATION ===
CACHE_DEFAULT_TTL=3600
CACHE_USER_SESSION_TTL=1800
CACHE_MODEL_RESPONSE_TTL=7200
CACHE_ENABLED=true
CACHE_BACKEND=redis

# === EMAIL CONFIGURATION ===
SMTP_SERVER=smtp.medical-ai.example.com
SMTP_PORT=587
SMTP_USERNAME=noreply@medical-ai.example.com
SMTP_PASSWORD=<smtp-password>
EMAIL_FROM=noreply@medical-ai.example.com
EMAIL_USE_TLS=true

# === BACKUP CONFIGURATION ===
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
BACKUP_RETENTION_DAYS=2555
BACKUP_ENCRYPTION_KEY=<backup-encryption-key>
BACKUP_STORAGE_S3=s3://medical-ai-backups
BACKUP_COMPRESSION=true

# === PERFORMANCE CONFIGURATION ===
WORKERS=4
MAX_WORKERS=16
MAX_WORKER_CONNECTIONS=1000
WORKER_CLASS=uvicorn.workers.UvicornWorker
TIMEOUT_KEEP_ALIVE=65
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=100

# === FEATURE FLAGS ===
FEATURE_FLAGS_CHAT_ENABLED=true
FEATURE_FLAGS_VIDEO_ENABLED=true
FEATURE_FLAGS_ANALYTICS_ENABLED=true
FEATURE_FLAGS_AUDIT_ENABLED=true
FEATURE_FLAGS_MODEL_DRIFT_ENABLED=true
```

### Nginx Configuration

```nginx
# deployment/nginx/nginx.prod.conf
# Production Nginx configuration for Medical AI Assistant
# HIPAA-compliant reverse proxy with security headers

user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # Basic Settings
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging Format (HIPAA-compliant)
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    '$ssl_protocol $ssl_cipher '
                    '$request_time $upstream_response_time';
    
    access_log /var/log/nginx/access.log main;
    
    # Security Headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' https://api.medical-ai.example.com" always;
    
    # HIPAA-specific headers
    add_header X-PHI-Protected "1" always;
    add_header Cache-Control "no-cache, no-store, must-revalidate, private" always;
    add_header Pragma "no-cache" always;
    add_header Expires "0" always;
    
    # Performance Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;
    
    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Upstream configuration
    upstream backend {
        least_conn;
        server backend-service-1.medical-ai-prod.svc.cluster.local:8000 max_fails=3 fail_timeout=30s;
        server backend-service-2.medical-ai-prod.svc.cluster.local:8000 max_fails=3 fail_timeout=30s;
        server backend-service-3.medical-ai-prod.svc.cluster.local:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream serving {
        ip_hash;
        server serving-service-1.medical-ai-prod.svc.cluster.local:8081 max_fails=3 fail_timeout=30s;
        server serving-service-2.medical-ai-prod.svc.cluster.local:8081 max_fails=3 fail_timeout=30s;
        keepalive 16;
    }
    
    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name medical-ai.example.com www.medical-ai.example.com;
        return 301 https://$server_name$request_uri;
    }
    
    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name medical-ai.example.com www.medical-ai.example.com;
        
        # SSL Certificates
        ssl_certificate /etc/ssl/certs/medical-ai.example.com.crt;
        ssl_certificate_key /etc/ssl/private/medical-ai.example.com.key;
        
        # HSTS
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
        
        # Root location
        location / {
            root /usr/share/nginx/html;
            try_files $uri $uri/ /index.html;
            
            # Cache static assets
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }
        
        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # Security headers for API
            add_header X-Content-Type-Options nosniff always;
            add_header X-Frame-Options DENY always;
        }
        
        # Model serving routes
        location /serving/ {
            limit_req zone=api burst=10 nodelay;
            
            proxy_pass http://serving;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        # Health checks
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        # Metrics endpoint
        location /metrics {
            proxy_pass http://backend/metrics;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Restrict metrics access
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }
        
        # WebSocket support for real-time features
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 86400;
        }
        
        # Block access to sensitive files
        location ~ /\. {
            deny all;
        }
        
        location ~ \.(htaccess|htpasswd|ini|log|sh|sql|conf)$ {
            deny all;
        }
    }
}
```

---

## Scaling Considerations

### Auto-scaling Configuration

#### Horizontal Pod Autoscaler (HPA)

```yaml
# Advanced HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medical-ai-backend-hpa
  namespace: medical-ai-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-ai-backend
  minReplicas: 5
  maxReplicas: 100
  metrics:
  # CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory utilization
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom application metrics
  - type: Pods
    pods:
      metric:
        name: active_user_sessions
      target:
        type: AverageValue
        averageValue: "100"
  - type: Pods
    pods:
      metric:
        name: api_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  - type: Pods
    pods:
      metric:
        name: queue_length
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 60  # 1 minute
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

#### Vertical Pod Autoscaler (VPA)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: medical-ai-backend-vpa
  namespace: medical-ai-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-ai-backend
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: backend
      minAllowed:
        cpu: 200m
        memory: 1Gi
      maxAllowed:
        cpu: 8
        memory: 32Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

### Cluster Autoscaler

```yaml
# Cluster Autoscaler for AWS EKS
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  scale-down-enabled: "true"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
  scale-down-utilization-threshold: "0.5"
  max-node-provision-time: "15m"
  balance-similar-node-groups: "true"
  skip-nodes-with-local-storage: "false"
  expander: "priority"
  
  # Node group priorities
  priority: |
    [
      {
        "predefinedPriority": -50,
        "expression": "sum(ng.cpu * 4 + ng.memory) / 128",
        "expectedNodeGroup": {"minSize": 1, "maxSize": 3}
      },
      {
        "predefinedPriority": -100,
        "expression": "1.0 / max(sum(ng.memory), 1)",
        "expectedNodeGroup": {"minSize": 1, "maxSize": 2, "taints": [{"key": "medical-workload", "value": "true"}]}
      },
      {
        "predefinedPriority": 100,
        "expression": "sum(ng.memory) / -512",  # Prefer memory-intensive
        "expectedNodeGroup": {"instanceTypes": ["r5.2xlarge", "r5.4xlarge"]}
      }
    ]
```

### Load Testing for Scaling

```bash
#!/bin/bash
# Load testing script to validate scaling

# Install k6 for load testing
curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz

# Create load test configuration
cat > load-test.js << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 500 },  // Ramp up to 500
    { duration: '5m', target: 500 },  // Stay at 500 users
    { duration: '2m', target: 1000 }, // Ramp up to 1000
    { duration: '5m', target: 1000 }, // Stay at 1000 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% of requests under 2s
    http_req_failed: ['rate<0.05'],     // Error rate under 5%
  },
};

export default function () {
  const response = http.get('https://api.medical-ai.example.com/health');
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(1);
}
EOF

# Run load test
./k6 run load-test.js

# Monitor scaling metrics
kubectl get hpa -n medical-ai-prod -w
kubectl top pods -n medical-ai-prod
kubectl top nodes
```

### Performance Tuning

#### Backend Optimization

```python
# backend/config/performance.py
# Performance tuning configuration

# Uvicorn configuration for production
UVICORN_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "worker_connections": 1000,
    "max_requests": 1000,
    "max_requests_jitter": 100,
    "timeout_keep_alive": 65,
    "keepalive_timeout": 5,
    "limit_concurrency": 1000,
    "backlog": 2048,
    "loop": "asyncio",
    "http": "httptools",
    "access_log": False,
    "log_config": None,
}

# Database connection pool
DATABASE_POOL_CONFIG = {
    "pool_size": 20,
    "max_overflow": 0,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
    "echo": False,
}

# Redis configuration
REDIS_CONFIG = {
    "max_connections": 100,
    "connection_pool_kwargs": {
        "retry_on_timeout": True,
        "health_check_interval": 30,
    },
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "socket_keepalive": True,
    "socket_keepalive_options": {},
}

# Model serving optimization
MODEL_CONFIG = {
    "batch_size": 32,
    "max_sequence_length": 512,
    "precision": "float16",
    "compile_model": True,
    "optimize_for_inference": True,
    "dynamic_batching": True,
    "batching_timeout": 100,  # ms
    "max_queue_size": 1000,
    "cuda_memory_fraction": 0.8,
    "omp_num_threads": 1,
}
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# deployment/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'medical-ai-prod'
    environment: 'production'

rule_files:
  - "alert-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Kubernetes API Server
  - job_name: 'kubernetes-apiserver'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  # Kubernetes Pods
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      target_label: __address__
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2

  # Medical AI Backend
  - job_name: 'medical-ai-backend'
    kubernetes_sd_configs:
    - role: endpoints
    namespaces:
      names:
      - medical-ai-prod
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: medical-ai-backend
    - source_labels: [__meta_kubernetes_endpoint_port_name]
      action: keep
      regex: http

  # Medical AI Frontend
  - job_name: 'medical-ai-frontend'
    kubernetes_sd_configs:
    - role: endpoints
    namespaces:
      names:
      - medical-ai-prod
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: medical-ai-frontend

  # Model Serving
  - job_name: 'medical-ai-serving'
    kubernetes_sd_configs:
    - role: endpoints
    namespaces:
      names:
      - medical-ai-prod
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: medical-ai-serving
    - source_labels: [__meta_kubernetes_endpoint_port_name]
      action: keep
      regex: http

  # Database monitoring
  - job_name: 'postgres-exporter'
    static_configs:
    - targets: ['postgres-exporter:9187']

  # Redis monitoring
  - job_name: 'redis-exporter'
    static_configs:
    - targets: ['redis-exporter:9121']

  # Node monitoring
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
    - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)
    - target_label: __address__
      replacement: kubernetes.default.svc:443
    - source_labels: [__meta_kubernetes_node_name]
      regex: (.+)
      target_label: __metrics_path__
      replacement: /api/v1/nodes/${1}/proxy/metrics
```

### Alert Rules

```yaml
# deployment/monitoring/alert-rules.yml
groups:
- name: medical-ai.rules
  rules:
  # Infrastructure Alerts
  - alert: PodCrashLooping
    expr: rate(kube_pod_container_status_restarts_total[15m]) * 60 * 15 > 3
    for: 15m
    labels:
      severity: critical
      team: devops
    annotations:
      summary: "Pod {{ $labels.pod }} is crash looping"
      description: "Pod {{ $labels.pod }} has been restarting {{ $value }} times per minute for the last 15 minutes."

  - alert: PodNotReady
    expr: kube_pod_status_ready{condition="false"} == 1
    for: 5m
    labels:
      severity: warning
      team: devops
    annotations:
      summary: "Pod {{ $labels.pod }} is not ready"
      description: "Pod {{ $labels.pod }} has been in not ready state for more than 5 minutes."

  # Application Health
  - alert: BackendDown
    expr: up{job="medical-ai-backend"} == 0
    for: 1m
    labels:
      severity: critical
      team: backend
    annotations:
      summary: "Backend service is down"
      description: "Medical AI Backend has been down for more than 1 minute."

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="medical-ai-backend"}[5m])) > 2
    for: 5m
    labels:
      severity: warning
      team: backend
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds for the last 5 minutes."

  # Healthcare Compliance Alerts
  - alert: PHIExposure
    expr: increase(security_phi_exposure_total[5m]) > 0
    for: 0m
    labels:
      severity: critical
      team: security
      compliance: hipaa
    annotations:
      summary: "Potential PHI exposure detected"
      description: "Potential PHI exposure detected. Immediate investigation required."
      runbook_url: "https://wiki.medical-ai.com/runbooks/phi-exposure"

  - alert: UnauthorizedAccess
    expr: increase(security_unauthorized_access_total[5m]) > 5
    for: 2m
    labels:
      severity: critical
      team: security
      compliance: hipaa
    annotations:
      summary: "Multiple unauthorized access attempts"
      description: "{{ $value }} unauthorized access attempts in the last 5 minutes."

  # Clinical Safety Alerts
  - alert: ModelAccuracyDrift
    expr: model_accuracy_score < 0.95
    for: 10m
    labels:
      severity: warning
      team: ml
      compliance: fda
    annotations:
      summary: "Model accuracy drift detected"
      description: "Model accuracy has dropped to {{ $value }} ({{ $value | humanizePercentage }})."

  - alert: ClinicalDecisionFailed
    expr: increase(ai_clinical_decision_failures_total[5m]) > 0
    for: 1m
    labels:
      severity: critical
      team: ml
      compliance: fda
    annotations:
      summary: "Clinical decision support failure"
      description: "Clinical decision support system failed {{ $value }} times in the last 5 minutes."

  # Performance Alerts
  - alert: HighCPUUsage
    expr: (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))) * 100 > 80
    for: 5m
    labels:
      severity: warning
      team: devops
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is above 80% on {{ $labels.instance }}."

  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
      team: devops
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 85% on {{ $labels.instance }}."

  - alert: DiskSpaceLow
    expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 15
    for: 5m
    labels:
      severity: critical
      team: devops
    annotations:
      summary: "Disk space is low"
      description: "Disk space is below 15% on {{ $labels.instance }}."

  # Database Alerts
  - alert: DatabaseConnectionsHigh
    expr: pg_stat_activity_count > 80
    for: 5m
    labels:
      severity: warning
      team: database
    annotations:
      summary: "High number of database connections"
      description: "Database has {{ $value }} active connections."

  - alert: DatabaseSlowQueries
    expr: rate(pg_stat_statements_mean_time_seconds[5m]) > 1
    for: 5m
    labels:
      severity: warning
      team: database
    annotations:
      summary: "Slow database queries detected"
      description: "Average query time is {{ $value }} seconds."

  # Security Alerts
  - alert: SuspiciousTraffic
    expr: rate(http_requests_total{service="nginx-ingress",status=~"4.."}[5m]) > 100
    for: 2m
    labels:
      severity: warning
      team: security
    annotations:
      summary: "Suspicious traffic detected"
      description: "High rate of 4xx errors: {{ $value }} requests per second."

  - alert: SSL Certificate Expiry
    expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 7
    for: 1h
    labels:
      severity: warning
      team: devops
    annotations:
      summary: "SSL certificate expires soon"
      description: "SSL certificate for {{ $labels.instance }} expires in {{ $value | humanizeDuration }}."
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Medical AI Assistant - System Overview",
    "tags": ["medical-ai", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Application Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"medical-ai-backend\"}",
            "legendFormat": "Backend Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"medical-ai-backend\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"medical-ai-backend\"}[5m])",
            "legendFormat": "{{ method }} {{ status }}"
          }
        ]
      },
      {
        "title": "Active Sessions",
        "type": "graph",
        "targets": [
          {
            "expr": "active_user_sessions",
            "legendFormat": "Active Sessions"
          }
        ]
      },
      {
        "title": "Model Inference Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket{job=\"medical-ai-serving\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Monitoring Deployment

```bash
#!/bin/bash
# Deploy monitoring stack

set -euo pipefail

NAMESPACE="monitoring"

# Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace $NAMESPACE \
  --values monitoring-values.yaml \
  --set grafana.adminPassword="$GRAFANA_ADMIN_PASSWORD" \
  --set prometheus.prometheusSpec.retention=30d

# Deploy custom configuration
kubectl apply -f deployment/monitoring/prometheus.yml -n $NAMESPACE
kubectl apply -f deployment/monitoring/alert-rules.yml -n $NAMESPACE

# Wait for deployment
kubectl wait --for=condition=Ready pods --all -n $NAMESPACE --timeout=600s

echo "Monitoring stack deployed successfully!"
echo "Grafana: https://grafana.medical-ai.example.com"
echo "Prometheus: https://prometheus.medical-ai.example.com"
echo "AlertManager: https://alertmanager.medical-ai.example.com"
```

---

## Security Configuration

### Kubernetes Security Policies

```yaml
# deployment/security/hipaa-security-rules.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: medical-ai-security
  labels:
    name: medical-ai-security
    compliance: hipaa
    security-policy: strict
  annotations:
    description: "Security namespace for HIPAA-compliant Medical AI Assistant"

---
# Pod Security Standards
apiVersion: v1
kind: ConfigMap
metadata:
  name: pod-security-standards
  namespace: medical-ai-prod
data:
  enforce: |
    apiVersion: v1
    kind: Namespace
    metadata:
      name: medical-ai-prod
    spec:
      forbiddenCapabilities:
      - "*"
      runAsUser:
        rule: "MustRunAsNonRoot"
      supplementalGroups:
        rule: "MustRunAs"
        ranges:
          - min: 1
            max: 65535
      fsGroup:
        rule: "MustRunAs"
        ranges:
          - min: 1
            max: 65535
      readOnlyRootFilesystem: true

---
# Network Policies for PHI protection
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-database-access
  namespace: medical-ai-prod
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend
    ports:
    - protocol: TCP
      port: 5432
  egress:
  - to:
    - podSelector: {}  # Allow egress for database operations
    ports:
    - protocol: TCP
      port: 443  # HTTPS for external connections

---
# Network Policy for API access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-api-access
  namespace: medical-ai-prod
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: serving
    ports:
    - protocol: TCP
      port: 8081
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379

---
# RBAC for service accounts
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: medical-ai-reader
  namespace: medical-ai-prod
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["services", "endpoints"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: medical-ai-reader-binding
  namespace: medical-ai-prod
subjects:
- kind: ServiceAccount
  name: medical-ai-sa
  namespace: medical-ai-prod
roleRef:
  kind: Role
  name: medical-ai-reader
  apiGroup: rbac.authorization.k8s.io
```

### Secrets Management

```bash
#!/bin/bash
# Create and manage secrets for Medical AI

NAMESPACE="medical-ai-prod"

# Generate secure passwords
export DB_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(openssl rand -base64 32)
export ENCRYPTION_KEY=$(openssl rand -base64 32)
export JWT_SECRET=$(openssl rand -base64 32)
export GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# Create Kubernetes secrets
kubectl create secret generic medical-ai-secrets \
  --namespace=$NAMESPACE \
  --from-literal=db-password="$DB_PASSWORD" \
  --from-literal=redis-password="$REDIS_PASSWORD" \
  --from-literal=secret-key="$SECRET_KEY" \
  --from-literal=encryption-key="$ENCRYPTION_KEY" \
  --from-literal=jwt-secret="$JWT_SECRET" \
  --from-literal=grafana-admin-password="$GRAFANA_ADMIN_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

# Create TLS secret for HTTPS
kubectl create secret tls medical-ai-tls \
  --namespace=$NAMESPACE \
  --cert=./certs/medical-ai.example.com.crt \
  --key=./certs/medical-ai.example.com.key \
  --dry-run=client -o yaml | kubectl apply -f -

# Create database credentials secret
kubectl create secret generic medical-ai-db-credentials \
  --namespace=$NAMESPACE \
  --from-literal=username="medical_ai_user" \
  --from-literal=password="$DB_PASSWORD" \
  --from-literal=database="medical_ai" \
  --dry-run=client -o yaml | kubectl apply -f -

# Enable secret encryption at rest
kubectl apply -f - << EOF
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: $(echo -n "$ENCRYPTION_KEY" | base64)
    - identity: {}
EOF
```

### Security Scanning

```yaml
# Security scanning pipeline
apiVersion: batch/v1
kind: CronJob
metadata:
  name: security-scan
  namespace: medical-ai-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: security-scanner
            image: aquasec/trivy:latest
            command:
            - /bin/sh
            - -c
            - |
              # Scan container images
              trivy image --security-checks vuln,config medical-ai/backend:latest
              trivy image --security-checks vuln,config medical-ai/frontend:latest
              
              # Scan Kubernetes manifests
              trivy k8s --no-progress --format json --output security-report.json
              
              # Send results to monitoring
              curl -X POST "http://prometheus-pushgateway.monitoring.svc.cluster.local:9091/metrics/job/security-scan" \
                -d "security_vulnerabilities_total $(jq '.Results[].Vulnerabilities | length' security-report.json)"
          restartPolicy: OnFailure
```

### Compliance Monitoring

```bash
#!/bin/bash
# HIPAA compliance validation script

NAMESPACE="medical-ai-prod"

echo "Running HIPAA compliance checks..."

# Check 1: Verify encryption at rest
echo "Checking encryption at rest..."
DATABASE_ENCRYPTED=$(kubectl get secret medical-ai-secrets -n $NAMESPACE -o jsonpath='{.data.encryption-key}' | base64 -d | wc -c)
if [ "$DATABASE_ENCRYPTED" -gt 0 ]; then
  echo "✓ Database encryption key present"
else
  echo "✗ Database encryption key missing"
  exit 1
fi

# Check 2: Verify audit logging
echo "Checking audit logging..."
AUDIT_LOGS_ENABLED=$(kubectl get deployment medical-ai-backend -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="AUDIT_LOGGING")].value}')
if [ "$AUDIT_LOGS_ENABLED" = "true" ]; then
  echo "✓ Audit logging enabled"
else
  echo "✗ Audit logging not enabled"
  exit 1
fi

# Check 3: Verify RBAC configuration
echo "Checking RBAC configuration..."
SERVICE_ACCOUNT=$(kubectl get deployment medical-ai-backend -n $NAMESPACE -o jsonpath='{.spec.template.spec.serviceAccountName}')
if [ "$SERVICE_ACCOUNT" != "default" ]; then
  echo "✓ Custom service account configured"
else
  echo "✗ Using default service account"
  exit 1
fi

# Check 4: Verify network policies
echo "Checking network policies..."
NETWORK_POLICIES=$(kubectl get networkpolicies -n $NAMESPACE --no-headers | wc -l)
if [ "$NETWORK_POLICIES" -gt 0 ]; then
  echo "✓ Network policies configured"
else
  echo "✗ No network policies configured"
  exit 1
fi

# Check 5: Verify security context
echo "Checking security context..."
NON_ROOT=$(kubectl get deployment medical-ai-backend -n $NAMESPACE -o jsonpath='{.spec.template.spec.securityContext.runAsNonRoot}')
if [ "$NON_ROOT" = "true" ]; then
  echo "✓ Non-root execution configured"
else
  echo "✗ Root execution detected"
  exit 1
fi

# Check 6: Verify resource limits
echo "Checking resource limits..."
CPU_LIMITS=$(kubectl get deployment medical-ai-backend -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}')
MEMORY_LIMITS=$(kubectl get deployment medical-ai-backend -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}')
if [ -n "$CPU_LIMITS" ] && [ -n "$MEMORY_LIMITS" ]; then
  echo "✓ Resource limits configured"
else
  echo "✗ Resource limits not configured"
  exit 1
fi

echo "HIPAA compliance validation completed successfully!"
```

---

## Troubleshooting

### Common Deployment Issues

#### 1. Pod Pending Issues

**Problem**: Pod stuck in Pending state

```bash
# Diagnose the issue
kubectl describe pod <pod-name> -n medical-ai-prod

# Check resource availability
kubectl describe node <node-name>

# Check resource requests
kubectl get pods -n medical-ai-prod -o wide

# Check node taints and tolerations
kubectl get nodes -o json | jq '.items[] | {name:.metadata.name, taints:.spec.taints}'
kubectl get pods -o json | jq '.items[] | {name:.metadata.name, tolerations:.spec.tolerations}'
```

**Solution**: Adjust resource requests or add tolerations

```yaml
# Add toleration to pod
tolerations:
- key: "medical-workload"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

#### 2. Image Pull Errors

**Problem**: ErrImagePull or ImagePullBackOff

```bash
# Check image availability
kubectl describe pod <pod-name> -n medical-ai-prod | grep -A 10 Events

# Verify image exists
docker pull medical-ai/backend:latest

# Check image pull policy
kubectl get deployment <deployment-name> -n medical-ai-prod -o jsonpath='{.spec.template.spec.containers[0].imagePullPolicy}'
```

**Solution**: Update image or registry configuration

```yaml
# Update image pull policy
imagePullPolicy: IfNotPresent  # or Always, Never
```

#### 3. Service Discovery Issues

**Problem**: Service not reachable

```bash
# Check service endpoints
kubectl get endpoints <service-name> -n medical-ai-prod

# Verify service selector
kubectl get service <service-name> -n medical-ai-prod -o yaml

# Test service connectivity
kubectl run debug-pod --rm -i --tty --image=busybox -- sh
# Inside the pod:
nslookup medical-ai-backend.medical-ai-prod.svc.cluster.local
telnet medical-ai-backend 8000
```

#### 4. Database Connection Issues

**Problem**: Cannot connect to database

```bash
# Check database pod status
kubectl get pods -n medical-ai-prod -l app=postgres

# Check database credentials
kubectl get secret medical-ai-secrets -n medical-ai-prod -o jsonpath='{.data.db-password}' | base64 -d

# Test database connection
kubectl run pg-test --rm -i --tty --image=postgres:15-alpine -- \
  psql "postgresql://medical_ai_user:$DB_PASSWORD@postgres-service.medical-ai-prod.svc.cluster.local:5432/medical_ai" -c "SELECT 1;"

# Check database logs
kubectl logs -f deployment/medical-ai-postgres -n medical-ai-prod
```

#### 5. High Memory Usage

**Problem**: Out of Memory (OOM) errors

```bash
# Check pod resource usage
kubectl top pods -n medical-ai-prod

# Check for OOMKilled events
kubectl get events -n medical-ai-prod --field-selector reason=OOMKilled

# Analyze memory usage over time
kubectl describe pod <pod-name> -n medical-ai-prod | grep -A 10 Conditions

# Check node memory pressure
kubectl describe node <node-name> | grep -A 5 MemoryPressure
```

**Solution**: Increase memory limits and add VPA

```yaml
resources:
  requests:
    memory: "1Gi"
  limits:
    memory: "4Gi"  # Increased from 2Gi
```

#### 6. Model Serving GPU Issues

**Problem**: GPU not available or allocation failures

```bash
# Check GPU availability
kubectl describe nodes -l accelerator=nvidia-tesla-v100

# Check GPU usage
kubectl get pods -n medical-ai-prod --all-namespaces -o wide | grep gpu

# Verify nvidia-device-plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Check GPU device allocation
kubectl exec -it <serving-pod> -n medical-ai-prod -- nvidia-smi
```

**Solution**: Ensure GPU nodes are tainted and nvidia-device-plugin is running

```yaml
# Add GPU resource requests
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

### Health Checks and Debugging

#### Application Health Endpoints

```bash
#!/bin/bash
# Health check script

DOMAIN="medical-ai.example.com"

echo "Checking application health..."

# Frontend health
curl -f https://$DOMAIN/health || echo "Frontend health check failed"

# Backend API health
curl -f https://$DOMAIN/api/v1/health || echo "Backend health check failed"

# Database connectivity
curl -f https://$DOMAIN/api/v1/health/db || echo "Database health check failed"

# Model serving health
curl -f https://$DOMAIN/serving/health || echo "Model serving health check failed"

# Detailed health status
curl -s https://$DOMAIN/api/v1/health/detailed | jq '.'
```

#### Log Analysis

```bash
# Application logs
kubectl logs -f deployment/medical-ai-backend -n medical-ai-prod --tail=100

# Error logs only
kubectl logs deployment/medical-ai-backend -n medical-ai-prod | grep ERROR

# Specific timeframe
kubectl logs --since=1h deployment/medical-ai-backend -n medical-ai-prod

# Multiple pods
kubectl logs -l app=backend -n medical-ai-prod --tail=50

# Previous container logs
kubectl logs --previous deployment/medical-ai-backend -n medical-ai-prod

# Export logs to file
kubectl logs deployment/medical-ai-backend -n medical-ai-prod > backend-logs.txt
```

### Performance Diagnostics

```bash
# Check resource usage over time
kubectl top pods -n medical-ai-prod --watch

# Check HPA status
kubectl get hpa -n medical-ai-prod -o yaml

# Check autoscaling events
kubectl get events -n medical-ai-prod --field-selector reason=HorizontalPodAutoscaler

# Check cluster autoscaler
kubectl describe configmap cluster-autoscaler-status -n kube-system

# Check service endpoints
kubectl get endpoints -n medical-ai-prod

# Check network policies
kubectl get networkpolicies -n medical-ai-prod -o yaml

# Check ingress status
kubectl get ingress -n medical-ai-prod

# Check SSL certificate
kubectl get certificates -n medical-ai-prod
```

---

## Compliance

### HIPAA Compliance Checklist

#### Administrative Safeguards

- [x] **Security Officer Assigned**: Designated security officer for PHI protection
- [x] **Workforce Training**: Regular security awareness training
- [x] **Access Management**: Role-based access control implemented
- [x] **Incident Response**: Security incident response procedures
- [x] **Business Associate Agreements**: Agreements with third-party vendors
- [x] **Risk Assessment**: Regular risk assessment and mitigation

#### Physical Safeguards

- [x] **Facility Access Controls**: Secure data center access
- [x] **Workstation Security**: Secure workstation configurations
- [x] **Device and Media Controls**: Secure device management

#### Technical Safeguards

- [x] **Access Control**: Unique user identification and authentication
- [x] **Audit Controls**: Comprehensive audit logging system
- [x] **Integrity**: Data integrity protection measures
- [x] **Person/Entity Authentication**: Strong authentication mechanisms
- [x] **Transmission Security**: Encrypted data transmission

### FDA Compliance for Clinical Decision Support

- [x] **Accuracy Monitoring**: Real-time model accuracy tracking
- [x] **Clinical Validation**: Clinical outcome validation system
- [x] **Bias Detection**: Bias detection and mitigation algorithms
- [x] **Safety Overrides**: Clinical safety override mechanisms
- [x] **Quality Management**: ISO 13485 quality management system

### ISO 27001 Information Security Management

- [x] **Risk Assessment**: Information security risk management
- [x] **Security Policies**: Comprehensive security policies
- [x] **Access Control**: Information access control management
- [x] **Cryptography**: Cryptographic controls implementation
- [x] **Incident Management**: Security incident management procedures

---

## Quick Reference

### Useful Commands

```bash
# Deployment
kubectl apply -f deployment/kubernetes/ -n medical-ai-prod
helm upgrade --install medical-ai ./helm-chart --namespace medical-ai-prod

# Monitoring
kubectl get pods -n medical-ai-prod
kubectl top pods -n medical-ai-prod
kubectl get hpa -n medical-ai-prod
kubectl describe ingress -n medical-ai-prod

# Scaling
kubectl scale deployment medical-ai-backend --replicas=10 -n medical-ai-prod
kubectl patch hpa medical-ai-backend-hpa -n medical-ai-prod -p '{"spec":{"maxReplicas":100}}'

# Troubleshooting
kubectl logs -f deployment/medical-ai-backend -n medical-ai-prod
kubectl describe pod <pod-name> -n medical-ai-prod
kubectl exec -it <pod-name> -n medical-ai-prod -- /bin/sh

# Security
kubectl auth can-i get secrets -n medical-ai-prod
kubectl get networkpolicies -n medical-ai-prod
kubectl get certificates -n medical-ai-prod
```

### Emergency Contacts

- **Security Team**: security@medical-ai.example.com
- **Compliance Team**: compliance@medical-ai.example.com
- **Clinical Safety**: clinical@medical-ai.example.com
- **DevOps**: devops@medical-ai.example.com
- **On-Call**: +1-555-MEDICAL (24/7)

### Emergency Rollback

```bash
# Rollback deployment
kubectl rollout undo deployment/medical-ai-backend -n medical-ai-prod

# Rollback to specific revision
kubectl rollout undo deployment/medical-ai-backend --to-revision=2 -n medical-ai-prod

# Check rollout status
kubectl rollout status deployment/medical-ai-backend -n medical-ai-prod

# Scale down to prevent issues
kubectl scale deployment medical-ai-backend --replicas=1 -n medical-ai-prod
```

---

**🎯 This deployment guide provides comprehensive instructions for deploying the Medical AI Assistant across multiple environments while maintaining HIPAA, FDA, and ISO 27001 compliance. All configurations prioritize security, scalability, and healthcare-specific requirements.**

**📞 For support and questions, contact the DevOps team or refer to the troubleshooting section above.**
