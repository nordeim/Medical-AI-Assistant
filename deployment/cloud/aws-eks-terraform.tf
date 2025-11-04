# AWS EKS Production Deployment Configuration for Medical AI Assistant
# Multi-AZ deployment with healthcare compliance
# Terraform configuration for AWS infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = "medical-ai-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment         = var.environment
      Project             = "medical-ai"
      ManagedBy          = "terraform"
      Compliance         = "HIPAA"
      DataClassification = "PHI"
      CostCenter         = "medical-ai"
      Owner              = "devops@medical-ai.example.com"
    }
  }
}

# Kubernetes provider
provider "kubernetes" {
  host                   = aws_eks_cluster.main.endpoint
  token                  = data.aws_eks_cluster_auth.main.token
  cluster_ca_certificate = base64decode(aws_eks_cluster.main.certificate_authority[0].data)
}

provider "helm" {
  kubernetes {
    host                   = aws_eks_cluster.main.endpoint
    token                  = data.aws_eks_cluster_auth.main.token
    cluster_ca_certificate = base64decode(aws_eks_cluster.main.certificate_authority[0].data)
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "medical-ai-prod"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "instance_types" {
  description = "EC2 instance types for node groups"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge", "c5.2xlarge"]
}

variable "node_groups" {
  description = "Node group configurations"
  type = map(object({
    desired_capacity = number
    max_capacity     = number
    min_capacity     = number
    instance_types   = list(string)
    capacity_type    = string
    labels           = map(string)
    taints           = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      desired_capacity = 6
      max_capacity     = 20
      min_capacity     = 3
      instance_types   = ["m5.xlarge", "m5.2xlarge"]
      capacity_type    = "ON_DEMAND"
      labels = {
        workload-type = "general"
      }
      taints = []
    }
    backend = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2
      instance_types   = ["c5.2xlarge", "c5.4xlarge"]
      capacity_type    = "ON_DEMAND"
      labels = {
        workload-type = "backend"
      }
      taints = [{
        key    = "medical-ai-dedicated"
        value  = "backend"
        effect = "NO_SCHEDULE"
      }]
    }
    frontend = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2
      instance_types   = ["m5.large", "m5.xlarge"]
      capacity_type    = "ON_DEMAND"
      labels = {
        workload-type = "frontend"
      }
      taints = [{
        key    = "medical-ai-dedicated"
        value  = "frontend"
        effect = "NO_SCHEDULE"
      }]
    }
    gpu = {
      desired_capacity = 2
      max_capacity     = 8
      min_capacity     = 1
      instance_types   = ["p3.2xlarge", "p3.8xlarge"]
      capacity_type    = "ON_DEMAND"
      labels = {
        accelerator          = "nvidia-tesla-v100"
        workload-type        = "gpu"
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# VPC and Networking
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # VPC Flow Logs for security monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.cluster_version
  
  vpc_config {
    subnet_ids = concat(module.vpc.private_subnets, module.vpc.public_subnets)
    
    # Security settings for HIPAA compliance
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
    
    # Cluster security group
    security_group_ids = [aws_security_group.eks_cluster.id]
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

# EKS Cluster IAM Role
resource "aws_iam_role" "eks_cluster" {
  name = "${var.cluster_name}-cluster"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSServicePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.eks_cluster.name
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS encryption key for ${var.cluster_name}"
  deletion_window_in_days = 30
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Purpose     = "EKS encryption"
  }
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${var.cluster_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# EKS Node Groups
resource "aws_eks_node_group" "main" {
  for_each = var.node_groups
  
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = each.key
  node_role       = aws_iam_role.eks_node_group[each.key].arn
  subnet_ids      = module.vpc.private_subnets
  instance_types  = each.value.instance_types
  
  capacity_type  = each.value.capacity_type
  ami_type        = "AL2_x86_64"
  disk_size       = 50
  
  scaling_config {
    desired_size = each.value.desired_capacity
    max_size     = each.value.max_capacity
    min_size     = each.value.min_capacity
  }
  
  # Taints
  taint {
    key    = each.value.taints[0].key
    value  = each.value.taints[0].value
    effect = each.value.taints[0].effect
  }
  
  # Labels
  labels = each.value.labels
  
  # Security settings
  remote_access {
    ec2_ssh_key = aws_key_pair.eks.key_name
    source_security_groups = [aws_security_group.eks_nodes.id]
  }
  
  # Update configuration
  update_config {
    max_unavailable_percentage = 25
  }
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    NodeGroup   = each.key
  }
}

# EKS Node Group IAM Roles
resource "aws_iam_role" "eks_node_group" {
  for_each = var.node_groups
  
  name = "${var.cluster_name}-${each.key}-node"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_node_group" {
  for_each = var.node_groups
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group[each.key].name
}

resource "aws_iam_role_policy_attachment" "eks_node_group_cni" {
  for_each = var.node_groups
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group[each.key].name
}

resource "aws_iam_role_policy_attachment" "eks_node_group_ecr" {
  for_each = var.node_groups
  
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group[each.key].name
}

# EKS Cluster Security Group
resource "aws_security_group" "eks_cluster" {
  name_prefix = "${var.cluster_name}-cluster-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = concat(
      module.vpc.private_subnets_cidr_blocks,
      module.vpc.public_subnets_cidr_blocks
    )
  }
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

# EKS Node Security Group
resource "aws_security_group" "eks_nodes" {
  name_prefix = "${var.cluster_name}-nodes-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port = 1025
    to_port   = 65535
    protocol  = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

# EC2 Key Pair for SSH access
resource "aws_key_pair" "eks" {
  key_name   = "${var.cluster_name}-key"
  public_key = var.ssh_public_key
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

# RDS PostgreSQL for production database
resource "aws_db_instance" "main" {
  identifier = "${var.cluster_name}-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 500
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn
  
  db_name  = "medical_ai"
  username = var.db_username
  password = var.db_password
  port     = 5432
  
  # Multi-AZ deployment for high availability
  multi_az               = true
  publicly_accessible    = false
  backup_retention_period = 30
  backup_window         = "03:00-04:00"
  maintenance_window    = "sun:04:00-sun:05:00"
  
  # Performance Insights for monitoring
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn
  
  # Security
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  # Final snapshot
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-final-snapshot"
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.cluster_name}-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

# KMS Key for RDS encryption
resource "aws_kms_key" "rds" {
  description             = "RDS encryption key for ${var.cluster_name}"
  deletion_window_in_days = 30
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Purpose     = "RDS encryption"
  }
}

# Redis ElastiCache
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.cluster_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = "${var.cluster_name}-redis"
  description                  = "Redis cluster for Medical AI"
  
  node_type                    = "cache.r6g.xlarge"
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  num_cache_clusters           = 3
  
  engine_version               = "7.0"
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  # Encryption in transit and at rest
  transit_encryption_enabled   = true
  auth_token                   = var.redis_password
  at_rest_encryption_enabled   = true
  kms_key_id                  = aws_kms_key.redis.arn
  
  # Backup configuration
  snapshot_retention_limit     = 7
  snapshot_window             = "03:00-05:00"
  
  # Maintenance
  maintenance_window          = "sun:05:00-sun:06:00"
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Compliance  = "HIPAA"
  }
}

# KMS Key for Redis encryption
resource "aws_kms_key" "redis" {
  description             = "Redis encryption key for ${var.cluster_name}"
  deletion_window_in_days = 30
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
    Purpose     = "Redis encryption"
  }
}

# Load Balancer Controller
module "eks_blueprint_addon" {
  source = "aws-ia/eks-blueprints-addons/aws"
  version = "~> 1.0"
  
  cluster_name      = aws_eks_cluster.main.name
  cluster_endpoint  = aws_eks_cluster.main.endpoint
  cluster_version   = var.cluster_version
  oidc_provider_arn = aws_eks_cluster.main.oidc_provider_arn
  
  enable_aws_load_balancer_controller = true
  enable_aws_ebs_csi_driver          = true
  enable_aws_efs_csi_driver          = true
  enable_aws_fsx_ovfs_csi_driver     = true
  enable_aws_cloudwatch_observability_addon = true
  enable_metrics_server              = true
  enable_cluster_autoscaler          = true
  enable_external_secrets_operator   = true
  enable_cert_manager                = true
  
  tags = {
    Environment = var.environment
    Project     = "medical-ai"
  }
}

# EKS Auth data source
data "aws_eks_cluster_auth" "main" {
  name = aws_eks_cluster.main.name
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = aws_security_group.eks_cluster.id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.eks_cluster.name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache replication group endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}