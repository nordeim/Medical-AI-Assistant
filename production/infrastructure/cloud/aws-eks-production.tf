# AWS EKS Production Infrastructure for Medical AI Assistant
# Healthcare-compliant multi-AZ deployment with HIPAA, FDA, and ISO 27001 compliance

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
    key    = "production/eks/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

# Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      Application = "medical-ai-assistant"
      Compliance  = "hipaa,fda,iso27001"
      ManagedBy   = "terraform"
      CostCenter  = "healthcare-ml"
      Team        = "medical-ai-devops"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  token                  = module.eks.cluster_auth_token
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    token                  = module.eks.cluster_auth_token
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  }
}

# Variables
variable "aws_region" {
  description = "AWS region for Medical AI production deployment"
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
  default     = "medical-ai-production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for deployment"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "node_groups" {
  description = "EKS node groups configuration"
  type = map(object({
    desired_capacity = number
    max_capacity     = number
    min_capacity     = number
    instance_types   = list(string)
    capacity_type    = string
    labels           = map(string)
  }))
  default = {
    general = {
      desired_capacity = 6
      max_capacity     = 20
      min_capacity     = 3
      instance_types   = ["c5.2xlarge", "c5.4xlarge"]
      capacity_type    = "SPOT"
      labels = {
        "node-type" = "general"
        "workload"  = "backend,frontend"
      }
    }
    gpu = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 1
      instance_types   = ["p3.2xlarge", "p3.8xlarge"]
      capacity_type    = "ON_DEMAND"
      labels = {
        "node-type"   = "gpu"
        "accelerator" = "nvidia-tesla-v100"
        "workload"    = "model-serving"
      }
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    database = {
      desired_capacity = 3
      max_capacity     = 6
      min_capacity     = 2
      instance_types   = ["r5.xlarge", "r5.2xlarge"]
      capacity_type    = "ON_DEMAND"
      labels = {
        "node-type" = "database"
        "workload"  = "postgresql,redis"
      }
      taints = [
        {
          key    = "database-node"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

variable "database_password" {
  description = "Database password for Medical AI"
  type        = string
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password for Medical AI"
  type        = string
  sensitive   = true
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = [for az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, index(var.availability_zones, az) + 10)]
  public_subnets  = [for az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, index(var.availability_zones, az) + 1)]
  
  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Create additional private subnets for database
  database_subnets = [for az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, index(var.availability_zones, az) + 20)]

  # Public subnet tags for load balancers
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  # Private subnet tags for internal load balancers
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  # Database subnet tags
  database_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name        = "${var.cluster_name}-eks-cluster"
  description = "Security group for EKS cluster"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS access"
  }

  ingress {
    from_port = 80
    to_port   = 80
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP redirect to HTTPS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "${var.cluster_name}-eks-cluster"
  }
}

resource "aws_security_group" "eks_nodes" {
  name        = "${var.cluster_name}-eks-nodes"
  description = "Security group for EKS nodes"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
    description     = "Node port range"
  }

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
    description = "Node to node communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "${var.cluster_name}-eks-nodes"
  }
}

# KMS Keys for Encryption
resource "aws_kms_key" "medical_ai" {
  description             = "KMS key for Medical AI production encryption"
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
        Sid    = "Allow EKS to use the key"
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
    Name        = "${var.cluster_name}-kms-key"
    Environment = var.environment
  }
}

resource "aws_kms_alias" "medical_ai" {
  name          = "alias/${var.cluster_name}-medical-ai"
  target_key_id = aws_kms_key.medical_ai.key_id
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]

  # Cluster security group
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                   = "Nodes on ephemeral ports"
      protocol                      = "tcp"
      from_port                     = 1025
      to_port                       = 65535
      type                          = "ingress"
      source_node_security_group    = true
    }
  }

  # Node security group
  node_security_group_additional_rules = {
    ingress_cluster_all = {
      description                   = "All cluster communication"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
    ingress_cluster_all_tcp = {
      description                   = "Cluster API to node"
      protocol                      = "tcp"
      from_port                     = 1025
      to_port                       = 65535
      type                          = "ingress"
      source_cluster_security_group = true
    }
    egress_all = {
      description = "Node all egress"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "egress"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  # EKS managed node groups
  eks_managed_node_groups = {
    for name, config in var.node_groups : name => {
      name           = name
      instance_types = config.instance_types
      capacity_type  = config.capacity_type

      min_size     = config.min_capacity
      max_size     = config.max_capacity
      desired_size = config.desired_capacity

      labels = merge(config.labels, {
        Environment = var.environment
        Compliance  = "hipaa,fda,iso27001"
      })

      taints = config.taints != null ? config.taints : []

      update_config = {
        max_unavailable_percentage = 10
      }

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = name == "database" ? 100 : 50
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 150
            encrypted             = true
            kms_key_id           = aws_kms_key.medical_ai.arn
            delete_on_termination = true
          }
        }
      }

      tags = {
        Name        = "${var.cluster_name}-${name}"
        Environment = var.environment
        NodeGroup   = name
      }
    }
  }

  # Enable cluster access entry
  enable_cluster_access_entry = true

  tags = {
    Environment = var.environment
    Application = "medical-ai-assistant"
  }
}

# RDS PostgreSQL Multi-AZ Cluster
module "rds" {
  source  = "terraform-aws-modules/rds-aurora/aws"
  version = "~> 9.0"

  name           = "${var.cluster_name}-database"
  engine         = "aurora-postgresql"
  engine_version = "15.4"
  database_name  = "medical_ai"
  master_username = "postgres"

  master_password = var.database_password
  
  vpc_id                  = module.vpc.vpc_id
  db_subnet_group_name   = module.db_subnet_group.db_subnet_group_name
  instance_class          = "db.r6g.xlarge"
  instances = {
    1 = {}
    2 = {
      instance_class = "db.r6g.xlarge"
    }
  }

  # High availability
  multi_az               = true
  publicly_accessible    = false
  storage_encrypted      = true
  kms_key_id            = aws_kms_key.medical_ai.arn

  # Performance optimization
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn         = aws_iam_role.rds_enhanced_monitoring.arn

  # Backup retention (35 days for HIPAA)
  backup_retention_period = 35
  preferred_backup_window = "03:00-04:00"
  preferred_maintenance_window = "Sun:04:00-Sun:05:00"

  # Network and security
  security_group_rules = {
    inbound_db = {
      type                     = "ingress"
      from_port                = 5432
      to_port                  = 5432
      protocol                 = "tcp"
      source_security_group_id = aws_security_group.eks_nodes.id
    }
  }

  tags = {
    Name        = "${var.cluster_name}-rds"
    Environment = var.environment
  }
}

# Database Subnet Group
module "db_subnet_group" {
  source = "terraform-aws-modules/rds/aws"

  name           = "${var.cluster_name}-db-subnet-group"
  subnet_ids     = module.vpc.database_subnets
  subnet_group_type = "database"

  tags = {
    Name        = "${var.cluster_name}-db-subnet-group"
    Environment = var.environment
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "medical_ai" {
  name       = "${var.cluster_name}-redis-subnet-group"
  subnet_ids = module.vpc.database_subnets

  tags = {
    Name        = "${var.cluster_name}-redis-subnet-group"
    Environment = var.environment
  }
}

resource "aws_elasticache_replication_group" "medical_ai" {
  replication_group_id         = "${var.cluster_name}-redis"
  description                  = "Redis cluster for Medical AI production"

  node_type                   = "cache.r6g.large"
  port                        = 6379
  parameter_group_name        = "default.redis7"
  num_cache_clusters          = 3
  automatic_failover_enabled  = true
  multi_az_enabled           = true

  subnet_group_name  = aws_elasticache_subnet_group.medical_ai.name
  security_group_ids = [aws_security_group.eks_nodes.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_password

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format      = "text"
    log_type        = "slow-log"
  }

  tags = {
    Name        = "${var.cluster_name}-redis"
    Environment = var.environment
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${var.cluster_name}/application"
  retention_in_days = 2555  # 7 years for HIPAA compliance

  kms_key_id = aws_kms_key.medical_ai.arn

  tags = {
    Name        = "${var.cluster_name}-app-logs"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "audit" {
  name              = "/aws/eks/${var.cluster_name}/audit"
  retention_in_days = 2555  # 7 years for HIPAA compliance

  kms_key_id = aws_kms_key.medical_ai.arn

  tags = {
    Name        = "${var.cluster_name}-audit-logs"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/eks/${var.cluster_name}/redis-slow"
  retention_in_days = 2555

  kms_key_id = aws_kms_key.medical_ai.arn

  tags = {
    Name        = "${var.cluster_name}-redis-slow"
    Environment = var.environment
  }
}

# Application Load Balancer for Ingress
resource "aws_lb" "medical_ai" {
  name               = "${var.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.eks_cluster.id]
  subnets           = module.vpc.public_subnets

  enable_deletion_protection = true
  enable_http2              = true
  enable_cross_zone_load_balancing = true
  enable_waf_fail_open      = false

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    prefix  = "${var.cluster_name}/access-logs"
    enabled = true
  }

  tags = {
    Name        = "${var.cluster_name}-alb"
    Environment = var.environment
  }
}

# ALB Target Groups
resource "aws_lb_target_group" "backend" {
  name     = "${var.cluster_name}-backend-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = module.vpc.vpc_id
  target_type = "instance"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = {
    Name        = "${var.cluster_name}-backend-tg"
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "frontend" {
  name     = "${var.cluster_name}-frontend-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = module.vpc.vpc_id
  target_type = "instance"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = {
    Name        = "${var.cluster_name}-frontend-tg"
    Environment = var.environment
  }
}

# ALB Listener
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.medical_ai.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.medical_ai.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }

  tags = {
    Name        = "${var.cluster_name}-https-listener"
    Environment = var.environment
  }
}

# SSL Certificate
resource "aws_acm_certificate" "medical_ai" {
  domain_name       = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = [
    "*.${var.domain_name}",
    "api.${var.domain_name}",
    "portal.${var.domain_name}",
  ]

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.cluster_name}-cert"
    Environment = var.environment
  }
}

# S3 Buckets for Backups and Model Storage
resource "aws_s3_bucket" "backups" {
  bucket = "${var.cluster_name}-backups"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.medical_ai.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }

  lifecycle_rule {
    id     = "glacier-transition"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 2555  # 7 years for HIPAA
    }
  }

  tags = {
    Name        = "${var.cluster_name}-backups"
    Environment = var.environment
    Compliance  = "HIPAA"
  }
}

resource "aws_s3_bucket" "model_storage" {
  bucket = "${var.cluster_name}-models"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.medical_ai.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }

  lifecycle_rule {
    id     = "ia-transition"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = 2555  # 7 years for HIPAA
    }
  }

  tags = {
    Name        = "${var.cluster_name}-models"
    Environment = var.environment
    Compliance  = "HIPAA"
  }
}

resource "aws_s3_bucket" "alb_logs" {
  bucket = "${var.cluster_name}-alb-logs"

  lifecycle_rule {
    id     = "delete-after-7-days"
    status = "Enabled"

    expiration {
      days = 7
    }
  }

  tags = {
    Name        = "${var.cluster_name}-alb-logs"
    Environment = var.environment
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "eks_cluster" {
  name = "${var.cluster_name}-eks-cluster-role"

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

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  ]

  tags = {
    Name        = "${var.cluster_name}-eks-cluster-role"
    Environment = var.environment
  }
}

resource "aws_iam_role" "eks_nodes" {
  name = "${var.cluster_name}-eks-nodes-role"

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

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
    "arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess"
  ]

  tags = {
    Name        = "${var.cluster_name}-eks-nodes-role"
    Environment = var.environment
  }
}

resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${var.cluster_name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
  ]

  tags = {
    Name        = "${var.cluster_name}-rds-monitoring-role"
    Environment = var.environment
  }
}

# Data Sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = aws_security_group.eks_cluster.id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_name" {
  description = "The name/id of the EKS cluster"
  value       = var.cluster_name
}

output "rds_cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = module.rds.cluster_endpoint
}

output "rds_cluster_reader_endpoint" {
  description = "RDS cluster reader endpoint"
  value       = module.rds.cluster_reader_endpoint
}

output "redis_replication_group_endpoint" {
  description = "Redis replication group endpoint"
  value       = aws_elasticache_replication_group.medical_ai.primary_endpoint_address
}

output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.medical_ai.dns_name
}

output "backup_bucket_name" {
  description = "Name of the backup S3 bucket"
  value       = aws_s3_bucket.backups.bucket
}

output "model_storage_bucket_name" {
  description = "Name of the model storage S3 bucket"
  value       = aws_s3_bucket.model_storage.bucket
}

output "kms_key_arn" {
  description = "ARN of the KMS encryption key"
  value       = aws_kms_key.medical_ai.arn
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnet_ids" {
  description = "List of IDs of database subnets"
  value       = module.vpc.database_subnets
}
