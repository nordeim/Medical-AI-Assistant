# GCP GKE Production Deployment Configuration for Medical AI Assistant
# Multi-zone deployment with healthcare compliance
# Terraform configuration for GCP infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
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
  
  backend "gcs" {
    bucket = "medical-ai-terraform-state"
    prefix = "production"
  }
}

# Google Cloud Provider Configuration
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  
  default_labels = {
    environment         = var.environment
    project             = "medical-ai"
    managed-by          = "terraform"
    compliance          = "HIPAA"
    data-classification = "PHI"
    cost-center         = "medical-ai"
  }
}

# Kubernetes provider
provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.default.default_access_token[0]
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.default.default_access_token[0]
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth.0.cluster_ca_certificate)
  }
}

# Data sources
data "google_client_config" "default" {}

data "google_container_cluster" "main" {
  name     = google_container_cluster.main.name
  location = var.gcp_region
}

# Variables
variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "medical-ai-prod"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28.1-gke.1000"
}

variable "network" {
  description = "VPC network name"
  type        = string
  default     = "medical-ai-network"
}

variable "subnetwork" {
  description = "VPC subnetwork name"
  type        = string
  default     = "medical-ai-subnet"
}

variable "ip_range_pods" {
  description = "Pods IP range name"
  type        = string
  default     = "medical-ai-pods"
}

variable "ip_range_services" {
  description = "Services IP range name"
  type        = string
  default     = "medical-ai-services"
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = var.network
  auto_create_subnetworks = false
  mtu                     = 1460
  
  routing_mode = "REGIONAL"
}

# Subnet for the cluster
resource "google_compute_subnetwork" "main" {
  name          = var.subnetwork
  ip_cidr_range = "10.0.0.0/24"
  region        = var.gcp_region
  network       = google_compute_network.main.id
  
  secondary_ip_range {
    range_name    = var.ip_range_pods
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = var.ip_range_services
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Firewall rules for cluster security
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.main.name
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "icmp"
  }
  
  source_ranges = [
    "10.0.0.0/24",
    "10.1.0.0/16",
    "10.2.0.0/16"
  ]
  
  target_tags = ["medical-ai-cluster"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.cluster_name}-allow-ssh"
  network = google_compute_network.main.name
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["medical-ai-cluster"]
}

# Cloud NAT for outbound internet access
resource "google_compute_router" "main" {
  name    = "${var.cluster_name}-router"
  region  = var.gcp_region
  network = google_compute_network.main.id
}

resource "google_compute_router_nat" "main" {
  name                               = "${var.cluster_name}-nat"
  router                            = google_compute_router.main.name
  region                            = var.gcp_region
  nat_ip_allocate_option           = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.gcp_region
  
  remove_default_node_pool = true
  initial_node_count       = 1
  network                  = google_compute_network.main.name
  subnetwork               = google_compute_subnetwork.main.name
  logging_service         = "logging.googleapis.com/kubernetes"
  monitoring_service      = "monitoring.googleapis.com/kubernetes"
  
  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }
  
  # Enable network policy
  network_policy {
    enabled = true
  }
  
  # Addons configuration
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    vertical_pod_autoscaling {
      enabled = true
      enforcement_history_lookback_duration = "86400s"
    }
    
    network_policy_config {
      disabled = false
    }
    
    dns_cache_config {
      enabled = true
    }
    
    gcp_filestore_csi_driver_config {
      enabled = true
    }
    
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    
    kubernetes_dashboard {
      disabled = true
    }
    
    network_policy_config {
      disabled = false
    }
  }
  
  # Security configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_global_access_config {
      enabled = true
    }
    master_ipv4_cidr_block = "172.16.0.0/28"
  }
  
  # Master authentication
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Security settings
  enable_legacy_abac = false
  
  # Resource labels
  resource_labels = {
    environment = var.environment
    project     = "medical-ai"
    compliance  = "HIPAA"
  }
  
  # Enableshielded nodes
  enable_shielded_nodes = true
  
  # Default max pods constraint
  default_max_pods_per_node = 110
  
  # Maintenance policy
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
  
  # Node locations for multi-zone deployment
  node_locations = [
    "${var.gcp_zone}",
    "${var.gcp_zone2}",
    "${var.gcp_zone3}"
  ]
}

# Default node pool (will be removed after creating specific node pools)
resource "google_container_node_pool" "preemptible_nodes" {
  name       = "preemptible-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 1
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 1
    max_node_count = 1
  }
  
  node_config {
    preemptible  = true
    machine_type = "e2-medium"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

# General purpose node pool
resource "google_container_node_pool" "general" {
  name       = "general-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 3
    max_node_count = 20
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
  
  node_config {
    spot         = false
    machine_type = "e2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
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
      workload-type = "general"
    }
    
    tags = ["medical-ai-cluster", "general-pool"]
  }
}

# Backend optimized node pool
resource "google_container_node_pool" "backend" {
  name       = "backend-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
  
  node_config {
    spot         = false
    machine_type = "e2-custom-8-32768"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
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
      workload-type = "backend"
    }
    
    taint {
      key    = "medical-ai-dedicated"
      value  = "backend"
      effect = "NO_SCHEDULE"
    }
    
    tags = ["medical-ai-cluster", "backend-pool"]
  }
}

# Frontend optimized node pool
resource "google_container_node_pool" "frontend" {
  name       = "frontend-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
  
  node_config {
    spot         = false
    machine_type = "e2-standard-2"
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
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
      workload-type = "frontend"
    }
    
    taint {
      key    = "medical-ai-dedicated"
      value  = "frontend"
      effect = "NO_SCHEDULE"
    }
    
    tags = ["medical-ai-cluster", "frontend-pool"]
  }
}

# GPU node pool
resource "google_container_node_pool" "gpu" {
  name       = "gpu-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 1
    max_node_count = 8
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
  
  node_config {
    spot         = false
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
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
      accelerator          = "nvidia-tesla-v100"
      workload-type        = "gpu"
    }
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
    
    tags = ["medical-ai-cluster", "gpu-pool"]
    
    # Guest accelerators
    guest_accelerator {
      type  = "nvidia-tesla-v100"
      count = 1
    }
  }
}

# Cloud SQL for PostgreSQL
resource "google_sql_database_instance" "main" {
  name             = "${var.cluster_name}-db"
  database_version = "POSTGRES_15"
  region           = var.gcp_region
  
  settings {
    tier = "db-custom-8-32768"
    
    disk_type = "PD_SSD"
    disk_size = 500
    disk_autoresize = true
    disk_autoresize_limit = 1000
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.gcp_region
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
      transaction_log_retention_days = 7
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
      require_ssl     = true
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    database_flags {
      name  = "log_statement"
      value = "all"
    }
    
    user_labels = {
      environment = var.environment
      project     = "medical-ai"
      compliance  = "HIPAA"
    }
    
    maintenance_window {
      hour          = 3
      day           = 7
      update_track  = "stable"
    }
    
    backup_configuration {
      enabled             = true
      start_time          = "03:00"
      location            = var.gcp_region
      point_in_time_recovery_enabled = true
    }
    
    insights_config {
      query_insights_enabled = true
      query_string_length     = 1024
      recordApplicationTags  = true
      recordClientAddress    = true
    }
  }
  
  deletion_protection = true
  
  labels = {
    environment = var.environment
    project     = "medical-ai"
  }
}

resource "google_sql_database" "database" {
  name     = "medical_ai"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "users" {
  name     = var.db_username
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Cloud Memorystore for Redis
resource "google_compute_network" "redis_network" {
  name = "${var.cluster_name}-redis-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "redis_subnetwork" {
  name          = "${var.cluster_name}-redis-subnet"
  ip_cidr_range = "10.3.0.0/24"
  region        = var.gcp_region
  network       = google_compute_network.main.id
}

resource "google_redis_instance" "main" {
  name                      = "${var.cluster_name}-redis"
  memory_size_gb           = 4
  region                   = var.gcp_region
  redis_version            = "REDIS_7_0"
  location_id              = var.gcp_zone
  alternative_location_id  = var.gcp_zone2
  
  authorized_network = google_compute_network.main.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_config = {
    maxmemory-policy = "allkeys-lru"
  }
  
  labels = {
    environment = var.environment
    project     = "medical-ai"
  }
  
  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "TWENTY_FOUR_HOURS"
    rdb_snapshot_start_time = "03:00"
  }
}

# Workload Identity for secure service account access
resource "google_service_account" "ksa" {
  account_id   = "medical-ai-ksa"
  display_name = "Medical AI Kubernetes Service Account"
  description  = "Service account for Medical AI workloads"
}

resource "google_container_cluster" "main_workload_identity" {
  name = var.cluster_name
  
  # Workload Identity binding will be added via kubectl or separate resource
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "main" {
  location      = var.gcp_region
  repository_id = "medical-ai-images"
  description   = "Container images for Medical AI Assistant"
  format        = "DOCKER"
  
  docker_config {
    immutable_tags = true
  }
}

# Outputs
output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.main.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.main.master_auth.0.cluster_ca_certificate
  sensitive   = true
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.main.host
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository"
  value       = google_artifact_registry_repository.main.id
}

# Additional variables for GCP deployment
variable "gcp_zone2" {
  description = "Secondary GCP zone for multi-zone deployment"
  type        = string
  default     = "us-central1-b"
}

variable "gcp_zone3" {
  description = "Tertiary GCP zone for multi-zone deployment"
  type        = string
  default     = "us-central1-c"
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}