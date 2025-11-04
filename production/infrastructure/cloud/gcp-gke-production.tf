# Google Cloud GKE Production Infrastructure for Medical AI Assistant
# Healthcare-compliant multi-zone deployment with HIPAA, FDA, and ISO 27001 compliance

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
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "gcs" {
    bucket = "medical-ai-terraform-state"
    prefix = "production/gke"
  }
}

# Provider Configuration
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.default_access_token[0]
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.default_access_token[0]
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
  }
}

# Random string for unique naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Variables
variable "gcp_project_id" {
  description = "GCP project ID for Medical AI production deployment"
  type        = string
  default     = "medical-ai-production"
}

variable "gcp_region" {
  description = "GCP region for Medical AI production deployment"
  type        = string
  default     = "us-central1"
}

variable "gcp_zones" {
  description = "GCP zones for deployment"
  type        = list(string)
  default     = ["us-central1-a", "us-central1-b", "us-central1-c"]
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "medical-ai-production"
}

variable "network_name" {
  description = "VPC network name"
  type        = string
  default     = "medical-ai-network"
}

variable "subnetwork_name" {
  description = "Subnet name"
  type        = string
  default     = "medical-ai-subnet"
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

# Data Sources
data "google_client_config" "default" {
  provider = google
}

data "google_container_registry_image" "gke_addons" {
  location = var.gcp_region
  name     = "gke-extras/k8s-addons"
}

# Enable Required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "sql.googleapis.com",
    "redis.googleapis.com",
    "compute.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "pubsub.googleapis.com",
    "iam.googleapis.com",
    "cloudkms.googleapis.com",
    "artifactregistry.googleapis.com",
    "servicemanagement.googleapis.com",
    "file.googleapis.com",
    "binaryauthorization.googleapis.com"
  ])

  project = var.gcp_project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy        = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = var.network_name
  auto_create_subnetworks = false
  mtu                     = 1460
  routing_mode           = "REGIONAL"

  depends_on = [
    google_project_service.required_apis["compute.googleapis.com"]
  ]

  tags = ["medical-ai", "production"]
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = var.subnetwork_name
  ip_cidr_range = "10.0.0.0/24"
  region        = var.gcp_region
  network       = google_compute_network.vpc.id

  # Secondary ranges for services
  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "172.16.0.0/16"
  }

  # Secondary ranges for pods
  secondary_ip_range {
    range_name    = "pods-range"
    ip_cidr_range = "10.1.0.0/16"
  }

  # VPC Flow logs for network monitoring
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }

  # Private Google Access
  private_ip_google_access = true

  depends_on = [
    google_project_service.required_apis["compute.googleapis.com"]
  ]
}

# Cloud NAT for private clusters
resource "google_compute_router" "router" {
  name    = "${var.cluster_name}-router"
  region  = var.gcp_region
  network = google_compute_network.vpc.id

  bgp {
    asn = 64514
  }
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.cluster_name}-nat"
  router                            = google_compute_router.router.name
  region                            = var.gcp_region
  nat_ip_allocate_option           = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }

  depends_on = [
    google_project_service.required_apis["compute.googleapis.com"]
  ]
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.vpc.name

  allow {
    protocol = "all"
  }

  source_ranges = ["10.0.0.0/16", "172.16.0.0/16", "10.1.0.0/16"]
  target_tags   = ["medical-ai", "gke-node"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.cluster_name}-allow-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]  # Google Cloud Shell IP ranges
  target_tags   = ["bastion-host"]
}

resource "google_compute_firewall" "allow_health_check" {
  name    = "${var.cluster_name}-allow-health-check"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = [
    "130.211.0.0/22",  # Google Cloud Load Balancer
    "35.191.0.0/16"    # Google Cloud Health Checkers
  ]
  target_tags = ["medical-ai", "gke-node"]
}

# Cloud KMS Key Ring and Key for Encryption
resource "google_kms_key_ring" "medical_ai" {
  name     = "${var.cluster_name}-keyring"
  location = var.gcp_region
}

resource "google_kms_crypto_key" "medical_ai" {
  name     = "medical-ai-encryption-key"
  key_ring = google_kms_key_ring.medical_ai.id

  rotation_period = "7776000s"  # 90 days

  lifecycle {
    prevent_destroy = true
  }
}

# Artifact Registry Repository
resource "google_artifact_registry_repository" "medical_ai" {
  location      = var.gcp_region
  repository_id = "${var.cluster_name}-repo"
  description   = "Container images for Medical AI production"

  format = "DOCKER"

  docker_config {
    immutable_tags = true
  }

  depends_on = [
    google_project_service.required_apis["artifactregistry.googleapis.com"]
  ]
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.gcp_zones[0]  # Single zone for now, can be expanded to multi-zonal

  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # IP allocation policy
  ip_allocation_policy {
    cluster_secondary_range_name  = google_compute_subnetwork.subnet.secondary_ip_range[0].range_name
    services_secondary_range_name = google_compute_subnetwork.subnet.secondary_ip_range[1].range_name
  }

  # Network tags
  network_policy {
    enabled = true
  }

  # Addons
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }

    vertical_pod_autoscaling {
      enabled = true
    }

    network_policy_config {
      disabled = false
    }

    gcp_filestore_csi_driver_config {
      enabled = true
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }

    dns_cache_config {
      enabled = true
    }
  }

  # Authentication and security
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_global_access_config {
      enabled = true
    }
    master_ipv4_cidr_block = "172.16.0.0/28"
  }

  enable_shielded_nodes = true

  security_posture_config {
    mode = "ENABLED"
  }

  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  # Binary authorization
  binary_authorization {
    enabled = true
  }

  # Logging and monitoring
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS"
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS"
    ]

    managed_prometheus {
      enabled = true
    }
  }

  # Resource labels
  resource_labels = {
    environment = var.environment
    application = "medical-ai-assistant"
    compliance  = "hipaa,fda,iso27001"
    team        = "medical-ai-devops"
  }

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  # Maintenance policy
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Release channel
  release_channel {
    channel = "RAPID"
  }

  depends_on = [
    google_project_service.required_apis["container.googleapis.com"]
  ]
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.cluster_name}-gke-nodes"
  display_name = "GKE Nodes Service Account for Medical AI"
}

resource "google_project_iam_member" "gke_nodes_permissions" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/artifactregistry.reader"
  ])

  project = var.gcp_project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Node Pools
resource "google_container_node_pool" "general_nodes" {
  name       = "general-nodes"
  location   = var.gcp_zones[0]
  cluster    = google_container_cluster.primary.name
  node_count = 3

  network_config {
    pod_range    = "pods-range"
    pods_per_node = 16
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 3
    max_node_count = 20
    location_policy = "ANY"
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  node_config {
    preemptible  = true
    machine_type = "e2-standard-4"
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    service_account = google_service_account.gke_nodes.email
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
      "node-type" = "general"
      "workload"  = "backend,frontend"
    }

    tags = ["medical-ai", "general-nodes"]
  }

  depends_on = [
    google_project_service.required_apis["container.googleapis.com"]
  ]
}

resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-nodes"
  location   = var.gcp_zones[0]
  cluster    = google_container_cluster.primary.name
  node_count = 1

  network_config {
    pod_range    = "pods-range"
    pods_per_node = 16
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 10
    location_policy = "ANY"
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    service_account = google_service_account.gke_nodes.email
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

    # GPU configuration
    guest_accelerator {
      type  = "nvidia-tesla-v100"
      count = 1
    }

    labels = {
      "node-type"   = "gpu"
      "accelerator" = "nvidia-tesla-v100"
      "workload"    = "model-serving"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    tags = ["medical-ai", "gpu-nodes"]
  }

  depends_on = [
    google_project_service.required_apis["container.googleapis.com"]
  ]
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "primary" {
  name             = "${var.cluster_name}-sql"
  database_version = "POSTGRES_15"
  region          = var.gcp_region

  deletion_protection = true

  settings {
    tier = "db-custom-4-15360"  # 4 vCPU, 15GB RAM

    disk_type = "PD_SSD"
    disk_size = 100
    disk_autoresize = true
    disk_autoresize_limit = 1000

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.gcp_region
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 35
        retention_unit   = "COUNT"
      }
      transaction_log_retention_days = 7
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      require_ssl     = true
    }

    database_flags {
      name  = "log_statement"
      value = "all"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }

    database_flags {
      name  = "shared_preload_libraries"
      value = "pgaudit,pg_stat_statements"
    }

    user_labels = {
      environment = var.environment
      compliance  = "hipaa,fda,iso27001"
    }

    insights_config {
      query_insights_enabled = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }

    maintenance_window {
      day          = 7
      hour         = 3
      update_track = "stable"
    }
  }

  depends_on = [
    google_project_service.required_apis["sql.googleapis.com"]
  ]
}

resource "google_sql_database" "database" {
  name     = "medical_ai"
  instance = google_sql_database_instance.primary.name
}

resource "google_sql_user" "users" {
  name     = "postgres"
  instance = google_sql_database_instance.primary.name
  password = var.database_password
}

# Redis Instance
resource "google_redis_instance" "cache" {
  name           = "${var.cluster_name}-redis"
  memory_size_gb = 4
  region        = var.gcp_region
  tier          = "STANDARD_HA"

  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version = "REDIS_7_0"

  region = var.gcp_region

  auth_enabled = true
  redis_config = {
    maxmemory-policy = "allkeys-lru"
  }

  depends_on = [
    google_project_service.required_apis["redis.googleapis.com"]
  ]
}

# Cloud Storage Buckets
resource "google_storage_bucket" "backups" {
  name          = "${var.cluster_name}-backups-${random_string.suffix.result}"
  location     = var.gcp_region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.medical_ai.id
  }

  lifecycle_rule {
    condition {
      age = 2555  # 7 years for HIPAA compliance
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [
    google_project_service.required_apis["storage.googleapis.com"]
  ]
}

resource "google_storage_bucket" "model_storage" {
  name          = "${var.cluster_name}-models-${random_string.suffix.result}"
  location     = var.gcp_region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.medical_ai.id
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  depends_on = [
    google_project_service.required_apis["storage.googleapis.com"]
  ]
}

# Cloud Load Balancer
resource "google_compute_region_network_endpoint_group" "backend_neg" {
  name                  = "${var.cluster_name}-backend-neg"
  region               = var.gcp_region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    service = google_cloud_run_service.backend.name
  }
}

resource "google_compute_region_network_endpoint_group" "frontend_neg" {
  name                  = "${var.cluster_name}-frontend-neg"
  region               = var.gcp_region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    service = google_cloud_run_service.frontend.name
  }
}

# Cloud Run Services for frontend and backend
resource "google_cloud_run_service" "backend" {
  name     = "${var.cluster_name}-backend"
  location = var.gcp_region

  template {
    metadata {
      labels = {
        run.googleapis.com/network-interfaces = '[{"network":"projects/${var.gcp_project_id}/global/networks/${google_compute_network.vpc.name}","subnetwork":"projects/${var.gcp_project_id}/regions/${var.gcp_region}/subnetworks/${google_compute_subnetwork.subnet.name}"}]'
      }
    }

    spec {
      service_account_name = google_service_account.gke_nodes.email
      container_concurrency = 80
      timeout_seconds      = 300
      containers {
        image = "gcr.io/${var.gcp_project_id}/medical-ai-backend:latest"

        ports {
          name           = "http1"
          container_port = 8000
        }

        env {
          name  = "PORT"
          value = "8000"
        }

        env {
          name  = "DATABASE_URL"
          value = "postgresql://postgres:${var.database_password}@${google_sql_database_instance.primary.connection_name}/medical_ai"
        }

        resources {
          limits = {
            cpu    = "4000m"
            memory = "8Gi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service" "frontend" {
  name     = "${var.cluster_name}-frontend"
  location = var.gcp_region

  template {
    metadata {
      labels = {
        run.googleapis.com/network-interfaces = '[{"network":"projects/${var.gcp_project_id}/global/networks/${google_compute_network.vpc.name}","subnetwork":"projects/${var.gcp_project_id}/regions/${var.gcp_region}/subnetworks/${google_compute_subnetwork.subnet.name}"}]'
      }
    }

    spec {
      service_account_name = google_service_account.gke_nodes.email
      container_concurrency = 100
      timeout_seconds      = 60
      containers {
        image = "gcr.io/${var.gcp_project_id}/medical-ai-frontend:latest"

        ports {
          name           = "http1"
          container_port = 80
        }

        env {
          name  = "PORT"
          value = "80"
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Cloud Run IAM
resource "google_cloud_run_service_iam_binding" "backend_invoker" {
  service  = google_cloud_run_service.backend.name
  location = var.gcp_region
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

resource "google_cloud_run_service_iam_binding" "frontend_invoker" {
  service  = google_cloud_run_service.frontend.name
  location = var.gcp_region
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

# Cloud Monitoring
resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "High CPU usage - Medical AI"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "High CPU condition"
    
    condition_threshold {
      filter          = "resource.type=\"gke_container\" AND cluster_name=\"${var.cluster_name}\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      duration        = "300s"
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
  
  alert_strategy {
    auto_close = "86400s"
  }

  depends_on = [
    google_project_service.required_apis["monitoring.googleapis.com"]
  ]
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Medical AI Email Notifications"
  type         = "email"

  labels = {
    email_address = "alerts@medical-ai.com"
  }

  depends_on = [
    google_project_service.required_apis["monitoring.googleapis.com"]
  ]
}

# Workload Identity Binding
resource "google_service_account_iam_member" "gke_nodes_workload_identity" {
  service_account_id = google_service_account.gke_nodes.id
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.gcp_project_id}.svc.id.goog[${google_container_cluster.primary.name}/${google_container_node_pool.general_nodes.name}]"
}

# Outputs
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.primary.connection_name
}

output "database_public_ip" {
  description = "Cloud SQL public IP"
  value       = google_sql_database_instance.primary.public_ip_address
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.cache.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.cache.port
}

output "backup_bucket_name" {
  description = "Backup storage bucket name"
  value       = google_storage_bucket.backups.name
}

output "model_storage_bucket_name" {
  description = "Model storage bucket name"
  value       = google_storage_bucket.model_storage.name
}

output "artifact_registry_repository" {
  description = "Artifact registry repository"
  value       = google_artifact_registry_repository.medical_ai.id
}

output "kms_key_name" {
  description = "KMS encryption key name"
  value       = google_kms_crypto_key.medical_ai.name
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.subnet.name
}

output "cloud_run_backend_url" {
  description = "Backend Cloud Run service URL"
  value       = google_cloud_run_service.backend.status[0].url
}

output "cloud_run_frontend_url" {
  description = "Frontend Cloud Run service URL"
  value       = google_cloud_run_service.frontend.status[0].url
}
