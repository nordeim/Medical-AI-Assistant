# Azure AKS Production Deployment Configuration for Medical AI Assistant
# Multi-region deployment with healthcare compliance
# Terraform configuration for Azure infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
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
  
  backend "azurerm" {
    resource_group_name  = "medical-ai-terraform"
    storage_account_name = "medicalaitfstate"
    container_name      = "tfstate"
    key                 = "production/terraform.tfstate"
    access_key          = var.storage_account_access_key
  }
}

# Azure Provider Configuration
provider "azurerm" {
  features {
    key_vault {
      recover_soft_deleted_key_vaults   = true
      purge_soft_deleted_secrets_on_destroy = true
      purge_soft_deleted_certificates_on_destroy = true
    }
    log_analytics_workspace {
      permanently_delete_on_destroy = true
    }
  }
  
  tenant_id       = var.azure_tenant_id
  subscription_id = var.azure_subscription_id
}

# Kubernetes provider
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
  token                  = azurerm_kubernetes_cluster.main.kube_config[0].token
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
    token                  = azurerm_kubernetes_cluster.main.kube_config[0].token
    cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
  }
}

# Variables
variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

variable "azure_tenant_id" {
  description = "Azure tenant ID"
  type        = string
  sensitive   = true
}

variable "azure_region" {
  description = "Primary Azure region"
  type        = string
  default     = "East US"
}

variable "azure_region_2" {
  description = "Secondary Azure region for DR"
  type        = string
  default     = "West US 2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "AKS cluster name"
  type        = string
  default     = "medical-ai-prod"
}

variable "dns_prefix" {
  description = "DNS prefix for cluster"
  type        = string
  default     = "medical-ai-dns"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28.3"
}

variable "storage_account_access_key" {
  description = "Storage account access key for Terraform state"
  type        = string
  sensitive   = true
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

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${var.cluster_name}-${random_id.random_suffix.hex}"
  location = var.azure_region
  
  tags = {
    environment         = var.environment
    project             = "medical-ai"
    managed-by          = "terraform"
    compliance          = "HIPAA"
    data-classification = "PHI"
    cost-center         = "medical-ai"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "vnet-${var.cluster_name}"
  address_space       = ["10.0.0.0/8"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# Subnet for AKS cluster
resource "azurerm_subnet" "aks" {
  name                 = "snet-aks-${var.cluster_name}"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.240.0.0/16"]
  
  # Network Security Group
  enforce_private_link_service_network_policies = false
}

# Subnet for Azure Database
resource "azurerm_subnet" "database" {
  name                 = "snet-database-${var.cluster_name}"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.241.0.0/16"]
  
  enforce_private_link_service_network_policies = false
}

# Subnet for Azure Cache for Redis
resource "azurerm_subnet" "redis" {
  name                 = "snet-redis-${var.cluster_name}"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.242.0.0/16"]
  
  enforce_private_link_service_network_policies = false
}

# Network Security Groups
resource "azurerm_network_security_group" "aks" {
  name                = "nsg-aks-${var.cluster_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  security_rule {
    name                       = "AllowAnyInBoundHTTP"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  security_rule {
    name                       = "AllowAnyInBoundHTTPS"
    priority                   = 200
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  security_rule {
    name                       = "AllowSSHInBound"
    priority                   = 300
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "35.235.240.0/20"
    destination_address_prefix = "*"
  }
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# AKS Managed Identity
resource "azurerm_user_assigned_identity" "aks" {
  name                = "id-aks-${var.cluster_name}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# Role assignments for AKS managed identity
resource "azurerm_role_assignment" "aks_network_contributor" {
  scope                = azurerm_virtual_network.main.id
  role_definition_name = "Network Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

resource "azurerm_role_assignment" "aks_contributor" {
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = var.cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.dns_prefix
  kubernetes_version  = var.kubernetes_version
  
  # Default node pool
  default_node_pool {
    name                  = "default"
    node_count            = 3
    vm_size              = "Standard_D4s_v3"
    os_disk_size_gb      = 100
    os_disk_type         = "Ephemeral"
    type                 = "VirtualMachineScaleSets"
    enable_auto_scaling  = true
    min_count           = 3
    max_count           = 10
    max_pods            = 30
    node_labels = {
      workload-type = "general"
    }
    node_taint = ["medical-ai-dedicated=general:NoSchedule"]
    
    # Upgrade settings
    upgrade_settings {
      max_surge = 1
    }
  }
  
  # Managed identity
  identity {
    type = "UserAssigned"
    user_assigned_identity_ids = [azurerm_user_assigned_identity.aks.id]
  }
  
  # Network profile
  network_profile {
    network_plugin     = "azure"
    network_policy     = "azure"
    service_cidr       = "10.0.0.0/16"
    dns_service_ip     = "10.0.0.10"
    docker_bridge_cidr = "172.17.0.1/16"
    outbound_type      = "loadBalancer"
    load_balancer_sku  = "Standard"
  }
  
  # Add-on profiles
  addon_profile {
    azure_keyvault_secrets_provider {
      enabled = true
      secret_identity {
        principal_id = azurerm_user_assigned_identity.aks.principal_id
        client_id    = azurerm_user_assigned_identity.aks.client_id
      }
    }
    
    oms_agent {
      enabled = true
      log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
    }
  }
  
  # RBAC enabled
  role_based_access_control {
    enabled = true
  }
  
  # API server authorized IP ranges (restrict access)
  api_server_access_profile {
    authorized_ip_ranges = [
      azurerm_subnet.aks.address_prefixes[0],
      "10.0.0.0/16"
    ]
  }
  
  # Azure Active Directory integration
  azure_active_directory_role_based_access_control {
    managed                = true
    tenant_id             = var.azure_tenant_id
    admin_group_object_ids = [azuread_group.aks_admins.id]
  }
  
  # Autoscaling profile
  autoscaling_profile {
    max_node_count = 50
    min_node_count = 3
  }
  
  # Maintenance window
  maintenance_window {
    allowed {
      day   = "Sun"
      hours = [3]
    }
  }
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
    compliance  = "HIPAA"
  }
}

# Additional node pools
resource "azurerm_kubernetes_cluster_node_pool" "backend" {
  name                  = "backend"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = "Standard_F8s_v2"
  os_disk_size_gb      = 100
  os_disk_type         = "Ephemeral"
  type                 = "VirtualMachineScaleSets"
  enable_auto_scaling  = true
  min_count           = 2
  max_count           = 10
  max_pods            = 50
  node_labels = {
    workload-type = "backend"
  }
  node_taint = ["medical-ai-dedicated=backend:NoSchedule"]
  
  # Upgrade settings
  upgrade_settings {
    max_surge = 1
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "frontend" {
  name                  = "frontend"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = "Standard_D4s_v3"
  os_disk_size_gb      = 50
  os_disk_type         = "Ephemeral"
  type                 = "VirtualMachineScaleSets"
  enable_auto_scaling  = true
  min_count           = 2
  max_count           = 10
  max_pods            = 30
  node_labels = {
    workload-type = "frontend"
  }
  node_taint = ["medical-ai-dedicated=frontend:NoSchedule"]
  
  # Upgrade settings
  upgrade_settings {
    max_surge = 1
  }
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = "Standard_NC6s_v3"
  os_disk_size_gb      = 100
  os_disk_type         = "Ephemeral"
  type                 = "VirtualMachineScaleSets"
  enable_auto_scaling  = true
  min_count           = 1
  max_count           = 8
  max_pods            = 30
  node_labels = {
    accelerator          = "nvidia-tesla-v100"
    workload-type        = "gpu"
  }
  node_taint = ["nvidia.com/gpu=true:NoSchedule"]
  
  # Upgrade settings
  upgrade_settings {
    max_surge = 1
  }
}

# Azure AD group for AKS admins
resource "azuread_group" "aks_admins" {
  display_name     = "AKS-Admins"
  description      = "Azure AD group for AKS administrators"
  security_enabled = true
  mail_enabled     = false
  members = [
    var.aks_admin_user_object_id
  ]
}

# Azure Database for PostgreSQL
resource "azurerm_database" "postgresql" {
  name                = "db-${var.cluster_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  server_name         = azurerm_postgresql_server.main.name
  charset             = "UTF8"
  collation           = "en_US.utf8"
}

resource "azurerm_postgresql_server" "main" {
  name                = "pg-${var.cluster_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  administrator_login          = var.db_username
  administrator_login_password = var.db_password
  
  sku_name   = "GP_Gen5_4"
  version    = "15"
  storage_mb = 327680
  
  auto_grow_enabled                = true
  backup_retention_days            = 30
  geo_redundant_backup_enabled     = true
  infrastructure_encryption_enabled = true
  public_network_access_enabled    = false
  ssl_enforcement_enabled          = true
  ssl_minimal_tls_version_enforced = "TLS1_2"
  
  # Performance insights
  performance_insights_enabled = true
  performance_insights_retention_days = 7
  
  threat_detection_policy {
    enabled = true
    disabled_alerts = []
    email_addresses = [var.security_alert_email]
    retention_days = 30
  }
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
    compliance  = "HIPAA"
  }
}

# Private DNS Zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  name                = " privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "postgres-dns-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "main" {
  name                = "redis-${var.cluster_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 2
  family             = "P"
  sku_name           = "Premium"
  redis_version      = "6.0"
  
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  redis_configuration {
    maxmemory_reserved = 2
    maxmemory_delta    = 2
    maxmemory_policy   = "allkeys-lru"
    
    # Backup configuration
    rdb_backup_enabled = true
    rdb_backup_frequency = 60
    rdb_backup_max_snapshot_count = 1
  }
  
  # Premium features
  patch_schedule {
    day_of_week        = "Sunday"
    start_hour_utc     = 3
    maintenance_window = "PT7H"
  }
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
    compliance  = "HIPAA"
  }
}

# Key Vault for secrets management
resource "azurerm_key_vault" "main" {
  name                       = "kv-${var.cluster_name}-${random_id.random_suffix.hex}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = var.azure_tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 90
  purge_protection_enabled   = true
  
  enabled_for_deployment          = true
  enabled_for_template_deployment = true
  enabled_for_disk_encryption     = true
  
  access_policy {
    tenant_id = var.azure_tenant_id
    object_id = azurerm_user_assigned_identity.aks.principal_id
    
    key_permissions = [
      "get", "list", "sign", "verify", "wrapKey", "unwrapKey"
    ]
    
    secret_permissions = [
      "get", "list", "set", "delete", "recover", "backup", "restore"
    ]
    
    storage_permissions = [
      "get", "list", "set", "delete", "update", "regenerateKey", "setSAS", "listSAS"
    ]
  }
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
    compliance  = "HIPAA"
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${var.cluster_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 90
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "ai-${var.cluster_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.main.id
  
  retention_in_days = 90
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = "acr${random_id.random_suffix.hex}"
  resource_group_name = azurerm_resource_group.main.location
  location            = azurerm_resource_group.main.location
  sku                = "Premium"
  admin_enabled       = false
  
  network_rule_set {
    default_action = "Deny"
    
    ip_rule {
      action   = "Allow"
      ip_range = "10.240.0.0/16"
    }
  }
  
  trust_policy {
    enabled = true
  }
  
  retention_policy {
    days = 30
  }
  
  export_policy {
    enabled = false
  }
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# Azure Monitor for Containers
resource "azurerm_kubernetes_cluster" "main" {
  # This block adds container insights to the AKS cluster
  addon_profile {
    oms_agent {
      enabled = true
      log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
    }
  }
}

# Container insights solution
resource "azurerm_log_analytics_solution" "container_insights" {
  solution_name         = "ContainerInsights"
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  workspace_name        = azurerm_log_analytics_workspace.main.name
  workspace_resource_id = azurerm_log_analytics_workspace.main.id
  
  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/ContainerInsights"
  }
}

# Azure Security Center
resource "azurerm_security_center_subscription" "default" {
  provider = azurerm

  security_contact_configurations {
    security_contact {
      email = var.security_alert_email
      phone = var.security_alert_phone
      alert_manager = true
      security_contact_roles = ["Owner"]
    }
  }
}

# Backup vault for disaster recovery
resource "azurerm_backup_vault" "main" {
  name                = "bv-${var.cluster_name}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  storage_type        = "LocallyRedundant"
  soft_delete_enabled = true
  
  tags = {
    environment = var.environment
    project     = "medical-ai"
  }
}

# Outputs
output "aks_cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.main.name
}

output "aks_cluster_fqdn" {
  description = "AKS cluster FQDN"
  value       = azurerm_kubernetes_cluster.main.fqdn
}

output "aks_cluster_kubeconfig" {
  description = "AKS cluster kubeconfig"
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

output "database_server_name" {
  description = "PostgreSQL server name"
  value       = azurerm_postgresql_server.main.name
}

output "redis_cache_name" {
  description = "Redis cache name"
  value       = azurerm_redis_cache.main.name
}

output "container_registry_name" {
  description = "Container registry name"
  value       = azurerm_container_registry.main.name
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = azurerm_key_vault.main.name
}

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.main.id
}

# Additional variables
variable "aks_admin_user_object_id" {
  description = "Azure AD object ID for AKS admin user"
  type        = string
}

variable "security_alert_email" {
  description = "Email address for security alerts"
  type        = string
}

variable "security_alert_phone" {
  description = "Phone number for security alerts"
  type        = string
}

# Random suffix for unique resource names
resource "random_id" "random_suffix" {
  byte_length = 3
}