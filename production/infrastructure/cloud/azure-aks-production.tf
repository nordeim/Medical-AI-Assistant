# Azure AKS Production Infrastructure for Medical AI Assistant
# Healthcare-compliant multi-zone deployment with HIPAA, FDA, and ISO 27001 compliance

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
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "azurerm" {
    resource_group_name  = "medical-ai-terraform-state"
    storage_account_name = "medicalaitfstatestorage"
    container_name       = "production-aks"
    key                  = "terraform.tfstate"
  }
}

# Provider Configuration
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy       = true
      recover_soft_deleted_key_vaults    = true
      purge_soft_deleted_secrets_on_destroy = true
      recover_soft_deleted_secrets_on_destroy = true
    }
  }
  skip_provider_registration = false
  use_msi                    = true
}

provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.primary kube_config.0.host
  token                  = data.azurerm_kubernetes_cluster_kube_config.this.kube_config_raw
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.primary kube_config[0].cluster_ca_certificate)
  load_config_file       = false
}

provider "helm" {
  kubernetes {
    host                   = azurerm_kubernetes_cluster.primary kube_config.0.host
    token                  = data.azurerm_kubernetes_cluster_kube_config.this.kube_config_raw
    cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.primary kube_config[0].cluster_ca_certificate)
    load_config_file       = false
  }
}

# Random string for unique naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "location" {
  description = "Azure region for Medical AI production deployment"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "medical-ai-production-rg"
}

variable "aks_cluster_name" {
  description = "AKS cluster name"
  type        = string
  default     = "medical-ai-production"
}

variable "network_plugin" {
  description = "Network plugin for AKS"
  type        = string
  default     = "azure"
}

variable "network_policy" {
  description = "Network policy for AKS"
  type        = string
  default     = "azure"
}

variable "pod_cidr" {
  description = "Pod CIDR for AKS"
  type        = string
  default     = "10.244.0.0/16"
}

variable "service_cidr" {
  description = "Service CIDR for AKS"
  type        = string
  default     = "10.0.0.0/16"
}

variable "dns_service_ip" {
  description = "DNS service IP for AKS"
  type        = string
  default     = "10.0.0.10"
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

variable "admin_username" {
  description = "Admin username for AKS nodes"
  type        = string
  default     = "azureuser"
}

# Enable Required Providers
data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    Environment = var.environment
    Application = "medical-ai-assistant"
    Compliance  = "hipaa,fda,iso27001"
    ManagedBy   = "terraform"
    CostCenter  = "healthcare-ml"
    Team        = "medical-ai-devops"
  }
}

# Azure Key Vault for encryption
resource "azurerm_key_vault" "medical_ai" {
  name                = "medical-ai-vault-${random_string.suffix.result}"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name           = "standard"

  soft_delete_retention_days = 90
  purge_protection_enabled   = true

  enabled_for_deployment          = true
  enabled_for_template_deployment = true
  enabled_for_disk_encryption     = true
  enable_rbac_authorization       = false

  tags = {
    Name        = "medical-ai-key-vault"
    Environment = var.environment
    Compliance  = "HIPAA"
  }
}

# Key Vault Key for encryption
resource "azurerm_key_vault_key" "medical_ai" {
  name         = "medical-ai-encryption-key"
  key_vault_id = azurerm_key_vault.medical_ai.id
  key_type     = "RSA"
  key_size     = 3072

  key_opts = [
    "decrypt",
    "encrypt",
    "sign",
    "unwrapKey",
    "verify",
    "wrapKey",
  ]

  depends_on = [
    azurerm_key_vault.medical_ai
  ]
}

# Key Vault Access Policy
resource "azurerm_key_vault_access_policy" "service_principal" {
  key_vault_id = azurerm_key_vault.medical_ai.id
  tenant_id   = data.azurerm_client_config.current.tenant_id
  object_id   = data.azurerm_client_config.current.object_id

  key_permissions = [
    "get",
    "list",
    "create",
    "delete",
    "update",
    "import",
    "backup",
    "restore",
    "decrypt",
    "encrypt",
    "unwrapKey",
    "wrapKey",
    "sign",
    "verify",
    "purge"
  ]

  secret_permissions = [
    "get",
    "list",
    "set",
    "delete",
    "backup",
    "restore",
    "recover",
    "purge"
  ]
}

# Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "medical-ai-vnet"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.0.0.0/8"]

  subnet {
    name           = "aks"
    address_prefixes = ["10.0.1.0/24"]
  }

  subnet {
    name           = "bastion"
    address_prefixes = ["10.0.2.0/24"]
  }

  tags = {
    Name        = "medical-ai-vnet"
    Environment = var.environment
  }
}

# Subnet for AKS nodes
resource "azurerm_subnet" "aks_nodes" {
  name                 = "aks-nodes"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.1.0.0/16"]
}

# Subnet for Azure Database for PostgreSQL
resource "azurerm_subnet" "database" {
  name                 = "database"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.2.0.0/16"]

  delegation {
    name = "postgres"
    service_delegation {
      name    = "Microsoft.DB/postgresqlServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

# Network Security Groups
resource "azurerm_network_security_group" "aks" {
  name                = "medical-ai-aks-nsg"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowAnyIn"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    Name        = "medical-ai-aks-nsg"
    Environment = var.environment
  }
}

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks_nodes.id
  network_security_group_id = azurerm_network_security_group.aks.id
}

# Public IP for Load Balancer
resource "azurerm_public_ip" "lb" {
  name                = "medical-ai-lb-pip"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = {
    Name        = "medical-ai-lb-pip"
    Environment = var.environment
  }
}

# Azure Container Registry
resource "azurerm_container_registry" "medical_ai" {
  name                = "medicalaicr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "Premium"
  admin_enabled       = false

  georeplications {
    location = "westus2"
  }

  georeplications {
    location = "canadacentral"
  }

  network_rule_set {
    default_action = "Deny"
    ip_rule {
      action   = "Allow"
      ip_range = "10.0.0.0/16"
    }
  }

  tags = {
    Name        = "medical-ai-acr"
    Environment = var.environment
  }
}

# User Assigned Managed Identity
resource "azurerm_user_assigned_identity" "aks" {
  name                = "medical-ai-aks-mi"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location

  tags = {
    Name        = "medical-ai-aks-mi"
    Environment = var.environment
  }
}

# Role Assignment for Managed Identity
resource "azurerm_role_assignment" "aks_vnet_contributor" {
  scope                = azurerm_virtual_network.vnet.id
  role_definition_name = "Contributor"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

resource "azurerm_role_assignment" "aks_key_vault_user" {
  scope              = azurerm_key_vault.medical_ai.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id       = azurerm_user_assigned_identity.aks.principal_id
}

resource "azurerm_role_assignment" "aks_monitoring_metrics_publisher" {
  scope                = data.azurerm_subscription.current.id
  role_definition_name = "Monitoring Metrics Publisher"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "primary" {
  name                = var.aks_cluster_name
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "medical-ai"

  kubernetes_version = "1.28.5"

  # Node pools
  default_node_pool {
    name                         = "default"
    node_count                   = 3
    vm_size                      = "Standard_D4s_v3"
    vnet_subnet_id              = azurerm_subnet.aks_nodes.id
    pod_subnet_id               = azurerm_subnet.aks_nodes.id
    enable_auto_scaling         = true
    max_count                   = 20
    min_count                   = 3
    max_pods                    = 50
    node_labels = {
      "node-type" = "general"
      "workload"  = "backend,frontend"
    }
    tags = {
      "node-pool"  = "default"
      "Environment" = var.environment
    }
  }

  # Network configuration
  network_profile {
    network_plugin     = var.network_plugin
    network_policy     = var.network_policy
    service_cidr       = var.service_cidr
    dns_service_ip     = var.dns_service_ip
    docker_bridge_cidr = "172.17.0.1/16"
    outbound_type      = "loadBalancer"
    load_balancer_sku  = "Standard"
  }

  # RBAC enabled
  role_based_access_control {
    enabled = true
    azure_active_directory {
      managed                = true
      azure_rbac_enabled     = true
      admin_group_object_ids = []
    }
  }

  # Security
  enable_pod_security_policy = false
  enable_rbac                = true
  enable_addons {
    oms_agent {
      enabled                    = true
      log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
    }
  }

  # Auto-upgrade
  auto_upgrade_channel = "stable"

  # Maintenance window
  maintenance_window {
    allowed {
      day   = "Sun"
      hours = [3]
    }
    allowed {
      day   = "Mon"
      hours = [3]
    }
    allowed {
      day   = "Tue"
      hours = [3]
    }
    allowed {
      day   = "Wed"
      hours = [3]
    }
    allowed {
      day   = "Thu"
      hours = [3]
    }
    allowed {
      day   = "Fri"
      hours = [3]
    }
    allowed {
      day   = "Sat"
      hours = [3]
    }
  }

  # System assigned identity
  identity {
    type = "UserAssigned"
    user_assigned_identity_ids = [
      azurerm_user_assigned_identity.aks.id
    ]
  }

  # Tags
  tags = {
    Environment = var.environment
    Application = "medical-ai-assistant"
    Compliance  = "hipaa,fda,iso27001"
    Team        = "medical-ai-devops"
  }
}

# GPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.primary.id
  vm_size               = "Standard_NC6s_v3"
  node_count           = 1
  vnet_subnet_id       = azurerm_subnet.aks_nodes.id
  pod_subnet_id        = azurerm_subnet.aks_nodes.id
  enable_auto_scaling  = true
  max_count           = 10
  min_count           = 0
  max_pods            = 30
  priority            = "Regular"
  node_labels = {
    "node-type"   = "gpu"
    "accelerator" = "nvidia-tesla-v100"
    "workload"    = "model-serving"
  }
  taint = [{
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }]

  tags = {
    "node-pool"  = "gpu"
    "Environment" = var.environment
  }
}

# Database Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "database" {
  name                  = "database"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.primary.id
  vm_size               = "Standard_D2s_v3"
  node_count           = 2
  vnet_subnet_id       = azurerm_subnet.aks_nodes.id
  pod_subnet_id        = azurerm_subnet.aks_nodes.id
  enable_auto_scaling  = true
  max_count           = 6
  min_count           = 1
  max_pods            = 40
  priority            = "Regular"
  node_labels = {
    "node-type" = "database"
    "workload"  = "postgresql,redis"
  }
  taint = [{
    key    = "database-node"
    value  = "true"
    effect = "NO_SCHEDULE"
  }]

  tags = {
    "node-pool"  = "database"
    "Environment" = var.environment
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "medical-ai-law-${random_string.suffix.result}"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  sku                = "PerGB2018"
  retention_in_days   = 2555  # 7 years for HIPAA compliance

  tags = {
    Name        = "medical-ai-law"
    Environment = var.environment
    Compliance  = "HIPAA"
  }
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "database" {
  name                   = "medical-ai-postgres"
  resource_group_name    = azurerm_resource_group.main.name
  location              = var.location
  version               = "15"

  # High availability
  zone                = "1"
  backup_retention_days = 35
  geo_redundant_backup_enabled = true
  storage_tier       = "IOPS"
  storage_gb        = 100
  max_storage_gb    = 1000

  # Networking
  vnet_id               = azurerm_virtual_network.vnet.id
  delegated_subnet_id   = azurerm_subnet.database.id
  private_dns_zone_id   = azurerm_private_dns_zone.postgres.id
  public_network_access_enabled = false

  # Authentication
  administrator_login    = "postgres"
  administrator_password = var.database_password

  # SSL
  ssl_minimal_tls_version_enforced = "TLS1_2"
  ssl_enforcement_enabled         = true

  # High availability and scalability
  high_availability {
    mode = "ZoneRedundant"
  }

  # Maintenance window
  maintenance_window {
    day_of_week  = 3
    hour_of_day  = 3
  }

  # Tags
  tags = {
    Name        = "medical-ai-postgres"
    Environment = var.environment
    Compliance  = "hipaa,fda,iso27001"
  }
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "database" {
  name      = "medical_ai"
  server_id = azurerm_postgresql_flexible_server.database.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Private DNS Zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  name                = "postgresql.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name

  tags = {
    Name        = "medical-ai-postgres-dns"
    Environment = var.environment
  }
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "medical-ai-postgres-vnet-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
}

# Redis Cache
resource "azurerm_redis_cache" "cache" {
  name                = "medical-ai-redis"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  capacity           = 1
  family            = "P"
  sku_name          = "Premium"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  replicas_per_primary = 1

  redis_configuration {
    maxmemory_reserved        = 2
    maxmemory_delta          = 2
    maxmemory_policy         = "allkeys-lru"
    enable_auth              = true
    keepalive                = 0
    maxclients               = 1000
    slowlog_log_slower_than  = 10
    slowlog_max_len          = 100
  }

  sku {
    capacity = 1
    family   = "P"
    name     = "Premium"
  }

  tags = {
    Name        = "medical-ai-redis"
    Environment = var.environment
    Compliance  = "hipaa,fda,iso27001"
  }
}

# Storage Account for backups
resource "azurerm_storage_account" "backups" {
  name                    = "medicalaibackups${random_string.suffix.result}"
  location                = var.location
  resource_group_name     = azurerm_resource_group.main.name
  account_tier            = "Standard"
  account_replication_type = "ZRS"
  account_kind           = "StorageV2"

  blob_properties {
    delete_retention_policy {
      days = 2555  # 7 years for HIPAA compliance
    }
  }

  cross_zone_redundancy_enabled = true
  enable_https_traffic_only    = true
  minimum_tls_version          = "TLS1_2"

  tags = {
    Name        = "medical-ai-backups"
    Environment = var.environment
    Compliance  = "HIPAA"
  }
}

# Storage Account for model storage
resource "azurerm_storage_account" "models" {
  name                    = "medicalaimodels${random_string.suffix.result}"
  location                = var.location
  resource_group_name     = azurerm_resource_group.main.name
  account_tier            = "Standard"
  account_replication_type = "ZRS"
  account_kind           = "StorageV2"

  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 90
    }
  }

  cross_zone_redundancy_enabled = true
  enable_https_traffic_only    = true
  minimum_tls_version          = "TLS1_2"

  tags = {
    Name        = "medical-ai-models"
    Environment = var.environment
    Compliance  = "HIPAA"
  }
}

# Azure Application Gateway
resource "azurerm_application_gateway" "medical_ai" {
  name                = "medical-ai-app-gateway"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  enable_http2        = true
  vnet_id            = azurerm_virtual_network.vnet.id
  subnet_id          = azurerm_subnet.aks_nodes.id

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 3
  }

  frontend_port {
    name = "http"
    port = 80
  }

  frontend_port {
    name = "https"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "app-gateway-frontend-ip"
    public_ip_address_id = azurerm_public_ip.lb.id
  }

  backend_address_pool {
    name = "backend-pool"
    fqdns = [azurerm_kubernetes_cluster.primary.fqdn]
  }

  backend_http_settings {
    name                  = "backend-http-settings"
    cookie_based_affinity = "Enabled"
    affinity_cookie_ttl  = 90
    port                  = 80
    protocol              = "Http"
    request_timeout      = 30
    probe_name           = "backend-probe"
  }

  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "app-gateway-frontend-ip"
    frontend_port_name             = "https"
    ssl_certificate_name          = "medical-ai-ssl-cert"
    protocol                      = "Https"
  }

  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "app-gateway-frontend-ip"
    frontend_port_name             = "http"
    protocol                      = "Http"
  }

  probe {
    name                = "backend-probe"
    host                = "medical-ai.com"
    path                = "/health"
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3
    protocol            = "Http"
  }

  redirect_configuration {
    name = "redirect-http-to-https"
    redirect_type = "Permanent"
    target_listener_name = "https-listener"
  }

  request_routing_rule {
    name                        = "https-rule"
    rule_type                  = "Basic"
    http_listener_name         = "https-listener"
    backend_address_pool_name  = "backend-pool"
    backend_http_settings_name = "backend-http-settings"
    priority                   = 100
  }

  request_routing_rule {
    name                        = "http-rule"
    rule_type                  = "Basic"
    http_listener_name         = "http-listener"
    redirect_configuration_name = "redirect-http-to-https"
    priority                   = 200
  }

  tags = {
    Name        = "medical-ai-app-gateway"
    Environment = var.environment
  }
}

# Application Gateway Certificate (would need to be managed via Azure Key Vault in production)
resource "azurerm_application_gateway_ssl_certificate" "medical_ai" {
  name                = "medical-ai-ssl-cert"
  application_gateway = azurerm_application_gateway.medical_ai.id
  key_vault_secret_id = azurerm_key_vault_secret.ssl_cert.id
  depends_on = [
    azurerm_key_vault_secret.ssl_cert
  ]
}

# Application Gateway WAF Policy
resource "azurerm_web_application_firewall_policy" "medical_ai" {
  name                = "medical-ai-waf-policy"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name

  policy_settings {
    enabled                     = true
    mode                        = "Prevention"
    request_body_check          = true
    file_upload_limit_in_mb     = 100
    max_request_body_size_in_kb = 128
  }

  managed_rules {
    rule_set_type    = "OWASP_3.2"
    rule_set_version = "3.2"
  }

  custom_rules {
    name     = "AllowHealthChecks"
    priority = 100
    rule_type = "MatchRule"
    action   = "Allow"

    match_conditions {
      match_variables {
        variable_name = "RequestUri"
      }
      operator     = "Contains"
      match_values = ["/health"]
    }
  }

  tags = {
    Name        = "medical-ai-waf-policy"
    Environment = var.environment
  }
}

# Key Vault Secret for SSL Certificate
resource "azurerm_key_vault_secret" "ssl_cert" {
  name         = "medical-ai-ssl-certificate"
  value        = "your-ssl-certificate-base64"
  key_vault_id = azurerm_key_vault.medical_ai.id

  depends_on = [
    azurerm_key_vault_access_policy.service_principal
  ]
}

# Monitor Alert for AKS
resource "azurerm_monitor_metric_alert" "aks_cpu_high" {
  name                = "AKS CPU High"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_kubernetes_cluster.primary.id]
  description         = "Alert when AKS cluster CPU usage is high"

  criteria {
    metric_namespace = "microsoft.containerservice/managedclusters"
    metric_name      = "nodeCpuUsageMillicores"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  window_duration = "PT5M"

  action {
    action_group_id = azurerm_monitor_action_group.main.id
  }
}

# Monitor Action Group
resource "azurerm_monitor_action_group" "main" {
  name                = "medical-ai-alerts"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "medicalai"

  email_receiver {
    name                    = "email-receiver"
    email_address           = "alerts@medical-ai.com"
    use_common_alert_schema = true
  }

  webhook_receiver {
    name        = "slack-webhook"
    service_uri = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  }

  tags = {
    Name        = "medical-ai-alerts"
    Environment = var.environment
  }
}

# Kubernetes Kubeconfig
data "azurerm_kubernetes_cluster_kube_config" "this" {
  depends_on   = [azurerm_kubernetes_cluster.primary]
  cluster_name = azurerm_kubernetes_cluster.primary.name
  resource_group_name = azurerm_resource_group.main.name
}

# Outputs
output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.primary.name
}

output "cluster_fqdn" {
  description = "AKS cluster FQDN"
  value       = azurerm_kubernetes_cluster.primary.fqdn
}

output "cluster_location" {
  description = "AKS cluster location"
  value       = azurerm_kubernetes_cluster.primary.location
}

output "database_fqdn" {
  description = "PostgreSQL Flexible Server FQDN"
  value       = azurerm_postgresql_flexible_server.database.fqdn
}

output "redis_hostname" {
  description = "Redis cache hostname"
  value       = azurerm_redis_cache.cache.hostname
}

output "redis_port" {
  description = "Redis cache port"
  value       = azurerm_redis_cache.cache.ssl_port
}

output "backup_storage_account_name" {
  description = "Backup storage account name"
  value       = azurerm_storage_account.backups.name
}

output "model_storage_account_name" {
  description = "Model storage account name"
  value       = azurerm_storage_account.models.name
}

output "container_registry_name" {
  description = "Container registry name"
  value       = azurerm_container_registry.medical_ai.name
}

output "application_gateway_fqdn" {
  description = "Application Gateway FQDN"
  value       = azurerm_application_gateway.medical_ai.fqdn
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = azurerm_key_vault.medical_ai.name
}

output "managed_identity_client_id" {
  description = "Managed Identity Client ID"
  value       = azurerm_user_assigned_identity.aks.client_id
}

output "log_analytics_workspace_id" {
  description = "Log Analytics Workspace ID"
  value       = azurerm_log_analytics_workspace.main.workspace_id
}

output "log_analytics_workspace_key" {
  description = "Log Analytics Workspace Key"
  value       = azurerm_log_analytics_workspace.main.primary_shared_key
  sensitive   = true
}

# Data Sources
data "azurerm_subscription" "current" {
}
