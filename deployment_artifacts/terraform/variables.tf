# Variables for meta-prompt-evolution-hub
variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "meta-prompt-evolution-hub"
}

variable "supported_regions" {
  description = "List of supported regions for multi-region deployment"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-southeast-1"]
}

variable "min_replicas" {
  description = "Minimum number of replicas"
  type        = number
  default     = 3
}

variable "max_replicas" {
  description = "Maximum number of replicas"
  type        = number
  default     = 50
}

variable "ssl_cert_domain" {
  description = "Domain for SSL certificate"
  type        = string
  default     = "meta-prompt-hub.com"
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 365
}
