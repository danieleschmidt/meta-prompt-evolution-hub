variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "meta-prompt-evolution-hub"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "subnet_ids" {
  description = "VPC subnet IDs"
  type        = list(string)
}

variable "node_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}

variable "node_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "node_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = 2
}

variable "node_instance_types" {
  description = "Node instance types"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage"
  type        = number
  default     = 20
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "metapromptevolution"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "mpehub"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}
