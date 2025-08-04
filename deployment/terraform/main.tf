# Meta-Prompt-Evolution-Hub Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
resource "aws_eks_cluster" "meta_prompt_evolution" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids = var.subnet_ids
    endpoint_config {
      private_access = true
      public_access  = true
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]

  tags = {
    Environment = var.environment
    Project     = "meta-prompt-evolution-hub"
  }
}

# Node Group
resource "aws_eks_node_group" "meta_prompt_evolution" {
  cluster_name    = aws_eks_cluster.meta_prompt_evolution.name
  node_group_name = "${var.cluster_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node_group_role.arn
  subnet_ids      = var.subnet_ids

  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }

  instance_types = var.node_instance_types

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Environment = var.environment
    Project     = "meta-prompt-evolution-hub"
  }
}

# RDS Database
resource "aws_db_instance" "meta_prompt_evolution" {
  identifier             = "${var.cluster_name}-db"
  engine                = "postgres"
  engine_version        = "15.4"
  instance_class        = var.db_instance_class
  allocated_storage     = var.db_allocated_storage
  storage_encrypted     = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.meta_prompt_evolution.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-db-final-snapshot"

  tags = {
    Environment = var.environment
    Project     = "meta-prompt-evolution-hub"
  }
}

# Redis Cache
resource "aws_elasticache_replication_group" "meta_prompt_evolution" {
  replication_group_id    = "${var.cluster_name}-cache"
  description            = "Redis cache for Meta-Prompt-Evolution-Hub"
  
  node_type              = var.redis_node_type
  port                   = 6379
  parameter_group_name   = "default.redis7"
  
  num_cache_clusters     = var.redis_num_cache_nodes
  
  subnet_group_name      = aws_elasticache_subnet_group.meta_prompt_evolution.name
  security_group_ids     = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Environment = var.environment
    Project     = "meta-prompt-evolution-hub"
  }
}
