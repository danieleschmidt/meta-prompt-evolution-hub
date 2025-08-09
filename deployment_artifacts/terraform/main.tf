terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "meta-prompt-vpc"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "meta-prompt-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids = [aws_subnet.main.id]
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier     = "meta-prompt-db"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage = 20
  storage_encrypted = true
  
  db_name  = "metaprompt"
  username = "dbuser"
  password = "changeme"
  
  skip_final_snapshot = true
}

# Outputs
output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}
