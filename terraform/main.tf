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
  }
  
  backend "s3" {
    bucket = "ml-terraform-state-bucket"
    key    = "ml-production-pipeline/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ml-production-pipeline"
      Environment = var.environment
      Owner       = "ml-team"
      CreatedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Configuration
resource "aws_vpc" "ml_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc-${var.environment}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ml_igw" {
  vpc_id = aws_vpc.ml_vpc.id
  
  tags = {
    Name = "${var.project_name}-igw-${var.environment}"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count = length(var.public_subnet_cidrs)
  
  vpc_id                  = aws_vpc.ml_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}-${var.environment}"
    Type = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count = length(var.private_subnet_cidrs)
  
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}-${var.environment}"
    Type = "private"
  }
}

# Route Tables
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.ml_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ml_igw.id
  }
  
  tags = {
    Name = "${var.project_name}-public-rt-${var.environment}"
  }
}

resource "aws_route_table_association" "public_rta" {
  count = length(aws_subnet.public_subnets)
  
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# NAT Gateway
resource "aws_eip" "nat_eip" {
  count  = length(aws_subnet.public_subnets)
  domain = "vpc"
  
  tags = {
    Name = "${var.project_name}-nat-eip-${count.index + 1}-${var.environment}"
  }
  
  depends_on = [aws_internet_gateway.ml_igw]
}

resource "aws_nat_gateway" "nat_gw" {
  count = length(aws_subnet.public_subnets)
  
  allocation_id = aws_eip.nat_eip[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id
  
  tags = {
    Name = "${var.project_name}-nat-gw-${count.index + 1}-${var.environment}"
  }
}

# Private Route Table
resource "aws_route_table" "private_rt" {
  count = length(aws_subnet.private_subnets)
  
  vpc_id = aws_vpc.ml_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gw[count.index].id
  }
  
  tags = {
    Name = "${var.project_name}-private-rt-${count.index + 1}-${var.environment}"
  }
}

resource "aws_route_table_association" "private_rta" {
  count = length(aws_subnet.private_subnets)
  
  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt[count.index].id
}

# Security Groups
resource "aws_security_group" "ml_api_sg" {
  name        = "${var.project_name}-api-sg-${var.environment}"
  description = "Security group for ML API"
  vpc_id      = aws_vpc.ml_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-api-sg-${var.environment}"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "ml_cluster" {
  name     = "${var.project_name}-cluster-${var.environment}"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = var.kubernetes_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public_subnets[*].id, aws_subnet.private_subnets[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
    security_group_ids      = [aws_security_group.eks_cluster_sg.id]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]
  
  tags = {
    Name = "${var.project_name}-cluster-${var.environment}"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "ml_nodes" {
  cluster_name    = aws_eks_cluster.ml_cluster.name
  node_group_name = "${var.project_name}-nodes-${var.environment}"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id
  
  scaling_config {
    desired_size = var.node_group_desired_size
    max_size     = var.node_group_max_size
    min_size     = var.node_group_min_size
  }
  
  update_config {
    max_unavailable = 1
  }
  
  instance_types = var.node_instance_types
  ami_type       = "AL2_x86_64"
  capacity_type  = "ON_DEMAND"
  disk_size      = 20
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read_only,
  ]
  
  tags = {
    Name = "${var.project_name}-nodes-${var.environment}"
  }
}

# S3 Bucket for Model Artifacts
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.project_name}-artifacts-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.project_name}-artifacts-${var.environment}"
    Purpose     = "ml-model-storage"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "ml_artifacts_versioning" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "ml_artifacts_encryption" {
  bucket = aws_s3_bucket.ml_artifacts.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_artifacts_pab" {
  bucket = aws_s3_bucket.ml_artifacts.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "eks_cluster_logs" {
  name              = "/aws/eks/${var.project_name}-cluster-${var.environment}/cluster"
  retention_in_days = 7
  
  tags = {
    Name = "${var.project_name}-eks-logs-${var.environment}"
  }
}

# Application Load Balancer
resource "aws_lb" "ml_alb" {
  name               = "${var.project_name}-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name = "${var.project_name}-alb-${var.environment}"
  }
}

resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg-${var.environment}"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.ml_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-alb-sg-${var.environment}"
  }
}

# EKS Security Groups
resource "aws_security_group" "eks_cluster_sg" {
  name        = "${var.project_name}-eks-cluster-sg-${var.environment}"
  description = "Security group for EKS cluster control plane"
  vpc_id      = aws_vpc.ml_vpc.id
  
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-eks-cluster-sg-${var.environment}"
  }
}