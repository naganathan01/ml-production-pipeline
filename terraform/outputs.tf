output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.ml_vpc.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.ml_vpc.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.ml_cluster.id
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.ml_cluster.arn
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.ml_cluster.endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.ml_cluster.vpc_config[0].cluster_security_group_id
}

output "eks_cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = aws_eks_cluster.ml_cluster.version
}

output "eks_cluster_ca_certificate" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.ml_cluster.certificate_authority[0].data
  sensitive   = true
}

output "eks_node_group_arn" {
  description = "Amazon Resource Name (ARN) of the EKS Node Group"
  value       = aws_eks_node_group.ml_nodes.arn
}

output "eks_node_group_status" {
  description = "Status of the EKS Node Group"
  value       = aws_eks_node_group.ml_nodes.status
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.arn
}

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.ml_alb.dns_name
}

output "load_balancer_hosted_zone_id" {
  description = "Hosted zone ID of the load balancer"
  value       = aws_lb.ml_alb.zone_id
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks_cluster_logs.name
}

# kubectl config command
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${aws_eks_cluster.ml_cluster.name}"
}

# Environment information
output "environment_info" {
  description = "Environment information"
  value = {
    environment     = var.environment
    project_name    = var.project_name
    aws_region      = var.aws_region
    cluster_name    = aws_eks_cluster.ml_cluster.name
    node_group_name = aws_eks_node_group.ml_nodes.node_group_name
  }
}