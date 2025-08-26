#!/bin/bash
set -e

echo "🚀 Starting Meta-Prompt Evolution Hub Deployment"
echo "================================================"

# Configuration
ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}

echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t meta-prompt-evolution:latest ./docker/

# Deploy to Kubernetes
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f ./regions/$REGION/kubernetes.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/meta-prompt-evolution -n production

# Health check
echo "🏥 Performing health check..."
kubectl get pods -n production -l app=meta-prompt-evolution

# Deploy monitoring
echo "📊 Deploying monitoring stack..."
kubectl apply -f ./monitoring/

echo "✅ Deployment completed successfully!"
echo "📊 Monitor at: https://grafana-$REGION.monitoring.example.com"
