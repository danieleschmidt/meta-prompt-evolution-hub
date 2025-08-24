#!/bin/bash
# Production deployment script for Meta-Prompt-Evolution-Hub
set -e

echo "🚀 Starting production deployment..."

# Build and push Docker image
echo "🐳 Building Docker image..."
docker build -f Dockerfile.prod -t meta-prompt-evolution:3.0.0 .
docker tag meta-prompt-evolution:3.0.0 meta-prompt-evolution:latest

# Create namespace if not exists
kubectl create namespace meta-prompt-evolution --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "☸️ Applying Kubernetes manifests..."
kubectl apply -f kubernetes/ -n meta-prompt-evolution

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/meta-prompt-evolution -n meta-prompt-evolution

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n meta-prompt-evolution
kubectl get services -n meta-prompt-evolution

echo "🎉 Production deployment completed successfully!"
