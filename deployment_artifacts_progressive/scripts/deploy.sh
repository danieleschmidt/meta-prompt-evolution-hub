#!/bin/bash
set -e

echo "🚀 Deploying Progressive Quality Gates to Production"

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t terragon/progressive-quality-gates:latest .
docker push terragon/progressive-quality-gates:latest

# Deploy to Kubernetes
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f kubernetes/

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/progressive-quality-gates -n production

# Run health check
echo "🔍 Running health checks..."
./health-check.sh

echo "✅ Deployment completed successfully!"
