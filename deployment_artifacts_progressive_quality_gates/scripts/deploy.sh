#!/bin/bash
set -e

# Progressive Quality Gates Deployment Script
echo "🚀 Starting Progressive Quality Gates Deployment"

# Configuration
NAMESPACE="quality-gates"
APP_NAME="progressive-quality-gates"
VERSION="${1:-1.0.0}"
ENVIRONMENT="${2:-production}"

# Create namespace if it doesn't exist
echo "📁 Creating namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "📦 Applying Kubernetes manifests"
kubectl apply -f deployment.yaml -n $NAMESPACE
kubectl apply -f service.yaml -n $NAMESPACE
kubectl apply -f hpa.yaml -n $NAMESPACE

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$APP_NAME -n $NAMESPACE

# Check deployment status
echo "🔍 Checking deployment status"
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

# Run health checks
echo "🏥 Running health checks"
kubectl port-forward -n $NAMESPACE svc/$APP_NAME-service 8080:80 &
PROXY_PID=$!
sleep 5

# Test health endpoints
curl -f http://localhost:8080/health/live || (echo "❌ Health check failed" && exit 1)
curl -f http://localhost:8080/health/ready || (echo "❌ Readiness check failed" && exit 1)

# Kill port-forward
kill $PROXY_PID

echo "✅ Deployment completed successfully"
echo "🎉 Progressive Quality Gates is now running in $ENVIRONMENT"
