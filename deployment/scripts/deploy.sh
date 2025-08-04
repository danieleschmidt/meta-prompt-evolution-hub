#!/bin/bash
set -euo pipefail

# Meta-Prompt-Evolution-Hub Deployment Script
echo "🚀 Starting deployment of Meta-Prompt-Evolution-Hub"

# Configuration
NAMESPACE=${NAMESPACE:-production}
IMAGE_TAG=${IMAGE_TAG:-latest}
ENVIRONMENT=${ENVIRONMENT:-production}

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl is required"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "⚠️  helm not found, using kubectl"; }

# Create namespace if it doesn't exist
echo "🏗️  Setting up namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy using Helm if available, otherwise use kubectl
if command -v helm >/dev/null 2>&1; then
    echo "📦 Deploying with Helm..."
    helm upgrade --install meta-prompt-evolution-hub ./helm/meta-prompt-evolution-hub \
        --namespace $NAMESPACE \
        --set image.tag=$IMAGE_TAG \
        --set environment=$ENVIRONMENT \
        --wait --timeout=600s
else
    echo "📦 Deploying with kubectl..."
    kubectl apply -f kubernetes/ -n $NAMESPACE
    kubectl rollout status deployment/meta-prompt-evolution-hub -n $NAMESPACE --timeout=600s
fi

# Verify deployment
echo "🔍 Verifying deployment..."
kubectl get pods -n $NAMESPACE -l app=meta-prompt-evolution-hub
kubectl get services -n $NAMESPACE -l app=meta-prompt-evolution-hub

# Run health check
echo "🏥 Running health check..."
HEALTH_URL=$(kubectl get ingress meta-prompt-evolution-hub-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')
if [ ! -z "$HEALTH_URL" ]; then
    curl -f "https://$HEALTH_URL/health" || echo "⚠️  Health check failed"
else
    echo "⚠️  Ingress not configured, skipping external health check"
fi

echo "✅ Deployment completed successfully!"
