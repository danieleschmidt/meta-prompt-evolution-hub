#!/bin/bash
set -e

echo "🚀 Deploying Meta-Prompt-Evolution-Hub..."

# Apply Kubernetes manifests
kubectl apply -f ../kubernetes/

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/meta-prompt-hub -n meta-prompt-hub

echo "✅ Deployment completed!"
