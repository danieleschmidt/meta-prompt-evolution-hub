#!/bin/bash
# Production health check script
set -e

NAMESPACE="meta-prompt-evolution"
SERVICE_URL="http://meta-prompt-evolution-service.${NAMESPACE}.svc.cluster.local"

echo "🏥 Performing health checks..."

# Check deployment status
kubectl get deployment meta-prompt-evolution -n $NAMESPACE

# Check pod health
kubectl get pods -n $NAMESPACE -l app=meta-prompt-evolution

# Check service endpoints
kubectl get endpoints -n $NAMESPACE

# Test health endpoint
if command -v curl &> /dev/null; then
    kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- curl -f $SERVICE_URL/health
fi

echo "✅ Health checks completed!"
