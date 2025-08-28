#!/bin/bash
set -e

echo "🔍 Running Progressive Quality Gates Health Checks"

# Check if service is running
if kubectl get pods -n production -l app=progressive-quality-gates | grep Running; then
    echo "✅ Service pods are running"
else
    echo "❌ Service pods are not running"
    exit 1
fi

# Check service endpoint
SERVICE_IP=$(kubectl get service progressive-quality-gates-service -n production -o jsonpath='{.spec.clusterIP}')
if curl -f http://$SERVICE_IP/health; then
    echo "✅ Health endpoint is responsive"
else
    echo "❌ Health endpoint is not responsive"
    exit 1
fi

# Run quick quality gates test
echo "🧪 Running quality gates test..."
kubectl exec -it deployment/progressive-quality-gates -n production -- python3 progressive_quality_gates_comprehensive.py

echo "✅ All health checks passed!"
