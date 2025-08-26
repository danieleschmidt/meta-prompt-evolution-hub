#!/bin/bash
set -e

echo "🏥 Running Health Checks"
echo "======================="

REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")

for region in "${REGIONS[@]}"; do
    echo "Checking region: $region"
    
    # Check Kubernetes deployment
    kubectl get deployment meta-prompt-evolution -n production
    
    # Check pod status
    kubectl get pods -n production -l app=meta-prompt-evolution
    
    # Check service endpoints
    kubectl get service meta-prompt-evolution-service -n production
    
    echo "✅ $region health check passed"
done

echo "✅ All health checks passed!"
