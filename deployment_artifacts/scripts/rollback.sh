#!/bin/bash
set -e

echo "🔄 Rolling Back Deployment"
echo "========================="

REGION=${1:-us-east-1}

echo "Rolling back in region: $REGION"

# Rollback Kubernetes deployment
kubectl rollout undo deployment/meta-prompt-evolution -n production

# Wait for rollback to complete
kubectl rollout status deployment/meta-prompt-evolution -n production

# Verify rollback
kubectl get pods -n production -l app=meta-prompt-evolution

echo "✅ Rollback completed successfully!"
