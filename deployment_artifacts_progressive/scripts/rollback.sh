#!/bin/bash
set -e

echo "🔄 Rolling back Progressive Quality Gates deployment"

PREVIOUS_VERSION=${1:-"previous"}

# Rollback deployment
kubectl rollout undo deployment/progressive-quality-gates -n production

# Wait for rollback to complete
kubectl rollout status deployment/progressive-quality-gates -n production

# Verify rollback
./health-check.sh

echo "✅ Rollback completed successfully!"
