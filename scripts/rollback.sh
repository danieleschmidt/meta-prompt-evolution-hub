#!/bin/bash
# Production rollback script
set -e

echo "🔄 Starting rollback..."

# Get previous revision
PREVIOUS_REVISION=$(kubectl rollout history deployment/meta-prompt-evolution -n meta-prompt-evolution --limit=2 | tail -n 1 | awk '{print $1}')

# Rollback to previous revision
kubectl rollout undo deployment/meta-prompt-evolution -n meta-prompt-evolution --to-revision=$PREVIOUS_REVISION

# Wait for rollback to complete
kubectl rollout status deployment/meta-prompt-evolution -n meta-prompt-evolution

echo "✅ Rollback completed successfully!"
