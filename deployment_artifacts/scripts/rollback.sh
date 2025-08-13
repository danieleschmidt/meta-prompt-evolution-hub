#!/bin/bash
set -euo pipefail

# meta-prompt-evolution-hub Rollback Script

echo "🔄 Starting rollback for meta-prompt-evolution-hub..."

PROJECT_NAME="meta-prompt-evolution-hub"
NAMESPACE="production"

# Get previous version
PREVIOUS_VERSION=$(kubectl rollout history deployment/$PROJECT_NAME-deployment -n $NAMESPACE --revision=2 | grep -oP 'image=.*:\K[^\s]+' || echo "latest")

log() {
    echo -e "\033[0;32m[$(date +'%Y-%m-%d %H:%M:%S')]\033[0m $1"
}

error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1" >&2
}

log "Rolling back to previous version: $PREVIOUS_VERSION"

# Rollback deployment
kubectl rollout undo deployment/$PROJECT_NAME-deployment -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/$PROJECT_NAME-deployment -n $NAMESPACE --timeout=300s

# Verify rollback
if kubectl get pods -n $NAMESPACE -l app=$PROJECT_NAME | grep -q Running; then
    log "✅ Rollback completed successfully"
else
    error "❌ Rollback failed"
    exit 1
fi

log "🎉 Rollback completed!"
