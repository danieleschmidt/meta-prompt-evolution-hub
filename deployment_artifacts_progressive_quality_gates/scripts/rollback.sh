#!/bin/bash
set -e

# Progressive Quality Gates Rollback Script
echo "🔄 Starting Progressive Quality Gates Rollback"

NAMESPACE="quality-gates"
APP_NAME="progressive-quality-gates"

# Get current deployment
echo "📊 Current deployment status:"
kubectl get deployment $APP_NAME -n $NAMESPACE

# Rollback to previous version
echo "⏪ Rolling back to previous version"
kubectl rollout undo deployment/$APP_NAME -n $NAMESPACE

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE

# Verify health
echo "🔍 Verifying health after rollback"
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

echo "✅ Rollback completed successfully"
