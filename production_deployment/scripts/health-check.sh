#!/bin/bash
# Health check script for meta-prompt-evolution-hub

NAMESPACE="production"
SERVICE_NAME="meta-prompt-evolution-hub-service"

echo "🏥 Running comprehensive health checks..."

# Check pod status
echo "📦 Checking pod status..."
kubectl get pods -l app=meta-prompt-evolution-hub -n $NAMESPACE

# Check service status
echo "🔗 Checking service status..."
kubectl get service $SERVICE_NAME -n $NAMESPACE

# Check HPA status
echo "📈 Checking auto-scaling status..."
kubectl get hpa -n $NAMESPACE

# Check recent events
echo "📋 Recent events..."
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10

echo "✅ Health check complete"
