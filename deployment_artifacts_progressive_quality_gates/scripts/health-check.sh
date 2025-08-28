#!/bin/bash

# Progressive Quality Gates Health Check Script
NAMESPACE="quality-gates"
APP_NAME="progressive-quality-gates"

echo "🏥 Progressive Quality Gates Health Check"
echo "========================================"

# Check deployment status
echo "📊 Deployment Status:"
kubectl get deployment $APP_NAME -n $NAMESPACE

# Check pod status
echo ""
echo "🧩 Pod Status:"
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

# Check service status
echo ""
echo "🌐 Service Status:"
kubectl get service -n $NAMESPACE

# Check HPA status
echo ""
echo "📈 HPA Status:"
kubectl get hpa -n $NAMESPACE

# Test endpoints via port-forward
echo ""
echo "🔍 Testing Health Endpoints:"
kubectl port-forward -n $NAMESPACE svc/$APP_NAME-service 8080:80 &
PROXY_PID=$!
sleep 3

echo "Testing /health/live..."
curl -s http://localhost:8080/health/live | jq .

echo "Testing /health/ready..."
curl -s http://localhost:8080/health/ready | jq .

# Kill port-forward
kill $PROXY_PID

echo ""
echo "✅ Health check completed"
