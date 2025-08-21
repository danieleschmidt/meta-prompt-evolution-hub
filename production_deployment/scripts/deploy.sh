#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_NAME="meta-prompt-evolution-hub"
VERSION="1.0.0"
NAMESPACE="production"

echo "🚀 Starting $PROJECT_NAME deployment..."

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check required tools
    for tool in kubectl docker; do
        if ! command -v $tool &> /dev/null; then
            echo -e "${RED}❌ $tool is required but not installed${NC}"
            exit 1
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Create namespace
create_namespace() {
    echo "📦 Creating namespace..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    echo -e "${GREEN}✅ Namespace ready${NC}"
}

# Build and tag image
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t $PROJECT_NAME:$VERSION .
    docker tag $PROJECT_NAME:$VERSION $PROJECT_NAME:latest
    echo -e "${GREEN}✅ Image built and tagged${NC}"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo "🚢 Deploying to Kubernetes..."
    
    # Apply manifests
    kubectl apply -f kubernetes/ -n $NAMESPACE
    
    # Wait for deployment
    echo "⏳ Waiting for deployment to be ready..."
    kubectl rollout status deployment/$PROJECT_NAME-deployment -n $NAMESPACE --timeout=300s
    
    echo -e "${GREEN}✅ Kubernetes deployment complete${NC}"
}

# Health checks
health_check() {
    echo "🏥 Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=$PROJECT_NAME -n $NAMESPACE --timeout=300s
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service $PROJECT_NAME-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [[ "$SERVICE_IP" != "pending" ]]; then
        echo -e "${GREEN}✅ Service available at: http://$SERVICE_IP${NC}"
    else
        echo -e "${YELLOW}⏳ Service IP pending (check: kubectl get svc -n $NAMESPACE)${NC}"
    fi
    
    echo -e "${GREEN}✅ Health checks passed${NC}"
}

# Monitor deployment
monitor_deployment() {
    echo "📊 Deployment monitoring commands:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo "  kubectl logs -f deployment/$PROJECT_NAME-deployment -n $NAMESPACE"
    echo "  kubectl get hpa -n $NAMESPACE"
    echo "  kubectl get service -n $NAMESPACE"
}

# Rollback function
rollback() {
    echo -e "${YELLOW}🔄 Rolling back deployment...${NC}"
    kubectl rollout undo deployment/$PROJECT_NAME-deployment -n $NAMESPACE
    kubectl rollout status deployment/$PROJECT_NAME-deployment -n $NAMESPACE
    echo -e "${GREEN}✅ Rollback completed${NC}"
}

# Main deployment
main() {
    echo "🎯 Meta Prompt Evolution Hub - Production Deployment"
    echo "=================================================="
    
    # Handle rollback
    if [[ "${1:-}" == "rollback" ]]; then
        rollback
        exit 0
    fi
    
    check_prerequisites
    create_namespace
    build_image
    deploy_kubernetes
    health_check
    monitor_deployment
    
    echo ""
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${GREEN}🚀 $PROJECT_NAME v$VERSION is now running in production${NC}"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        main rollback
        ;;
    "health")
        health_check
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health]"
        exit 1
        ;;
esac
