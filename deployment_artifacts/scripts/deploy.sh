#!/bin/bash
set -euo pipefail

# meta-prompt-evolution-hub Production Deployment Script
# Version: 1.0.0
# Environment: production

echo "🚀 Starting meta-prompt-evolution-hub deployment..."

# Configuration
PROJECT_NAME="meta-prompt-evolution-hub"
VERSION="1.0.0"
ENVIRONMENT="production"
PRIMARY_REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if required tools are installed
    for tool in kubectl helm terraform docker aws; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
        exit 1
    fi
    
    # Check kubectl context
    if ! kubectl cluster-info &> /dev/null; then
        error "kubectl not connected to cluster"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd terraform/
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply deployment
    terraform apply tfplan
    
    # Get outputs
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    DB_ENDPOINT=$(terraform output -raw database_endpoint)
    REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
    
    log "Infrastructure deployment completed"
    cd ..
}

# Build and push Docker image
build_and_push_image() {
    log "Building and pushing Docker image..."
    
    # Build image
    docker build -t $PROJECT_NAME:$VERSION .
    docker build -t $PROJECT_NAME:latest .
    
    # Tag for ECR
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$PRIMARY_REGION.amazonaws.com"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $PROJECT_NAME --region $PRIMARY_REGION || \
    aws ecr create-repository --repository-name $PROJECT_NAME --region $PRIMARY_REGION
    
    # Get ECR login
    aws ecr get-login-password --region $PRIMARY_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    # Tag and push
    docker tag $PROJECT_NAME:$VERSION $ECR_REGISTRY/$PROJECT_NAME:$VERSION
    docker tag $PROJECT_NAME:latest $ECR_REGISTRY/$PROJECT_NAME:latest
    
    docker push $ECR_REGISTRY/$PROJECT_NAME:$VERSION
    docker push $ECR_REGISTRY/$PROJECT_NAME:latest
    
    log "Image build and push completed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace production --dry-run=client -o yaml | kubectl apply -f -
    
    # Update kubeconfig
    aws eks update-kubeconfig --region $PRIMARY_REGION --name $CLUSTER_NAME
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/deployment.yaml
    kubectl apply -f kubernetes/service.yaml
    kubectl apply -f kubernetes/hpa.yaml
    kubectl apply -f kubernetes/ingress.yaml
    
    # Apply security policies
    kubectl apply -f security/network-policy.yaml
    kubectl apply -f security/rbac-role.yaml
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/$PROJECT_NAME-deployment -n production --timeout=300s
    
    log "Kubernetes deployment completed"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false
    
    # Install Grafana dashboard
    kubectl create configmap grafana-dashboard \
        --from-file=monitoring/grafana_dashboard.json \
        -n monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    log "Monitoring deployment completed"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=$PROJECT_NAME -n production --timeout=300s
    
    # Check service endpoints
    EXTERNAL_IP=$(kubectl get service $PROJECT_NAME-service -n production -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [ -n "$EXTERNAL_IP" ]; then
        # Test health endpoint
        for i in {1..10}; do
            if curl -f -s "http://$EXTERNAL_IP/health/live" > /dev/null; then
                log "Health check passed"
                break
            fi
            if [ $i -eq 10 ]; then
                error "Health check failed after 10 attempts"
                exit 1
            fi
            sleep 10
        done
    else
        warning "External IP not ready, skipping external health check"
    fi
    
    log "Health checks completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f terraform/tfplan
}

# Main deployment flow
main() {
    log "Starting production deployment for $PROJECT_NAME v$VERSION"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    deploy_infrastructure
    build_and_push_image
    deploy_to_kubernetes
    deploy_monitoring
    run_health_checks
    
    log "🎉 Deployment completed successfully!"
    log "Application should be available at: https://meta-prompt-hub.com"
}

# Run main function
main "$@"
