#!/bin/bash
set -euo pipefail

echo "🚀 Starting Meta Prompt Evolution Hub deployment..."

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    for tool in kubectl helm terraform docker aws; do
        if ! command -v $tool &> /dev/null; then
            echo "❌ $tool is required but not installed"
            exit 1
        fi
    done
    echo "✅ Prerequisites check passed"
}

# Deploy infrastructure  
deploy_infrastructure() {
    echo "Deploying infrastructure..."
    cd terraform/
    terraform init
    terraform plan -out=tfplan
    terraform apply tfplan
    cd ..
    echo "✅ Infrastructure deployed"
}

# Build and push image
build_image() {
    echo "Building Docker image..."
    docker build -t meta-prompt-evolution-hub:1.0.0 .
    echo "✅ Image built"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo "Deploying to Kubernetes..."
    kubectl create namespace production --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f kubernetes/
    kubectl rollout status deployment/meta-prompt-evolution-hub-deployment -n production
    echo "✅ Kubernetes deployment complete"
}

# Health checks
health_check() {
    echo "Running health checks..."
    kubectl wait --for=condition=ready pod -l app=meta-prompt-evolution-hub -n production --timeout=300s
    echo "✅ Health checks passed"
}

# Main deployment
main() {
    check_prerequisites
    deploy_infrastructure  
    build_image
    deploy_kubernetes
    health_check
    echo "🎉 Deployment completed successfully!"
}

main "$@"
