#!/bin/bash
set -euo pipefail

# meta-prompt-evolution-hub Health Check Script

PROJECT_NAME="meta-prompt-evolution-hub"
NAMESPACE="production"

log() {
    echo -e "\033[0;32m[$(date +'%Y-%m-%d %H:%M:%S')]\033[0m $1"
}

error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1" >&2
}

warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check pod status
check_pods() {
    log "Checking pod status..."
    
    READY_PODS=$(kubectl get pods -n $NAMESPACE -l app=$PROJECT_NAME --field-selector=status.phase=Running | grep -c Ready || echo "0")
    TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -l app=$PROJECT_NAME | tail -n +2 | wc -l)
    
    if [ "$READY_PODS" -gt 0 ]; then
        log "✅ $READY_PODS/$TOTAL_PODS pods are ready"
    else
        error "❌ No pods are ready"
        return 1
    fi
}

# Check service endpoints
check_service() {
    log "Checking service endpoints..."
    
    ENDPOINTS=$(kubectl get endpoints $PROJECT_NAME-service -n $NAMESPACE -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
    
    if [ "$ENDPOINTS" -gt 0 ]; then
        log "✅ Service has $ENDPOINTS endpoints"
    else
        error "❌ Service has no endpoints"
        return 1
    fi
}

# Check application health
check_app_health() {
    log "Checking application health..."
    
    # Port forward for health check
    kubectl port-forward svc/$PROJECT_NAME-service 8080:80 -n $NAMESPACE &
    PF_PID=$!
    
    sleep 5
    
    # Check health endpoints
    if curl -f -s "http://localhost:8080/health/live" > /dev/null; then
        log "✅ Liveness check passed"
    else
        error "❌ Liveness check failed"
        kill $PF_PID 2>/dev/null || true
        return 1
    fi
    
    if curl -f -s "http://localhost:8080/health/ready" > /dev/null; then
        log "✅ Readiness check passed"
    else
        error "❌ Readiness check failed"
        kill $PF_PID 2>/dev/null || true
        return 1
    fi
    
    # Cleanup
    kill $PF_PID 2>/dev/null || true
}

# Main health check
main() {
    log "🩺 Running comprehensive health check for $PROJECT_NAME"
    
    CHECKS_PASSED=0
    TOTAL_CHECKS=3
    
    if check_pods; then
        ((CHECKS_PASSED++))
    fi
    
    if check_service; then
        ((CHECKS_PASSED++))
    fi
    
    if check_app_health; then
        ((CHECKS_PASSED++))
    fi
    
    log "Health check results: $CHECKS_PASSED/$TOTAL_CHECKS checks passed"
    
    if [ "$CHECKS_PASSED" -eq "$TOTAL_CHECKS" ]; then
        log "🎉 All health checks passed!"
        exit 0
    else
        error "❌ Some health checks failed"
        exit 1
    fi
}

main "$@"
