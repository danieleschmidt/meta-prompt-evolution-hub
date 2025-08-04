# Meta-Prompt-Evolution-Hub Deployment Guide

## Overview
This guide covers the deployment of Meta-Prompt-Evolution-Hub to production environments using Kubernetes and modern DevOps practices.

## Prerequisites
- Kubernetes cluster (v1.28+)
- Docker registry access
- Helm 3.x (optional but recommended)
- kubectl configured for cluster access
- Terraform (for infrastructure provisioning)

## Infrastructure Setup

### 1. Terraform Infrastructure
```bash
cd deployment/terraform
terraform init
terraform plan
terraform apply
```

### 2. Kubernetes Deployment

#### Using Helm (Recommended)
```bash
cd deployment
helm install meta-prompt-evolution-hub ./helm/meta-prompt-evolution-hub \
    --namespace production \
    --create-namespace \
    --set image.tag=latest
```

#### Using kubectl
```bash
kubectl apply -f deployment/kubernetes/ -n production
```

### 3. Verify Deployment
```bash
kubectl get pods -n production
kubectl get services -n production
kubectl get ingress -n production
```

## Configuration

### Environment Variables
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `ENVIRONMENT`: Environment name (production, staging, development)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `PROMETHEUS_ENABLED`: Enable Prometheus metrics (true/false)

### Resource Requirements
- **Minimum**: 0.5 CPU, 1Gi Memory
- **Recommended**: 1 CPU, 2Gi Memory
- **Auto-scaling**: 2-10 replicas based on CPU utilization

## Monitoring and Observability

### Prometheus Metrics
Available at `/metrics` endpoint:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration histogram
- `evolution_evaluations_total`: Total evolution evaluations
- `cache_hits_total`: Cache hit counter

### Grafana Dashboards
Pre-configured dashboards available in `deployment/monitoring/grafana/`

### Distributed Tracing
Jaeger tracing configured for request flow analysis.

## Security

### TLS Configuration
- Automatic certificate management with cert-manager
- HTTPS enforced for all external traffic
- Internal service mesh with mTLS

### RBAC
- Dedicated service account with minimal required permissions
- Network policies restricting pod-to-pod communication
- Pod security policies enforcing security standards

## Troubleshooting

### Common Issues

1. **Pod not starting**
   ```bash
   kubectl describe pod <pod-name> -n production
   kubectl logs <pod-name> -n production
   ```

2. **Service unavailable**
   ```bash
   kubectl get endpoints -n production
   kubectl port-forward service/meta-prompt-evolution-hub-service 8080:80 -n production
   ```

3. **Database connection issues**
   - Verify database credentials in secrets
   - Check network policies and security groups
   - Verify database availability

### Health Checks
- **Liveness**: `/health` endpoint
- **Readiness**: `/ready` endpoint
- **Metrics**: `/metrics` endpoint

## Backup and Recovery

### Database Backups
Automated daily backups configured via CronJob:
```bash
kubectl apply -f deployment/backup/cronjob.yaml
```

### Manual Backup
```bash
./deployment/backup/backup.sh
```

### Recovery Procedures
1. Restore from latest backup
2. Verify data integrity
3. Update DNS/load balancer
4. Monitor application health

## CI/CD Integration

### GitHub Actions
Pipeline configured in `.github/workflows/ci-cd.yml`
- Automated testing and quality gates
- Docker image building and pushing
- Deployment to staging and production

### Jenkins
Alternative pipeline available in `deployment/jenkins/Jenkinsfile`

## Performance Tuning

### Resource Optimization
- Monitor CPU and memory usage via Grafana
- Adjust resource requests/limits based on actual usage
- Configure appropriate JVM heap sizes

### Caching Strategy
- Redis for distributed caching
- Application-level caching for frequently accessed data
- CDN for static assets

### Database Optimization
- Connection pooling configuration
- Query optimization and indexing
- Read replicas for read-heavy workloads

## Support and Maintenance

### Log Analysis
```bash
# View application logs
kubectl logs -f deployment/meta-prompt-evolution-hub -n production

# View logs from specific time range
kubectl logs --since=1h deployment/meta-prompt-evolution-hub -n production
```

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment meta-prompt-evolution-hub --replicas=5 -n production

# Check auto-scaling status
kubectl get hpa -n production
```

### Updates and Rollbacks
```bash
# Rolling update
kubectl set image deployment/meta-prompt-evolution-hub container=image:new-tag -n production

# Rollback to previous version
kubectl rollout undo deployment/meta-prompt-evolution-hub -n production
```
