# Deployment Guide

This guide covers different deployment strategies for Meta-Prompt-Evolution-Hub.

## Quick Start

### Local Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Development with Hot Reload

```bash
# Use development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Access services
# - Application: http://localhost:8080
# - Ray Dashboard: http://localhost:8265
# - Grafana: http://localhost:3000 (admin/admin123)
# - Prometheus: http://localhost:9090
```

## Production Deployment

### Docker Compose Production

```bash
# Use production profile
docker-compose --profile production up -d

# With custom environment
ENV=production docker-compose --profile production up -d
```

### Build Production Image

```bash
# Build production image
docker build -f Dockerfile.prod -t meta-prompt-hub:latest .

# Run production container
docker run -d \
  --name meta-prompt-hub \
  -p 8080:8080 \
  -e DATABASE_URL="postgresql://..." \
  -e REDIS_URL="redis://..." \
  meta-prompt-hub:latest
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl and helm
brew install kubectl helm  # macOS
# or
apt-get install kubectl helm  # Ubuntu

# Add Helm repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### Deploy Dependencies

```bash
# PostgreSQL
helm install postgres bitnami/postgresql \
  --set auth.postgresPassword="secure-password" \
  --set auth.database="meta_prompt_hub"

# Redis  
helm install redis bitnami/redis \
  --set auth.password="secure-redis-password"

# Prometheus & Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack
```

### Deploy Application

```bash
# Create namespace
kubectl create namespace meta-prompt-hub

# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/ -n meta-prompt-hub

# Check deployment status
kubectl get pods -n meta-prompt-hub
kubectl logs -f deployment/meta-prompt-hub -n meta-prompt-hub
```

## Cloud Deployments

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name meta-prompt-hub --region us-west-2

# Deploy with Helm
helm install meta-prompt-hub ./deployment/helm/meta-prompt-evolution-hub/ \
  --set image.tag=latest \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=meta-prompt.your-domain.com
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create meta-prompt-hub \
  --zone us-central1-a \
  --num-nodes 3

# Deploy application
kubectl apply -f deployment/kubernetes/
```

### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name meta-prompt-hub \
  --node-count 3 \
  --enable-addons monitoring

# Deploy with Helm
helm install meta-prompt-hub ./deployment/helm/meta-prompt-evolution-hub/
```

## Environment Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
POSTGRES_DB=meta_prompt_hub
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password

# Cache
REDIS_URL=redis://host:6379/0
REDIS_PASSWORD=secure_redis_password

# Distributed Computing
RAY_ADDRESS=ray://ray-head:10001
RAY_DASHBOARD_HOST=0.0.0.0
RAY_DASHBOARD_PORT=8265

# LLM APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Security
SECRET_KEY=your-secret-key-32-chars-long
JWT_EXPIRATION_HOURS=24

# Monitoring
PROMETHEUS_METRICS_PORT=9090
SENTRY_DSN=your-sentry-dsn
```

### Optional Configuration

```bash
# Performance
MAX_CONCURRENT_EVALUATIONS=100
EVALUATION_TIMEOUT_SECONDS=30
WORKER_CONCURRENCY=4

# Features
DEBUG=false
TESTING=false
LOG_LEVEL=INFO

# Backup
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
DATA_RETENTION_DAYS=365
```

## Scaling Configuration

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: meta-prompt-hub-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: meta-prompt-hub
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Ray Cluster Scaling

```bash
# Scale Ray workers
docker-compose up --scale ray-worker=5

# Or in Kubernetes
kubectl scale deployment ray-worker --replicas=5
```

## Health Checks

### Application Health

```bash
# Health endpoint
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/health/detailed
```

### Service Health

```bash
# PostgreSQL
pg_isready -h localhost -p 5432 -U postgres

# Redis
redis-cli -h localhost -p 6379 ping

# Ray
curl http://localhost:8265/api/overview
```

## Monitoring Setup

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'meta-prompt-hub'
    static_configs:
      - targets: ['app:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboards

```bash
# Import dashboard
grafana-cli plugins install grafana-piechart-panel
curl -X POST \
  http://admin:admin123@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @deployment/monitoring/grafana/dashboard.json
```

## Security Considerations

### Container Security

```dockerfile
# Use non-root user
USER 1000:1000

# Read-only root filesystem
--read-only --tmpfs /tmp
```

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: meta-prompt-hub-netpol
spec:
  podSelector:
    matchLabels:
      app: meta-prompt-hub
  policyTypes:
  - Ingress
  - Egress
```

### Secrets Management

```bash
# Kubernetes secrets
kubectl create secret generic meta-prompt-hub-secrets \
  --from-literal=database-password=secure-password \
  --from-literal=api-keys=json-string
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
pg_dump $DATABASE_URL | gzip > backup-$(date +%Y%m%d).sql.gz

# Upload to cloud storage
aws s3 cp backup-$(date +%Y%m%d).sql.gz s3://your-backup-bucket/
```

### Application State Backup

```bash
# Backup Ray cluster state
ray memory --stats-only > ray-state-backup.json

# Backup Redis data
redis-cli --rdb backup.rdb
```

## Troubleshooting

### Common Issues

```bash
# Container won't start
docker logs meta-prompt-hub
kubectl logs deployment/meta-prompt-hub

# Database connection issues
nc -zv postgres 5432
kubectl exec -it postgres-pod -- psql -U postgres

# Ray cluster issues  
ray status
kubectl logs ray-head-pod
```

### Performance Issues

```bash
# Check resource usage
docker stats
kubectl top nodes
kubectl top pods

# Profile application
py-spy top --pid $(pgrep python)
```

## Maintenance

### Updates

```bash
# Rolling update in Kubernetes
kubectl set image deployment/meta-prompt-hub app=meta-prompt-hub:v2.0.0

# Zero-downtime update with Docker Compose
docker-compose up -d --force-recreate --no-deps app
```

### Database Migrations

```bash
# Run migrations
docker exec meta-prompt-hub python -m alembic upgrade head

# Or in Kubernetes
kubectl exec deployment/meta-prompt-hub -- python -m alembic upgrade head
```