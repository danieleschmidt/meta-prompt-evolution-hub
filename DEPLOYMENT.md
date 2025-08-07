# 🚀 Deployment Guide - Sentiment Analyzer Pro

This guide covers all deployment options from local development to enterprise Kubernetes clusters.

## 📋 Prerequisites

### System Requirements
- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Memory**: Minimum 4GB, Recommended 8GB+
- **Storage**: 10GB+ available space
- **Network**: Internet access for dependencies
- **OS**: Linux (Ubuntu/RHEL), macOS, Windows with WSL

### Software Dependencies
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.20+ (for K8s deployment)
- **Python**: 3.9+ (for local development)
- **Nginx**: 1.18+ (for reverse proxy)

## 🏃 Quick Deployment Options

### Option 1: Docker Compose (Recommended)
```bash
# Clone repository
git clone https://github.com/danieleschmidt/sentiment-analyzer-pro
cd sentiment-analyzer-pro

# Start all services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

**Services Started:**
- 🎯 Sentiment Analyzer API: `http://localhost:8000`
- 🏥 Health Checks: `http://localhost:8080`
- 📊 Metrics: `http://localhost:9090`
- 🗄️ Redis Cache: `localhost:6379`
- 📈 Grafana Dashboard: `http://localhost:3000` (admin/admin123)

### Option 2: Standalone Docker
```bash
# Build image
docker build -t sentiment-analyzer-pro .

# Run container
docker run -d \
  --name sentiment-analyzer \
  -p 8000:8000 \
  -p 8080:8080 \
  -e CACHE_SIZE=10000 \
  -e MIN_WORKERS=4 \
  sentiment-analyzer-pro

# Check logs
docker logs sentiment-analyzer
```

### Option 3: Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv sentiment-env
source sentiment-env/bin/activate  # Linux/Mac
# sentiment-env\Scripts\activate  # Windows

# Install dependencies (optional, works without)
pip install numpy pandas scikit-learn

# Run directly
python3 production_api_server.py
```

## 🌐 Production Kubernetes Deployment

### Step 1: Prepare Kubernetes Cluster
```bash
# Verify cluster access
kubectl cluster-info

# Create namespace
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-analyzer
  labels:
    app: sentiment-analyzer-pro
EOF
```

### Step 2: Configure Secrets and ConfigMaps
```bash
# Create secrets (replace with actual values)
kubectl create secret generic sentiment-analyzer-secrets \
  --namespace=sentiment-analyzer \
  --from-literal=API_KEY="your-api-key" \
  --from-literal=REDIS_PASSWORD="your-redis-password"

# Apply configuration
kubectl apply -f production_deployment.yaml
```

### Step 3: Deploy Application
```bash
# Deploy all components
kubectl apply -f production_deployment.yaml

# Verify deployment
kubectl get pods -n sentiment-analyzer
kubectl get services -n sentiment-analyzer

# Check application logs
kubectl logs -f deployment/sentiment-analyzer -n sentiment-analyzer
```

### Step 4: Set Up Ingress (Optional)
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Update production_deployment.yaml with your domain
# Then apply
kubectl apply -f production_deployment.yaml
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Main API port |
| `HEALTH_CHECK_PORT` | 8080 | Health check port |
| `PROMETHEUS_PORT` | 9090 | Metrics port |
| `POPULATION_SIZE` | 100 | Evolutionary population size |
| `CACHE_SIZE` | 50000 | Cache size (entries) |
| `MIN_WORKERS` | 8 | Minimum worker threads |
| `MAX_WORKERS` | 32 | Maximum worker threads |
| `RATE_LIMIT_RPM` | 10000 | Rate limit (requests/minute) |
| `REDIS_URL` | redis://localhost:6379 | Redis connection |
| `LOG_LEVEL` | INFO | Logging level |
| `ENABLE_CACHING` | true | Enable caching |
| `ENABLE_MONITORING` | true | Enable metrics |

### Docker Compose Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  sentiment-analyzer:
    environment:
      - POPULATION_SIZE=200
      - CACHE_SIZE=100000
      - MAX_WORKERS=64
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4.0'
```

### Kubernetes Resource Limits
```yaml
# In production_deployment.yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "4000m"
  requests:
    memory: "2Gi" 
    cpu: "2000m"
```

## 📊 Monitoring and Observability

### Health Checks
```bash
# Basic health check
curl http://your-domain/health

# Detailed health with metrics
curl http://your-domain:8080/health

# Readiness probe
curl http://your-domain:8080/ready
```

### Prometheus Metrics
```bash
# Access metrics endpoint
curl http://your-domain:8080/metrics

# Key metrics to monitor:
# - sentiment_analyzer_requests_total
# - sentiment_analyzer_error_rate
# - sentiment_analyzer_processing_time
# - sentiment_analyzer_uptime_seconds
```

### Grafana Dashboards
1. Access Grafana: `http://localhost:3000`
2. Login: admin/admin123
3. Import dashboard from `monitoring/grafana/dashboards/`

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Or use Let's Encrypt (production)
certbot certonly --nginx -d your-domain.com
```

### Network Security
```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sentiment-analyzer-netpol
spec:
  podSelector:
    matchLabels:
      app: sentiment-analyzer-pro
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nginx-ingress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
```

### Security Best Practices
1. **Run as non-root user**: ✅ Built into Docker image
2. **Read-only filesystem**: ✅ Configured in K8s deployment
3. **Security contexts**: ✅ Applied in all manifests
4. **Network policies**: ✅ Restrict pod-to-pod communication
5. **Resource limits**: ✅ Prevent resource exhaustion
6. **Secret management**: ✅ Use Kubernetes secrets

## ⚡ Performance Tuning

### High-Traffic Configuration
```yaml
# For 100K+ requests/minute
environment:
  POPULATION_SIZE: 200
  CACHE_SIZE: 500000
  MIN_WORKERS: 16
  MAX_WORKERS: 128
  RATE_LIMIT_RPM: 100000

# Kubernetes HPA
spec:
  minReplicas: 10
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

### Redis Optimization
```yaml
# redis configuration
command: 
  - redis-server
  - --maxmemory 8gb
  - --maxmemory-policy allkeys-lru
  - --tcp-keepalive 60
  - --timeout 0
```

### NGINX Optimization
```nginx
# In nginx.conf
worker_processes auto;
worker_connections 8192;
keepalive_timeout 65;
keepalive_requests 10000;

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:100m rate=1000r/m;
```

## 🐛 Troubleshooting

### Common Issues

#### 1. High Memory Usage
```bash
# Check memory usage
kubectl top pods -n sentiment-analyzer

# Solutions:
# - Reduce CACHE_SIZE
# - Adjust POPULATION_SIZE
# - Increase memory limits
```

#### 2. Slow Response Times
```bash
# Check processing metrics
curl http://your-domain:8080/metrics | grep processing_time

# Solutions:
# - Enable Redis caching
# - Increase MIN_WORKERS
# - Optimize evolutionary parameters
```

#### 3. Pod Restarts
```bash
# Check restart reasons
kubectl describe pod <pod-name> -n sentiment-analyzer

# Common causes:
# - OOMKilled: Increase memory limits
# - Liveness probe failed: Adjust probe timing
# - Image pull errors: Check image availability
```

#### 4. Connection Issues
```bash
# Test connectivity
kubectl port-forward service/sentiment-analyzer-service 8000:80 -n sentiment-analyzer
curl http://localhost:8000/health

# Check service endpoints
kubectl get endpoints -n sentiment-analyzer
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python3 production_api_server.py

# Or in Docker
docker run -e LOG_LEVEL=DEBUG sentiment-analyzer-pro
```

### Performance Profiling
```bash
# Run load test
python3 -c "
import requests
import time
import concurrent.futures

def test_request():
    response = requests.post('http://localhost:8000/analyze', 
                           json={'text': 'Test sentiment'})
    return response.status_code == 200

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(test_request) for _ in range(1000)]
    results = [f.result() for f in futures]

duration = time.time() - start
success_rate = sum(results) / len(results)
print(f'Processed 1000 requests in {duration:.2f}s')
print(f'Success rate: {success_rate:.2%}')
print(f'Throughput: {1000/duration:.1f} req/s')
"
```

## 📈 Scaling Guidelines

### Vertical Scaling
```yaml
# Increase resources for single instance
resources:
  limits:
    memory: "8Gi"
    cpu: "8000m"
  requests:
    memory: "4Gi"
    cpu: "4000m"

# Environment tuning
environment:
  MAX_WORKERS: 64
  CACHE_SIZE: 1000000
  POPULATION_SIZE: 500
```

### Horizontal Scaling
```yaml
# Auto-scaling configuration
spec:
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: sentiment_analyzer_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### Multi-Region Deployment
```bash
# Deploy to multiple regions
for region in us-east-1 us-west-2 eu-west-1; do
  kubectl apply -f production_deployment.yaml --context=$region
done

# Set up cross-region load balancing
# (Implementation depends on cloud provider)
```

## 🎯 Production Checklist

### Before Deployment
- [ ] **Security**: Secrets configured, network policies applied
- [ ] **Resources**: CPU/memory limits set appropriately
- [ ] **Monitoring**: Health checks, metrics, alerting configured
- [ ] **Backup**: Redis persistence enabled if needed
- [ ] **DNS**: Domain names and SSL certificates ready
- [ ] **Load Testing**: Performance validated under expected load

### After Deployment
- [ ] **Health Check**: All endpoints responding correctly
- [ ] **Metrics**: Prometheus scraping metrics successfully
- [ ] **Logs**: Application logs flowing to aggregation system
- [ ] **Alerts**: Monitoring alerts configured and tested
- [ ] **Documentation**: Runbooks and incident procedures ready
- [ ] **Scaling**: HPA functioning and scaling appropriately

## 📞 Support and Maintenance

### Log Collection
```bash
# Kubernetes logs
kubectl logs -f deployment/sentiment-analyzer -n sentiment-analyzer --tail=100

# Container logs
docker logs sentiment-analyzer --tail=100 -f

# Application logs
tail -f /app/logs/api.log
```

### Backup and Recovery
```bash
# Redis data backup (if persistence enabled)
kubectl exec -it redis-pod -n sentiment-analyzer -- redis-cli BGSAVE

# Configuration backup
kubectl get all -n sentiment-analyzer -o yaml > sentiment-analyzer-backup.yaml
```

### Updates and Rollbacks
```bash
# Rolling update
kubectl set image deployment/sentiment-analyzer \
  sentiment-analyzer=sentiment-analyzer-pro:v2.0.0 \
  -n sentiment-analyzer

# Rollback if needed
kubectl rollout undo deployment/sentiment-analyzer -n sentiment-analyzer

# Check rollout status
kubectl rollout status deployment/sentiment-analyzer -n sentiment-analyzer
```

---

## 🆘 Emergency Procedures

### Service Down
1. Check health endpoint: `curl http://your-domain/health`
2. Review recent logs for errors
3. Check resource usage (CPU/memory)
4. Restart service if necessary
5. Scale up replicas if under high load

### High Error Rate
1. Check metrics: `curl http://your-domain/metrics`
2. Review error logs for patterns
3. Verify external dependencies (Redis)
4. Consider rolling back recent changes
5. Enable debug logging for investigation

### Performance Degradation
1. Monitor processing time metrics
2. Check cache hit rates
3. Review resource utilization
4. Scale workers or replicas as needed
5. Optimize evolutionary parameters if necessary

For additional support, consult the API documentation or create an issue in the GitHub repository.