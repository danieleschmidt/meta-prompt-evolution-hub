# Monitoring and Observability

This document outlines the monitoring and observability setup for Meta-Prompt-Evolution-Hub.

## Overview

The platform uses a comprehensive observability stack:

- **Metrics**: Prometheus for metrics collection and storage
- **Visualization**: Grafana for dashboards and alerting
- **Logging**: Structured logging with JSON output
- **Tracing**: OpenTelemetry for distributed tracing
- **Health Checks**: Built-in health endpoints

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│    Grafana      │
│                 │    │                 │    │                 │
│ - /metrics      │    │ - Metrics Store │    │ - Dashboards    │
│ - /health       │    │ - Alert Rules   │    │ - Notifications │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Structured     │    │  AlertManager   │
│  Logging        │    │                 │
│                 │    │ - PagerDuty     │
│ - JSON Format   │    │ - Slack         │
│ - Log Levels    │    │ - Email         │
└─────────────────┘    └─────────────────┘
```

## Key Metrics

### Application Metrics

#### HTTP Metrics
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration histogram
- `http_requests_in_flight`: Concurrent requests

#### Evolution Metrics
- `evolution_generations_completed_total`: Completed generations
- `evolution_best_fitness`: Current best fitness score
- `evolution_population_diversity`: Population diversity metric
- `prompt_evaluations_total`: Total prompt evaluations
- `prompt_evaluations_failed_total`: Failed evaluations
- `evaluation_duration_seconds`: Evaluation duration histogram

#### Business Metrics
- `prompt_fitness_score`: Individual prompt fitness scores
- `llm_api_calls_total`: Total LLM API calls
- `llm_api_cost_dollars`: API usage costs
- `user_sessions_active`: Active user sessions
- `ab_test_conversions_total`: A/B test conversions

### System Metrics

#### Application Performance
- `process_cpu_seconds_total`: CPU usage
- `process_resident_memory_bytes`: Memory usage
- `process_open_fds`: Open file descriptors
- `process_max_fds`: Maximum file descriptors

#### Database Metrics (via postgres_exporter)
- `pg_up`: Database availability
- `pg_stat_database_numbackends`: Active connections
- `pg_stat_database_xact_commit`: Transaction commits
- `pg_stat_database_blks_hit`: Buffer cache hits
- `pg_locks_count`: Active locks

#### Cache Metrics (via redis_exporter)
- `redis_up`: Redis availability
- `redis_memory_used_bytes`: Memory usage
- `redis_connected_clients`: Connected clients
- `redis_keyspace_hits_total`: Cache hits
- `redis_keyspace_misses_total`: Cache misses

## Dashboards

### Main Dashboard

Access at: `http://localhost:3000/d/meta-prompt-hub`

**Panels Include:**
- System health overview
- Evolution progress tracking
- Performance metrics
- Resource utilization
- Error rates and alerts

### Business Intelligence Dashboard

**Key Metrics:**
- Prompt quality trends
- Cost optimization insights
- User engagement metrics
- A/B test performance

### Infrastructure Dashboard

**Monitoring:**
- Server resource usage
- Database performance
- Cache hit ratios
- Network metrics

## Alerting

### Alert Configuration

Alerts are defined in `deployment/monitoring/alert_rules.yml`:

#### Critical Alerts
- **ApplicationDown**: Application unavailable
- **PostgresDown**: Database unavailable
- **RedisDown**: Cache unavailable
- **RayClusterDown**: Distributed computing unavailable

#### Warning Alerts
- **HighErrorRate**: >5% error rate for 5 minutes
- **HighLatency**: >2s 95th percentile latency
- **HighMemoryUsage**: >85% memory usage
- **LowPromptQuality**: Average fitness <0.5

### Notification Channels

Configure in Grafana:
- **Slack**: Development team notifications
- **PagerDuty**: On-call engineer alerts
- **Email**: Management summaries

## Health Checks

### Application Health Endpoints

#### Basic Health Check
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

#### Detailed Health Check
```bash
curl http://localhost:8080/health/detailed
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12
    },
    "redis": {
      "status": "healthy", 
      "response_time_ms": 3
    },
    "ray_cluster": {
      "status": "healthy",
      "nodes_active": 3,
      "workers_available": 24
    },
    "llm_apis": {
      "openai": "healthy",
      "anthropic": "healthy"
    }
  }
}
```

#### Readiness Check
```bash
curl http://localhost:8080/health/ready
```

Used by Kubernetes readiness probes.

#### Liveness Check
```bash
curl http://localhost:8080/health/live
```

Used by Kubernetes liveness probes.

## Logging

### Log Configuration

#### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)
```

#### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about application flow
- **WARNING**: Warning messages about potential issues
- **ERROR**: Error messages for handled exceptions
- **CRITICAL**: Critical errors requiring immediate attention

### Log Aggregation

#### Docker Compose
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

#### Kubernetes
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [INPUT]
        Name tail
        Path /var/log/containers/*.log
        Parser docker
        Tag kube.*
```

## Distributed Tracing

### OpenTelemetry Setup

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Trace Example
```python
@tracer.start_as_current_span("evaluate_prompt")
def evaluate_prompt(prompt, test_case):
    with tracer.start_as_current_span("llm_api_call"):
        response = llm_client.complete(prompt)
    
    with tracer.start_as_current_span("calculate_fitness"):
        fitness = calculate_fitness(response, test_case)
    
    return fitness
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

#### Evolution Performance
- **Generations per hour**: Evolution throughput
- **Average fitness improvement**: Quality of evolution
- **Time to convergence**: Speed of optimization

#### System Performance
- **Response time**: API endpoint latency
- **Throughput**: Requests per second
- **Resource utilization**: CPU, memory, disk usage

#### Business Performance
- **Cost per evaluation**: LLM API cost efficiency
- **Success rate**: Percentage of successful evaluations
- **User satisfaction**: Feedback scores

### Performance Benchmarking

```python
import time
import statistics
from typing import List

def benchmark_evaluation(prompt: str, test_cases: List[dict], iterations: int = 100):
    """Benchmark prompt evaluation performance."""
    durations = []
    
    for _ in range(iterations):
        start_time = time.time()
        evaluate_prompt(prompt, test_cases)
        duration = time.time() - start_time
        durations.append(duration)
    
    return {
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "p95": sorted(durations)[int(0.95 * len(durations))],
        "min": min(durations),
        "max": max(durations)
    }
```

## Monitoring Best Practices

### 1. Define SLIs and SLOs

#### Service Level Indicators (SLIs)
- **Availability**: Uptime percentage
- **Latency**: Response time percentiles
- **Quality**: Error rate percentage

#### Service Level Objectives (SLOs)
- **99.9% availability** over 30-day period
- **95th percentile latency < 2 seconds**
- **Error rate < 0.1%** for 24-hour period

### 2. Alert Hygiene

- **Actionable**: Every alert should require action
- **Contextualized**: Include relevant information
- **Prioritized**: Use severity levels appropriately
- **Documented**: Link to runbooks and remediation

### 3. Dashboard Design

- **Purpose-driven**: Each dashboard serves specific users
- **Layered**: Overview → Detail drill-down
- **Contextual**: Include relevant metadata
- **Accessible**: Consider colorblind users

### 4. Log Management

- **Structured**: Use JSON format consistently
- **Contextual**: Include request IDs and user IDs  
- **Searchable**: Use consistent field names
- **Retention**: Configure appropriate retention policies

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage by component
kubectl top pods
docker stats

# Check for memory leaks
py-spy dump --pid $(pgrep python)
```

#### Database Performance
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check connection usage
SELECT count(*) FROM pg_stat_activity;
```

#### Cache Performance
```bash
# Check Redis memory usage
redis-cli info memory

# Check cache hit ratio
redis-cli info stats | grep keyspace
```

### Monitoring Tools

#### Command Line Tools
```bash
# Check application metrics
curl -s http://localhost:8080/metrics | grep evolution

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets

# Test alerts
curl -s http://localhost:9090/api/v1/alerts
```

#### Grafana Queries

Most useful PromQL queries for troubleshooting:

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Memory usage
process_resident_memory_bytes / 1024 / 1024

# Database connections
pg_stat_database_numbackends
```

## Security Considerations

### Metrics Security
- **Authentication**: Secure Prometheus and Grafana endpoints
- **Authorization**: Role-based access to dashboards
- **Network**: Use TLS for metrics collection
- **Data**: Avoid exposing sensitive data in metrics

### Log Security
- **Sanitization**: Remove PII and secrets from logs  
- **Encryption**: Encrypt logs in transit and at rest
- **Access**: Implement proper access controls
- **Retention**: Follow data retention policies

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Monitoring Best Practices](https://sre.google/sre-book/monitoring-distributed-systems/)