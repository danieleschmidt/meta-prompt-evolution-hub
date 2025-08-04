# Meta-Prompt-Evolution-Hub Operational Runbook

## Overview
This runbook provides operational procedures for managing Meta-Prompt-Evolution-Hub in production.

## Daily Operations

### Health Monitoring
1. Check Grafana dashboard for system health
2. Review error rates and response times
3. Monitor resource utilization
4. Verify backup completion

### Log Review
```bash
# Check for errors in last 24 hours
kubectl logs --since=24h deployment/meta-prompt-evolution-hub -n production | grep ERROR

# Monitor real-time logs
kubectl logs -f deployment/meta-prompt-evolution-hub -n production
```

## Incident Response

### Service Unavailable (HTTP 503)
1. **Check pod status**
   ```bash
   kubectl get pods -n production -l app=meta-prompt-evolution-hub
   ```

2. **Check service endpoints**
   ```bash
   kubectl get endpoints -n production
   ```

3. **Check ingress configuration**
   ```bash
   kubectl describe ingress -n production
   ```

4. **Escalation**: If pods are healthy but service is unavailable, check ingress controller and load balancer

### High Error Rate (>5%)
1. **Check application logs**
   ```bash
   kubectl logs deployment/meta-prompt-evolution-hub -n production --tail=1000 | grep ERROR
   ```

2. **Check database connectivity**
   ```bash
   kubectl exec -it deployment/meta-prompt-evolution-hub -n production -- python -c "
   import psycopg2
   # Test database connection
   "
   ```

3. **Check external dependencies**
   - Redis availability
   - External API endpoints
   - Network connectivity

### High Response Time (>1s p95)
1. **Check resource utilization**
   - CPU usage approaching limits
   - Memory usage approaching limits
   - Disk I/O bottlenecks

2. **Database performance**
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   ```

3. **Cache performance**
   - Redis hit rate
   - Application cache statistics

### Pod Crashes (CrashLoopBackOff)
1. **Check pod logs**
   ```bash
   kubectl logs <pod-name> -n production --previous
   ```

2. **Check resource limits**
   - Memory limits (OOMKilled)
   - CPU throttling

3. **Check liveness/readiness probes**
   ```bash
   kubectl describe pod <pod-name> -n production
   ```

## Scaling Operations

### Manual Scaling
```bash
# Scale up during high traffic
kubectl scale deployment meta-prompt-evolution-hub --replicas=8 -n production

# Scale down during low traffic
kubectl scale deployment meta-prompt-evolution-hub --replicas=3 -n production
```

### Auto-scaling Configuration
```bash
# Check HPA status
kubectl get hpa -n production

# Modify auto-scaling thresholds
kubectl edit hpa meta-prompt-evolution-hub-hpa -n production
```

## Database Operations

### Backup Verification
```bash
# Check recent backups
ls -la /backups/db_backup_*.sql.gz

# Validate backup integrity
gunzip -t /backups/db_backup_latest.sql.gz
```

### Database Migration
```bash
# Apply database migrations
kubectl exec -it deployment/meta-prompt-evolution-hub -n production -- python manage.py migrate
```

### Database Performance Tuning
```sql
-- Monitor connection usage
SELECT count(*) as connections, state 
FROM pg_stat_activity 
GROUP BY state;

-- Check table sizes
SELECT schemaname,tablename,
  pg_size_pretty(size) as size,
  pg_size_pretty(total_size) as total_size
FROM (
  SELECT schemaname, tablename,
    pg_relation_size(schemaname||'.'||tablename) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as total_size
  FROM pg_tables
) as sizes
ORDER BY total_size DESC;
```

## Security Operations

### Certificate Management
```bash
# Check certificate expiration
kubectl get certificates -n production

# Force certificate renewal
kubectl delete certificate meta-prompt-evolution-tls -n production
```

### Security Scanning
```bash
# Run vulnerability scan
trivy image meta-prompt-evolution-hub:latest

# Check security policies
kubectl get psp,netpol -n production
```

## Performance Monitoring

### Key Metrics to Monitor
- **Request Rate**: >100 req/sec normal, >500 req/sec high load
- **Response Time**: <200ms p95 target, <500ms p95 acceptable
- **Error Rate**: <1% target, <5% acceptable
- **CPU Utilization**: <70% target, scale at >80%
- **Memory Utilization**: <80% target, investigate at >90%

### Performance Troubleshooting
1. **Identify bottlenecks**
   - CPU-bound: Optimize algorithms, scale horizontally
   - Memory-bound: Optimize caching, increase memory limits
   - I/O-bound: Optimize database queries, add read replicas

2. **Cache optimization**
   ```bash
   # Check cache hit rates
   redis-cli info stats
   ```

3. **Database optimization**
   ```sql
   -- Identify missing indexes
   SELECT schemaname, tablename, attname, n_distinct, correlation
   FROM pg_stats
   WHERE n_distinct > 100 AND correlation < 0.1;
   ```

## Disaster Recovery

### Backup Restoration
1. **Identify backup to restore**
   ```bash
   aws s3 ls s3://backup-bucket/db_backups/ | sort
   ```

2. **Restore database**
   ```bash
   # Download backup
   aws s3 cp s3://backup-bucket/db_backups/db_backup_20231201_120000.sql.gz .
   
   # Restore database
   gunzip -c db_backup_20231201_120000.sql.gz | psql $DATABASE_URL
   ```

3. **Verify data integrity**
   ```sql
   -- Check record counts
   SELECT 'prompts' as table_name, count(*) FROM prompts
   UNION ALL
   SELECT 'evaluations' as table_name, count(*) FROM evaluations;
   ```

### Service Recovery
1. **Redeploy application**
   ```bash
   kubectl rollout restart deployment/meta-prompt-evolution-hub -n production
   ```

2. **Verify service health**
   ```bash
   curl -f https://meta-prompt-evolution.example.com/health
   ```

3. **Monitor for issues**
   - Check error rates in Grafana
   - Monitor application logs
   - Verify all endpoints responding

## Maintenance Windows

### Planned Maintenance Checklist
1. **Pre-maintenance**
   - [ ] Schedule maintenance window
   - [ ] Notify stakeholders
   - [ ] Backup current state
   - [ ] Prepare rollback plan

2. **During maintenance**
   - [ ] Set maintenance mode
   - [ ] Apply updates/changes
   - [ ] Test functionality
   - [ ] Monitor for issues

3. **Post-maintenance**
   - [ ] Remove maintenance mode
   - [ ] Verify service health
   - [ ] Monitor metrics
   - [ ] Document changes

## Alerts and Escalation

### Alert Severity Levels
- **P1 (Critical)**: Service completely unavailable
- **P2 (High)**: Significant performance degradation
- **P3 (Medium)**: Minor issues, functionality impacted
- **P4 (Low)**: Informational, no immediate action needed

### Escalation Procedures
1. **P1 Incidents**: Immediate escalation to on-call engineer
2. **P2 Incidents**: Escalate within 15 minutes
3. **P3 Incidents**: Address within 1 hour
4. **P4 Incidents**: Address during business hours

### Contact Information
- **On-call Engineer**: [Contact details]
- **Development Team**: [Contact details]
- **Infrastructure Team**: [Contact details]
- **Management**: [Contact details]

## Useful Commands

### Kubernetes Debugging
```bash
# Get pod resource usage
kubectl top pods -n production

# Check pod events
kubectl get events -n production --sort-by='.lastTimestamp'

# Debug networking
kubectl exec -it <pod-name> -n production -- nslookup kubernetes.default.svc.cluster.local

# Check persistent volumes
kubectl get pv,pvc -n production
```

### Database Debugging
```bash
# Connect to database
kubectl exec -it deployment/meta-prompt-evolution-hub -n production -- psql $DATABASE_URL

# Check database size
kubectl exec -it deployment/meta-prompt-evolution-hub -n production -- du -sh /var/lib/postgresql/data
```

### Log Analysis
```bash
# Search for specific errors
kubectl logs deployment/meta-prompt-evolution-hub -n production | grep -i "timeout\|connection\|error"

# Export logs for analysis
kubectl logs deployment/meta-prompt-evolution-hub -n production --since=24h > app-logs.txt
```
