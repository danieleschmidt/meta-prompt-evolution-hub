# Meta-Prompt-Evolution-Hub Deployment Guide

## Prerequisites
- Kubernetes cluster
- kubectl configured

## Deployment Steps
1. Run: ./scripts/deploy.sh
2. Verify: kubectl get pods -n meta-prompt-hub
3. Test: curl http://<service-endpoint>/health

## Troubleshooting
- Check logs: kubectl logs -f deployment/meta-prompt-hub -n meta-prompt-hub
- Check events: kubectl get events -n meta-prompt-hub
