# 📡 API Documentation - Sentiment Analyzer Pro

Complete RESTful API reference for the Sentiment Analyzer Pro service.

## 🔗 Base URL

```
Production: https://sentiment-api.yourdomain.com
Development: http://localhost:8000
Health Checks: http://localhost:8080
```

## 🗺️ API Endpoints Overview

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/analyze` | POST | Single text analysis | 100/min |
| `/batch` | POST | Batch text analysis | 10/min |
| `/health` | GET | Service health status | 60/min |
| `/ready` | GET | Readiness probe | Unlimited |
| `/metrics` | GET | Prometheus metrics | 60/min |
| `/` | GET | API information | Unlimited |

## 📝 Authentication

Currently, the API supports:
- **Open Access**: No authentication required for basic usage
- **Rate Limiting**: Based on client IP address
- **API Key** (Optional): Pass in `X-API-Key` header for higher limits

```bash
# With API key
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/analyze
```

## 📊 Single Text Analysis

### `POST /analyze`

Analyze sentiment of a single text string.

#### Request
```json
{
  "text": "I absolutely love this amazing product!"
}
```

#### Request Headers
| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | ✅ | `application/json` |
| `X-Request-ID` | ❌ | Custom request identifier for tracking |
| `X-API-Key` | ❌ | API key for authentication |

#### Response (Success)
```json
{
  "success": true,
  "data": {
    "text": "I absolutely love this amazing product!",
    "label": "positive",
    "confidence": 0.89,
    "processing_time": 0.023,
    "prompt_used": "gen3_optimized_42",
    "model_used": "evolutionary-optimized",
    "from_cache": false
  },
  "timestamp": 1703123456.789,
  "processing_time": 0.025,
  "request_id": "req_1703123456.789_a1b2c3d4",
  "version": "1.0.0"
}
```

#### Response (Error)
```json
{
  "success": false,
  "error": "Text too long (max 10000 characters)",
  "timestamp": 1703123456.789,
  "processing_time": 0.001,
  "request_id": "req_1703123456.789_a1b2c3d4",
  "version": "1.0.0"
}
```

#### Status Codes
- `200 OK`: Analysis successful
- `400 Bad Request`: Invalid input (empty text, too long, etc.)
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Service error

#### cURL Example
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: custom-123" \
  -d '{"text": "This product is amazing and I love it!"}'
```

#### Python Example
```python
import requests

response = requests.post('http://localhost:8000/analyze', 
                        json={'text': 'Great service and fast delivery!'})
result = response.json()

if result['success']:
    data = result['data']
    print(f"Sentiment: {data['label']} ({data['confidence']:.2f})")
else:
    print(f"Error: {result['error']}")
```

#### JavaScript Example
```javascript
fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Request-ID': 'web-client-123'
  },
  body: JSON.stringify({
    text: 'Outstanding quality and excellent service!'
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log(`Sentiment: ${data.data.label} (${data.data.confidence})`);
  } else {
    console.error(`Error: ${data.error}`);
  }
});
```

## 📦 Batch Text Analysis

### `POST /batch`

Analyze sentiment of multiple texts in a single request.

#### Request
```json
{
  "texts": [
    "I love this product!",
    "Terrible customer service",
    "It's an okay experience",
    "Outstanding quality!",
    "Very disappointed with the purchase"
  ]
}
```

#### Response (Success)
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "I love this product!",
        "label": "positive",
        "confidence": 0.92
      },
      {
        "text": "Terrible customer service",
        "label": "negative",
        "confidence": 0.88
      },
      {
        "text": "It's an okay experience",
        "label": "neutral",
        "confidence": 0.67
      },
      {
        "text": "Outstanding quality!",
        "label": "positive",
        "confidence": 0.95
      },
      {
        "text": "Very disappointed with the purchase",
        "label": "negative",
        "confidence": 0.91
      }
    ],
    "total_texts": 5,
    "successful": 5
  },
  "timestamp": 1703123456.789,
  "processing_time": 0.156,
  "request_id": "req_batch_1703123456.789",
  "version": "1.0.0"
}
```

#### Batch Limits
- **Maximum texts per batch**: 100
- **Individual text length**: 10,000 characters
- **Total request size**: 1MB
- **Processing timeout**: 60 seconds

#### cURL Example
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product!",
      "Poor quality",
      "Average experience"
    ]
  }'
```

#### Python Batch Example
```python
import requests

texts = [
    "Excellent service and fast delivery!",
    "Product quality is disappointing",
    "Average experience, nothing special",
    "Highly recommend this to everyone!",
    "Customer support was unhelpful"
]

response = requests.post('http://localhost:8000/batch', 
                        json={'texts': texts})
result = response.json()

if result['success']:
    for i, analysis in enumerate(result['data']['results']):
        print(f"Text {i+1}: {analysis['label']} ({analysis['confidence']:.2f})")
    print(f"Batch processed: {result['data']['successful']}/{result['data']['total_texts']}")
```

## 🏥 Health and Status Endpoints

### `GET /health`

Comprehensive health check with detailed metrics.

#### Response
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 86400.5,
    "analyzer_type": "scalable",
    "requests_total": 15420,
    "error_rate": 0.002,
    "avg_processing_time": 0.045,
    "issues": [],
    "analyzer_health": {
      "status": "healthy",
      "health_score": 95,
      "metrics": {
        "cache_hit_rate": 0.78,
        "throughput_per_second": 125.5,
        "memory_usage_mb": 1248.3,
        "active_workers": 12
      }
    }
  },
  "timestamp": 1703123456.789
}
```

#### Status Values
- `healthy`: All systems operating normally
- `degraded`: Some issues but service functional
- `starting`: Service recently started, warming up
- `unhealthy`: Critical issues affecting service

### `GET /ready`

Simple readiness probe for Kubernetes.

#### Response
```json
{
  "status": "ready",
  "timestamp": 1703123456.789
}
```

## 📊 Metrics Endpoint

### `GET /metrics`

Prometheus-compatible metrics for monitoring.

#### Response (Text Format)
```
# HELP sentiment_analyzer_requests_total Total number of requests
# TYPE sentiment_analyzer_requests_total counter
sentiment_analyzer_requests_total 15420

# HELP sentiment_analyzer_requests_success Total successful requests
# TYPE sentiment_analyzer_requests_success counter
sentiment_analyzer_requests_success 15389

# HELP sentiment_analyzer_error_rate Request error rate
# TYPE sentiment_analyzer_error_rate gauge
sentiment_analyzer_error_rate 0.002

# HELP sentiment_analyzer_uptime_seconds Service uptime in seconds
# TYPE sentiment_analyzer_uptime_seconds gauge
sentiment_analyzer_uptime_seconds 86400.5

# HELP sentiment_analyzer_avg_processing_time Average processing time
# TYPE sentiment_analyzer_avg_processing_time gauge
sentiment_analyzer_avg_processing_time 0.045
```

## ℹ️ Service Information

### `GET /`

Basic service information and available endpoints.

#### Response
```json
{
  "service": "Sentiment Analyzer Pro",
  "version": "1.0.0",
  "analyzer_type": "scalable",
  "endpoints": ["/analyze", "/batch", "/health", "/ready", "/metrics"]
}
```

## ⚠️ Error Handling

### Error Response Format
All errors follow this consistent format:
```json
{
  "success": false,
  "error": "Descriptive error message",
  "timestamp": 1703123456.789,
  "processing_time": 0.001,
  "request_id": "req_1703123456.789_error",
  "version": "1.0.0"
}
```

### Common Error Codes and Messages

#### 400 Bad Request
- `"Missing 'text' field"`: POST body must contain 'text' field
- `"Text cannot be empty"`: Text field cannot be empty or whitespace
- `"Text too long (max 10000 characters)"`: Text exceeds length limit
- `"Invalid JSON"`: Request body is not valid JSON
- `"Batch size too large (max 100 texts)"`: Too many texts in batch request

#### 429 Too Many Requests
- `"Rate limit exceeded for client {client_id}"`: Client has exceeded rate limits

#### 500 Internal Server Error
- `"Internal processing error: {details}"`: Unexpected error during analysis
- `"Service temporarily unavailable"`: Service is overloaded or restarting

### Error Handling Best Practices

#### Python
```python
import requests
import time

def analyze_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:8000/analyze',
                json={'text': text},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited, wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                # Other error, return error info
                return response.json()
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
    
    return {"success": False, "error": "Max retries exceeded"}
```

#### JavaScript
```javascript
async function analyzeWithRetry(text, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        timeout: 10000
      });
      
      if (response.ok) {
        return await response.json();
      } else if (response.status === 429) {
        // Rate limited, wait and retry
        const waitTime = Math.pow(2, attempt) * 1000;
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      } else {
        return await response.json();
      }
    } catch (error) {
      console.error(`Attempt ${attempt + 1} failed:`, error);
      if (attempt < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }
  
  return { success: false, error: 'Max retries exceeded' };
}
```

## 🚦 Rate Limiting

### Rate Limit Headers
Responses include rate limiting information:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703123516
```

### Rate Limit Tiers

| Endpoint | Default Limit | With API Key |
|----------|---------------|--------------|
| `/analyze` | 100/minute | 1000/minute |
| `/batch` | 10/minute | 100/minute |
| `/health` | 60/minute | 300/minute |
| `/ready` | Unlimited | Unlimited |
| `/metrics` | 60/minute | Unlimited |

## 📈 Performance Guidelines

### Request Optimization
1. **Use batch processing**: More efficient for multiple texts
2. **Enable caching**: Identical texts return cached results
3. **Implement request pooling**: Reuse HTTP connections
4. **Add request timeouts**: Prevent hanging connections

### Response Time Expectations

| Request Type | Expected Time | 95th Percentile |
|--------------|---------------|-----------------|
| Single analysis (cached) | <10ms | <20ms |
| Single analysis (new) | <50ms | <100ms |
| Batch (10 texts) | <200ms | <500ms |
| Batch (100 texts) | <2s | <5s |

### Throughput Capacity

| Deployment | Requests/Second | Concurrent Users |
|------------|-----------------|------------------|
| Single container | 200-500 | 100-250 |
| Docker Compose | 1,000-2,000 | 500-1,000 |
| Kubernetes (3 pods) | 3,000-6,000 | 1,500-3,000 |
| Kubernetes (auto-scale) | 10,000+ | 5,000+ |

## 🔍 Monitoring and Observability

### Key Metrics to Monitor
- **Request Rate**: `sentiment_analyzer_requests_total`
- **Error Rate**: `sentiment_analyzer_error_rate`
- **Response Time**: `sentiment_analyzer_avg_processing_time`
- **Cache Performance**: Cache hit rate from `/health` endpoint
- **Resource Usage**: Memory and CPU from system metrics

### Health Check Automation
```bash
#!/bin/bash
# health-check.sh
ENDPOINT="http://localhost:8080/health"
RESPONSE=$(curl -s "$ENDPOINT")
STATUS=$(echo "$RESPONSE" | jq -r '.data.status')

if [ "$STATUS" == "healthy" ]; then
  echo "✅ Service is healthy"
  exit 0
else
  echo "❌ Service is $STATUS"
  echo "$RESPONSE" | jq '.data.issues'
  exit 1
fi
```

### Alerting Rules (Prometheus)
```yaml
groups:
- name: sentiment-analyzer
  rules:
  - alert: HighErrorRate
    expr: sentiment_analyzer_error_rate > 0.05
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: SlowResponseTime
    expr: sentiment_analyzer_avg_processing_time > 1.0
    for: 5m
    annotations:
      summary: "Slow response times detected"
      
  - alert: ServiceDown
    expr: up{job="sentiment-analyzer"} == 0
    for: 1m
    annotations:
      summary: "Sentiment analyzer service is down"
```

## 📚 SDK and Client Libraries

### Python Client
```python
class SentimentClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def analyze(self, text):
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"text": text},
            headers=self.headers
        )
        return response.json()
    
    def batch_analyze(self, texts):
        response = requests.post(
            f"{self.base_url}/batch",
            json={"texts": texts},
            headers=self.headers
        )
        return response.json()

# Usage
client = SentimentClient()
result = client.analyze("I love this product!")
```

### Node.js Client
```javascript
class SentimentClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = { 'Content-Type': 'application/json' };
    if (apiKey) this.headers['X-API-Key'] = apiKey;
  }
  
  async analyze(text) {
    const response = await fetch(`${this.baseUrl}/analyze`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ text })
    });
    return response.json();
  }
  
  async batchAnalyze(texts) {
    const response = await fetch(`${this.baseUrl}/batch`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ texts })
    });
    return response.json();
  }
}

// Usage
const client = new SentimentClient();
const result = await client.analyze('Amazing product quality!');
```

---

## 🆘 Support

For additional API support:
- 📖 Check the [Deployment Guide](DEPLOYMENT.md)
- 🐛 Report issues on [GitHub](https://github.com/danieleschmidt/sentiment-analyzer-pro/issues)
- 📧 Contact support: sentiment-support@company.com

## 📄 API Changelog

### v1.0.0 (Current)
- ✨ Initial release with basic sentiment analysis
- ✨ Batch processing support
- ✨ Evolutionary prompt optimization
- ✨ Production-ready error handling and monitoring