"""
Production API Server for Sentiment Analyzer Pro

Fast, scalable HTTP API server with comprehensive monitoring,
health checks, and production-ready features.
"""

import asyncio
import json
import time
import logging
import os
import signal
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor

# HTTP server imports (using built-in modules for minimal dependencies)
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if os.getenv('LOG_FORMAT') != 'json' else None,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/api.log') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)

if os.getenv('LOG_FORMAT') == 'json':
    import json
    import sys
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': self.formatTime(record, self.datefmt),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_entry)
    
    # Apply JSON formatter to all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)

# Import sentiment analyzers with fallbacks
try:
    from scalable_sentiment_analyzer import ScalableSentimentAnalyzer
    SCALABLE_AVAILABLE = True
    logger.info("Scalable sentiment analyzer available")
except ImportError as e:
    logger.warning(f"Scalable analyzer not available: {e}")
    SCALABLE_AVAILABLE = False

try:
    from robust_sentiment_analyzer import RobustSentimentAnalyzer
    ROBUST_AVAILABLE = True
    logger.info("Robust sentiment analyzer available")
except ImportError as e:
    logger.warning(f"Robust analyzer not available: {e}")
    ROBUST_AVAILABLE = False

try:
    from standalone_sentiment_demo import StandaloneSentimentAnalyzer
    STANDALONE_AVAILABLE = True
    logger.info("Standalone sentiment analyzer available")
except ImportError as e:
    logger.warning(f"Standalone analyzer not available: {e}")
    STANDALONE_AVAILABLE = False

@dataclass
class APIResponse:
    """Standard API response format"""
    success: bool
    data: Any = None
    error: str = None
    timestamp: float = None
    processing_time: float = None
    request_id: str = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class SentimentAnalyzerAPI:
    """Production API server for sentiment analysis"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.version = "1.0.0"
        
        # Configuration from environment
        self.config = {
            'port': int(os.getenv('PORT', 8000)),
            'health_port': int(os.getenv('HEALTH_CHECK_PORT', 8080)),
            'metrics_port': int(os.getenv('PROMETHEUS_PORT', 9090)),
            'max_workers': int(os.getenv('MAX_WORKERS', 16)),
            'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            'enable_monitoring': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
            'rate_limit_rpm': int(os.getenv('RATE_LIMIT_RPM', 1000)),
            'max_text_length': int(os.getenv('MAX_TEXT_LENGTH', 10000))
        }
        
        # Initialize analyzer
        self._init_analyzer()
        
        # Metrics collection
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'processing_time_total': 0.0,
            'uptime_seconds': 0.0
        }
        
        # Thread pool for request processing
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        
        logger.info(f"SentimentAnalyzerAPI initialized with config: {self.config}")
    
    def _init_analyzer(self):
        """Initialize the best available analyzer"""
        if SCALABLE_AVAILABLE:
            self.analyzer = ScalableSentimentAnalyzer(
                cache_size=int(os.getenv('CACHE_SIZE', 10000)),
                min_workers=int(os.getenv('MIN_WORKERS', 4)),
                max_workers=int(os.getenv('MAX_WORKERS', 16)),
                enable_distributed_cache=os.getenv('ENABLE_DISTRIBUTED_CACHE', 'false').lower() == 'true'
            )
            self.analyzer_type = "scalable"
            logger.info("Using ScalableSentimentAnalyzer")
            
        elif ROBUST_AVAILABLE:
            self.analyzer = RobustSentimentAnalyzer(
                max_text_length=self.config['max_text_length'],
                rate_limit_rpm=self.config['rate_limit_rpm']
            )
            self.analyzer_type = "robust"
            logger.info("Using RobustSentimentAnalyzer")
            
        elif STANDALONE_AVAILABLE:
            self.analyzer = StandaloneSentimentAnalyzer()
            self.analyzer_type = "standalone"
            logger.info("Using StandaloneSentimentAnalyzer")
            
        else:
            raise RuntimeError("No sentiment analyzer available!")
    
    def analyze_sentiment(self, text: str, request_id: str = None) -> APIResponse:
        """Analyze sentiment with error handling"""
        start_time = time.time()
        
        try:
            self.request_count += 1
            self.metrics['requests_total'] += 1
            
            # Input validation
            if not isinstance(text, str):
                raise ValueError("Text must be a string")
            
            if len(text) > self.config['max_text_length']:
                raise ValueError(f"Text too long (max {self.config['max_text_length']} characters)")
            
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Perform analysis based on analyzer type
            if self.analyzer_type == "scalable":
                if hasattr(self.analyzer, 'analyze_sentiment_sync'):
                    result = self.analyzer.analyze_sentiment_sync(text)
                else:
                    # Fallback to basic method
                    result = self.analyzer.analyze_sentiment(text)
                    
            elif self.analyzer_type == "robust":
                result = self.analyzer.analyze_sentiment(text, enable_rate_limit=False)
                
            elif self.analyzer_type == "standalone":
                result = self.analyzer.analyze_sentiment(text)
            
            # Normalize result format
            if hasattr(result, '__dict__'):
                result_data = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
            else:
                result_data = result
            
            processing_time = time.time() - start_time
            self.metrics['requests_success'] += 1
            self.metrics['processing_time_total'] += processing_time
            
            return APIResponse(
                success=True,
                data=result_data,
                processing_time=processing_time,
                request_id=request_id
            )
            
        except Exception as e:
            self.error_count += 1
            self.metrics['requests_error'] += 1
            processing_time = time.time() - start_time
            
            logger.error(f"Analysis failed for request {request_id}: {str(e)}")
            
            return APIResponse(
                success=False,
                error=str(e),
                processing_time=processing_time,
                request_id=request_id
            )
    
    def batch_analyze(self, texts: List[str], request_id: str = None) -> APIResponse:
        """Batch sentiment analysis"""
        start_time = time.time()
        
        try:
            if not isinstance(texts, list):
                raise ValueError("Texts must be a list")
            
            if len(texts) > 100:  # Reasonable batch limit
                raise ValueError("Batch size too large (max 100 texts)")
            
            if not texts:
                raise ValueError("Texts list cannot be empty")
            
            # Process batch
            results = []
            for i, text in enumerate(texts):
                text_result = self.analyze_sentiment(text, f"{request_id}_{i}" if request_id else None)
                if text_result.success:
                    results.append(text_result.data)
                else:
                    results.append({
                        "text": text,
                        "error": text_result.error,
                        "label": "unknown",
                        "confidence": 0.0
                    })
            
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=True,
                data={
                    "results": results,
                    "total_texts": len(texts),
                    "successful": sum(1 for r in results if "error" not in r)
                },
                processing_time=processing_time,
                request_id=request_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Batch analysis failed for request {request_id}: {str(e)}")
            
            return APIResponse(
                success=False,
                error=str(e),
                processing_time=processing_time,
                request_id=request_id
            )
    
    def get_health(self) -> APIResponse:
        """Health check endpoint"""
        uptime = time.time() - self.start_time
        self.metrics['uptime_seconds'] = uptime
        
        # Calculate health metrics
        error_rate = self.error_count / max(self.request_count, 1)
        avg_processing_time = self.metrics['processing_time_total'] / max(self.metrics['requests_success'], 1)
        
        # Determine health status
        status = "healthy"
        issues = []
        
        if error_rate > 0.1:
            status = "degraded"
            issues.append(f"High error rate: {error_rate:.2%}")
        
        if avg_processing_time > 1.0:
            status = "degraded"
            issues.append(f"High processing time: {avg_processing_time:.2f}s")
        
        if uptime < 30:
            status = "starting"
            issues.append("Service recently started")
        
        # Get analyzer-specific health if available
        analyzer_health = {}
        if hasattr(self.analyzer, 'health_check'):
            try:
                analyzer_health = self.analyzer.health_check()
            except Exception as e:
                logger.warning(f"Analyzer health check failed: {e}")
        
        health_data = {
            "status": status,
            "version": self.version,
            "uptime_seconds": uptime,
            "analyzer_type": self.analyzer_type,
            "requests_total": self.request_count,
            "error_rate": error_rate,
            "avg_processing_time": avg_processing_time,
            "issues": issues,
            "analyzer_health": analyzer_health
        }
        
        return APIResponse(success=True, data=health_data)
    
    def get_metrics(self) -> str:
        """Prometheus-style metrics"""
        uptime = time.time() - self.start_time
        error_rate = self.error_count / max(self.request_count, 1)
        avg_processing_time = self.metrics['processing_time_total'] / max(self.metrics['requests_success'], 1)
        
        metrics = f"""# HELP sentiment_analyzer_requests_total Total number of requests
# TYPE sentiment_analyzer_requests_total counter
sentiment_analyzer_requests_total {self.metrics['requests_total']}

# HELP sentiment_analyzer_requests_success Total number of successful requests  
# TYPE sentiment_analyzer_requests_success counter
sentiment_analyzer_requests_success {self.metrics['requests_success']}

# HELP sentiment_analyzer_requests_error Total number of failed requests
# TYPE sentiment_analyzer_requests_error counter  
sentiment_analyzer_requests_error {self.metrics['requests_error']}

# HELP sentiment_analyzer_uptime_seconds Service uptime in seconds
# TYPE sentiment_analyzer_uptime_seconds gauge
sentiment_analyzer_uptime_seconds {uptime}

# HELP sentiment_analyzer_error_rate Request error rate
# TYPE sentiment_analyzer_error_rate gauge
sentiment_analyzer_error_rate {error_rate}

# HELP sentiment_analyzer_avg_processing_time Average processing time in seconds
# TYPE sentiment_analyzer_avg_processing_time gauge
sentiment_analyzer_avg_processing_time {avg_processing_time}

# HELP sentiment_analyzer_info Service information
# TYPE sentiment_analyzer_info gauge
sentiment_analyzer_info{{version="{self.version}",analyzer_type="{self.analyzer_type}"}} 1
"""
        return metrics

class APIRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler"""
    
    def __init__(self, api_server, *args, **kwargs):
        self.api_server = api_server
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use proper logging"""
        logger.info(f"{self.client_address[0]} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/health':
                response = self.api_server.get_health()
                self.send_json_response(200, asdict(response))
                
            elif path == '/ready':
                # Readiness check (simpler than health check)
                self.send_json_response(200, {"status": "ready", "timestamp": time.time()})
                
            elif path == '/metrics':
                metrics_text = self.api_server.get_metrics()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(metrics_text.encode())
                
            elif path == '/':
                self.send_json_response(200, {
                    "service": "Sentiment Analyzer Pro",
                    "version": self.api_server.version,
                    "analyzer_type": self.api_server.analyzer_type,
                    "endpoints": ["/analyze", "/batch", "/health", "/ready", "/metrics"]
                })
                
            else:
                self.send_json_response(404, {"error": "Endpoint not found"})
                
        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self.send_json_response(500, {"error": "Internal server error"})
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
            except json.JSONDecodeError:
                self.send_json_response(400, {"error": "Invalid JSON"})
                return
            
            request_id = self.headers.get('X-Request-ID', f"req_{time.time()}")
            
            if path == '/analyze':
                text = data.get('text')
                if not text:
                    self.send_json_response(400, {"error": "Missing 'text' field"})
                    return
                
                response = self.api_server.analyze_sentiment(text, request_id)
                status_code = 200 if response.success else 400
                self.send_json_response(status_code, asdict(response))
                
            elif path == '/batch':
                texts = data.get('texts')
                if not texts:
                    self.send_json_response(400, {"error": "Missing 'texts' field"})
                    return
                
                response = self.api_server.batch_analyze(texts, request_id)
                status_code = 200 if response.success else 400
                self.send_json_response(status_code, asdict(response))
                
            else:
                self.send_json_response(404, {"error": "Endpoint not found"})
                
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self.send_json_response(500, {"error": "Internal server error"})
    
    def send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response"""
        response_json = json.dumps(data, default=str)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_json)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Request-ID')
        self.end_headers()
        self.wfile.write(response_json.encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Request-ID')
        self.end_headers()

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for concurrent request handling"""
    daemon_threads = True
    allow_reuse_address = True

def create_handler_factory(api_server):
    """Create request handler factory"""
    def handler(*args, **kwargs):
        return APIRequestHandler(api_server, *args, **kwargs)
    return handler

def signal_handler(signum, frame, servers):
    """Graceful shutdown handler"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    for server in servers:
        server.shutdown()
    
    logger.info("Servers shut down completed")

def main():
    """Main server function"""
    logger.info("Starting Sentiment Analyzer Pro API Server")
    
    # Initialize API server
    api_server = SentimentAnalyzerAPI()
    
    # Create HTTP servers
    main_server = ThreadedHTTPServer(
        ('0.0.0.0', api_server.config['port']),
        create_handler_factory(api_server)
    )
    
    health_server = ThreadedHTTPServer(
        ('0.0.0.0', api_server.config['health_port']),
        create_handler_factory(api_server)
    )
    
    servers = [main_server, health_server]
    
    # Setup signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        signal_handler(signum, frame, servers)
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # Start servers in separate threads
    def run_server(server, name):
        logger.info(f"Starting {name} server on {server.server_address}")
        try:
            server.serve_forever()
        except Exception as e:
            logger.error(f"{name} server error: {e}")
        finally:
            server.server_close()
            logger.info(f"{name} server stopped")
    
    # Start health/metrics server
    health_thread = threading.Thread(
        target=run_server,
        args=(health_server, "health"),
        daemon=True
    )
    health_thread.start()
    
    # Start main API server (blocking)
    logger.info(f"API Server ready - Main API: http://0.0.0.0:{api_server.config['port']}")
    logger.info(f"Health endpoint: http://0.0.0.0:{api_server.config['health_port']}/health")
    logger.info(f"Metrics endpoint: http://0.0.0.0:{api_server.config['health_port']}/metrics")
    
    try:
        run_server(main_server, "main API")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()