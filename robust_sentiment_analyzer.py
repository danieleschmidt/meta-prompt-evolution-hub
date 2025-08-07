"""
Robust Sentiment Analyzer: Generation 2 - Enhanced Error Handling & Validation

Adds comprehensive error handling, input validation, logging, monitoring,
and security features to the sentiment analysis system.
"""

import asyncio
import json
import time
import logging
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from functools import wraps
import traceback
import numpy as np

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SentimentError(Exception):
    """Base exception for sentiment analysis errors"""
    pass

class ValidationError(SentimentError):
    """Input validation error"""
    pass

class ProcessingError(SentimentError):
    """Processing pipeline error"""
    pass

class RateLimitError(SentimentError):
    """Rate limiting error"""
    pass

class SecurityError(SentimentError):
    """Security-related error"""
    pass

class SentimentLabel(Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    UNKNOWN = "unknown"  # For error cases

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_text: Optional[str] = None

@dataclass
class SentimentResult:
    text: str
    label: SentimentLabel
    confidence: float
    processing_time: float
    prompt_used: str
    model_used: str = "evolutionary-optimized"
    validation_result: Optional[ValidationResult] = None
    error_details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None

@dataclass
class HealthMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_processing_time: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

class InputValidator:
    """Robust input validation and sanitization"""
    
    def __init__(self, max_length: int = 10000, min_length: int = 1):
        self.max_length = max_length
        self.min_length = min_length
        
        # Security patterns
        self.malicious_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',    # Event handlers
            r'eval\s*\(',    # eval() calls
            r'exec\s*\(',    # exec() calls
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
    
    def validate_text(self, text: Any) -> ValidationResult:
        """Comprehensive text validation"""
        errors = []
        warnings = []
        
        # Type checking
        if not isinstance(text, str):
            if text is None:
                errors.append("Input text cannot be None")
                return ValidationResult(False, errors)
            
            # Try to convert to string
            try:
                text = str(text)
                warnings.append(f"Input converted from {type(text).__name__} to string")
            except Exception as e:
                errors.append(f"Cannot convert input to string: {str(e)}")
                return ValidationResult(False, errors)
        
        # Length validation
        if len(text) < self.min_length:
            errors.append(f"Text too short (minimum {self.min_length} characters)")
        
        if len(text) > self.max_length:
            errors.append(f"Text too long (maximum {self.max_length} characters)")
            # Truncate for safety
            text = text[:self.max_length]
            warnings.append(f"Text truncated to {self.max_length} characters")
        
        # Security validation
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                errors.append(f"Potentially malicious content detected")
                break
        
        # Content validation
        if not text.strip():
            errors.append("Text cannot be empty or whitespace only")
        
        # Character encoding validation
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Text contains invalid UTF-8 characters")
        
        # Sanitize text
        sanitized_text = self._sanitize_text(text)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_text=sanitized_text
        )
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize input text"""
        # Remove potential XSS/injection attempts
        sanitized = text
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized

class RateLimiter:
    """Rate limiting for API protection"""
    
    def __init__(self, requests_per_minute: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Cleanup old requests
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time)
        
        # Get current window requests
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests (older than 1 minute)
        minute_ago = current_time - 60
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def _cleanup_old_requests(self, current_time: float):
        """Clean up old request records"""
        minute_ago = current_time - 60
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > minute_ago
            ]
            if not self.requests[identifier]:
                del self.requests[identifier]
        
        self.last_cleanup = current_time

def error_handler(func):
    """Decorator for comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {func.__name__}: {str(e)}")
            raise
        except ProcessingError as e:
            logger.error(f"Processing error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Internal processing error: {str(e)}")
    return wrapper

def async_error_handler(func):
    """Decorator for async error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {func.__name__}: {str(e)}")
            raise
        except ProcessingError as e:
            logger.error(f"Processing error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Internal processing error: {str(e)}")
    return wrapper

class RobustSentimentAnalyzer:
    """Robust sentiment analyzer with comprehensive error handling"""
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 max_text_length: int = 10000,
                 rate_limit_rpm: int = 1000,
                 enable_monitoring: bool = True):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.enable_monitoring = enable_monitoring
        
        # Core components with error handling
        try:
            self.validator = InputValidator(max_length=max_text_length)
            self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
            self._initialize_core_system()
            
        except Exception as e:
            logger.critical(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise ProcessingError(f"Initialization failed: {str(e)}")
        
        # Health monitoring
        self.health_metrics = HealthMetrics()
        self.start_time = time.time()
        
        logger.info(f"RobustSentimentAnalyzer initialized successfully")
    
    def _initialize_core_system(self):
        """Initialize core sentiment analysis system"""
        from sentiment_analyzer import SentimentEvolutionHub
        
        try:
            self.core_analyzer = SentimentEvolutionHub(
                population_size=self.population_size,
                mutation_rate=self.mutation_rate
            )
        except Exception as e:
            raise ProcessingError(f"Failed to initialize core analyzer: {str(e)}")
    
    def _generate_request_id(self, text: str) -> str:
        """Generate unique request ID for tracking"""
        timestamp = str(time.time())
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"req_{timestamp}_{text_hash}"
    
    @error_handler
    def analyze_sentiment(self, 
                         text: str, 
                         client_id: str = "default",
                         enable_rate_limit: bool = True) -> SentimentResult:
        """Robust sentiment analysis with full error handling"""
        
        start_time = time.time()
        request_id = self._generate_request_id(text)
        
        logger.info(f"Processing sentiment analysis request {request_id}")
        
        try:
            # Update metrics
            self.health_metrics.total_requests += 1
            
            # Rate limiting
            if enable_rate_limit and not self.rate_limiter.check_rate_limit(client_id):
                self.health_metrics.failed_requests += 1
                raise RateLimitError(f"Rate limit exceeded for client {client_id}")
            
            # Input validation
            validation_result = self.validator.validate_text(text)
            if not validation_result.is_valid:
                self.health_metrics.failed_requests += 1
                raise ValidationError(f"Input validation failed: {', '.join(validation_result.errors)}")
            
            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(f"Request {request_id}: {warning}")
            
            # Use sanitized text
            clean_text = validation_result.sanitized_text or text
            
            # Perform analysis
            try:
                core_result = self.core_analyzer.analyze_sentiment(clean_text)
                
                result = SentimentResult(
                    text=text,  # Original text
                    label=core_result.label,
                    confidence=core_result.confidence,
                    processing_time=time.time() - start_time,
                    prompt_used=core_result.prompt_used,
                    model_used=core_result.model_used,
                    validation_result=validation_result,
                    request_id=request_id
                )
                
                self.health_metrics.successful_requests += 1
                logger.info(f"Request {request_id} completed successfully")
                
                return result
                
            except Exception as e:
                logger.error(f"Core analysis failed for request {request_id}: {str(e)}")
                
                # Return error result instead of crashing
                self.health_metrics.failed_requests += 1
                return SentimentResult(
                    text=text,
                    label=SentimentLabel.UNKNOWN,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    prompt_used="error_fallback",
                    validation_result=validation_result,
                    error_details=str(e),
                    request_id=request_id
                )
        
        except (RateLimitError, ValidationError) as e:
            # Re-raise expected errors
            raise
        except Exception as e:
            self.health_metrics.failed_requests += 1
            logger.error(f"Unexpected error in request {request_id}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Analysis failed: {str(e)}")
    
    @async_error_handler
    async def async_analyze_sentiment(self, 
                                    text: str, 
                                    client_id: str = "default",
                                    enable_rate_limit: bool = True) -> SentimentResult:
        """Async sentiment analysis with error handling"""
        
        # Add small delay to simulate async processing
        await asyncio.sleep(0.01)
        
        # Use sync method (in production, would be fully async)
        return self.analyze_sentiment(text, client_id, enable_rate_limit)
    
    @error_handler
    def batch_analyze(self, 
                     texts: List[str], 
                     client_id: str = "default",
                     fail_fast: bool = False) -> List[SentimentResult]:
        """Batch analysis with robust error handling"""
        
        if not texts:
            raise ValidationError("Text list cannot be empty")
        
        if len(texts) > 1000:  # Reasonable batch size limit
            raise ValidationError("Batch size too large (max 1000 texts)")
        
        results = []
        failed_count = 0
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text, client_id)
                results.append(result)
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to analyze text {i}: {str(e)}")
                
                if fail_fast:
                    raise ProcessingError(f"Batch analysis failed at item {i}: {str(e)}")
                
                # Add error result
                error_result = SentimentResult(
                    text=text,
                    label=SentimentLabel.UNKNOWN,
                    confidence=0.0,
                    processing_time=0.0,
                    prompt_used="error_fallback",
                    error_details=str(e)
                )
                results.append(error_result)
        
        logger.info(f"Batch analysis completed: {len(texts)} total, {failed_count} failed")
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates
        if self.health_metrics.total_requests > 0:
            success_rate = self.health_metrics.successful_requests / self.health_metrics.total_requests
            error_rate = self.health_metrics.failed_requests / self.health_metrics.total_requests
        else:
            success_rate = 1.0
            error_rate = 0.0
        
        return {
            "status": "healthy" if error_rate < 0.1 else "degraded" if error_rate < 0.5 else "unhealthy",
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "total_requests": self.health_metrics.total_requests,
            "successful_requests": self.health_metrics.successful_requests,
            "failed_requests": self.health_metrics.failed_requests,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "requests_per_hour": self.health_metrics.total_requests / max(uptime / 3600, 0.001),
            "core_analyzer_metrics": self.core_analyzer.get_metrics() if hasattr(self.core_analyzer, 'get_metrics') else {}
        }
    
    def export_error_logs(self, filepath: str, last_n_hours: int = 24):
        """Export error logs for analysis"""
        # In production, would read from log files
        health_status = self.get_health_status()
        
        export_data = {
            "timestamp": time.time(),
            "export_period_hours": last_n_hours,
            "health_status": health_status,
            "configuration": {
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "rate_limit_rpm": self.rate_limiter.requests_per_minute,
                "max_text_length": self.validator.max_length
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Error logs exported to {filepath}")

# Circuit breaker pattern for resilience
class CircuitBreaker:
    """Circuit breaker for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise ProcessingError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise

if __name__ == "__main__":
    # Demo of robust sentiment analyzer
    analyzer = RobustSentimentAnalyzer()
    
    # Test valid inputs
    test_texts = [
        "I love this product!",
        "This is terrible service",
        "It's okay, nothing special",
        "",  # Empty text - should fail validation
        "A" * 20000,  # Too long - should be truncated
        None,  # Invalid type
        123,  # Invalid type but convertible
    ]
    
    print("Testing robust sentiment analysis:")
    for i, text in enumerate(test_texts):
        try:
            result = analyzer.analyze_sentiment(text, client_id="test_client")
            print(f"\n{i+1}. Success:")
            print(f"   Text: {str(text)[:50]}...")
            print(f"   Sentiment: {result.label.value}")
            print(f"   Confidence: {result.confidence:.2f}")
            if result.validation_result:
                print(f"   Warnings: {result.validation_result.warnings}")
            if result.error_details:
                print(f"   Error: {result.error_details}")
                
        except Exception as e:
            print(f"\n{i+1}. Error: {str(e)}")
    
    # Health check
    print(f"\nHealth Status:")
    health = analyzer.get_health_status()
    for key, value in health.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Export logs
    analyzer.export_error_logs("robust_analyzer_logs.json")
    print("Error logs exported to robust_analyzer_logs.json")