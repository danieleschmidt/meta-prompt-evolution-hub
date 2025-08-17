"""
Generation 2: Robust Security & Monitoring System
Comprehensive error handling, logging, security, and real-time monitoring
"""

import asyncio
import json
import time
import uuid
import logging
import hashlib
import hmac
import secrets
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
import threading
import queue
import os
from pathlib import Path


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class LogLevel(Enum):
    """Enhanced logging levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60


@dataclass
class SecurityEvent:
    """Security event tracking."""
    event_id: str
    event_type: str
    severity: str
    timestamp: float
    source: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class AuditLog:
    """Comprehensive audit logging."""
    log_id: str
    timestamp: float
    operation: str
    user_id: str
    resource_id: str
    action: str
    result: str
    ip_address: str
    user_agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecureLogger:
    """Enhanced logging with security features."""
    
    def __init__(self, log_dir: str = "logs", encryption_key: Optional[bytes] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.encryption_key = encryption_key or secrets.token_bytes(32)
        
        # Setup loggers
        self.setup_loggers()
        
        # Security event queue
        self.security_events = queue.Queue()
        self.audit_logs = queue.Queue()
        
        # Start background processors
        self.start_background_processors()
    
    def setup_loggers(self):
        """Setup multiple specialized loggers."""
        # Main application logger
        self.app_logger = logging.getLogger("meta_prompt_evolution.app")
        self.app_logger.setLevel(logging.DEBUG)
        
        # Security logger
        self.security_logger = logging.getLogger("meta_prompt_evolution.security")
        self.security_logger.setLevel(logging.INFO)
        
        # Performance logger
        self.performance_logger = logging.getLogger("meta_prompt_evolution.performance")
        self.performance_logger.setLevel(logging.INFO)
        
        # Audit logger
        self.audit_logger = logging.getLogger("meta_prompt_evolution.audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Setup handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup file and console handlers with rotation."""
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s | %(pathname)s:%(lineno)d'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.app_logger.addHandler(console_handler)
        
        # File handlers with rotation
        from logging.handlers import RotatingFileHandler
        
        # Application logs
        app_handler = RotatingFileHandler(
            self.log_dir / "application.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setFormatter(formatter)
        self.app_logger.addHandler(app_handler)
        
        # Security logs
        security_handler = RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        security_handler.setFormatter(formatter)
        self.security_logger.addHandler(security_handler)
        
        # Performance logs
        perf_handler = RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3
        )
        perf_handler.setFormatter(formatter)
        self.performance_logger.addHandler(perf_handler)
        
        # Audit logs
        audit_handler = RotatingFileHandler(
            self.log_dir / "audit.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        audit_handler.setFormatter(formatter)
        self.audit_logger.addHandler(audit_handler)
    
    def start_background_processors(self):
        """Start background threads for log processing."""
        # Security event processor
        security_thread = threading.Thread(
            target=self.process_security_events,
            daemon=True
        )
        security_thread.start()
        
        # Audit log processor
        audit_thread = threading.Thread(
            target=self.process_audit_logs,
            daemon=True
        )
        audit_thread.start()
    
    def process_security_events(self):
        """Process security events in background."""
        while True:
            try:
                event = self.security_events.get(timeout=1.0)
                self.security_logger.warning(
                    f"SECURITY_EVENT | {event.event_type} | {event.severity} | "
                    f"{event.source} | {json.dumps(event.details)}"
                )
                
                # Critical events require immediate attention
                if event.severity == "CRITICAL":
                    self.security_logger.critical(
                        f"CRITICAL_SECURITY_EVENT | {event.event_id} | "
                        f"Immediate attention required | {event.details}"
                    )
                
            except queue.Empty:
                continue
            except Exception as e:
                self.app_logger.error(f"Error processing security event: {e}")
    
    def process_audit_logs(self):
        """Process audit logs in background."""
        while True:
            try:
                audit = self.audit_logs.get(timeout=1.0)
                self.audit_logger.info(
                    f"AUDIT | {audit.operation} | {audit.user_id} | "
                    f"{audit.action} | {audit.result} | {audit.ip_address}"
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                self.app_logger.error(f"Error processing audit log: {e}")
    
    def log_security_event(self, event_type: str, severity: str, source: str, 
                          details: Dict[str, Any], user_id: Optional[str] = None):
        """Log security event."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            source=source,
            details=details,
            user_id=user_id
        )
        
        self.security_events.put(event)
    
    def log_audit_event(self, operation: str, user_id: str, resource_id: str,
                       action: str, result: str, ip_address: str = "127.0.0.1",
                       user_agent: str = "internal", metadata: Optional[Dict] = None):
        """Log audit event."""
        audit = AuditLog(
            log_id=str(uuid.uuid4()),
            timestamp=time.time(),
            operation=operation,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
        
        self.audit_logs.put(audit)


class SecurityManager:
    """Comprehensive security management."""
    
    def __init__(self, logger: SecureLogger):
        self.logger = logger
        self.api_keys = {}
        self.session_tokens = {}
        self.rate_limits = defaultdict(list)
        self.blocked_ips = set()
        self.security_policies = self.load_security_policies()
        
    def load_security_policies(self) -> Dict[str, Any]:
        """Load security policies and configurations."""
        return {
            "max_prompt_length": 10000,
            "max_requests_per_minute": 100,
            "allowed_file_types": [".txt", ".json", ".csv"],
            "blocked_patterns": ["<script>", "javascript:", "eval("],
            "encryption_required": True,
            "audit_all_operations": True,
            "session_timeout_minutes": 30,
            "password_min_length": 12,
            "require_2fa": False
        }
    
    def generate_api_key(self, user_id: str, scope: List[str]) -> str:
        """Generate secure API key."""
        key_data = {
            "user_id": user_id,
            "scope": scope,
            "created": time.time(),
            "expires": time.time() + (365 * 24 * 3600)  # 1 year
        }
        
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = key_data
        
        self.logger.log_security_event(
            "API_KEY_GENERATED",
            "INFO",
            "SecurityManager",
            {"user_id": user_id, "scope": scope}
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        if api_key not in self.api_keys:
            self.logger.log_security_event(
                "INVALID_API_KEY",
                "WARNING",
                "SecurityManager",
                {"api_key_prefix": api_key[:8]}
            )
            return None
        
        key_data = self.api_keys[api_key]
        
        # Check expiration
        if time.time() > key_data["expires"]:
            self.logger.log_security_event(
                "EXPIRED_API_KEY",
                "WARNING",
                "SecurityManager",
                {"user_id": key_data["user_id"]}
            )
            return None
        
        return key_data
    
    def check_rate_limit(self, user_id: str, operation: str) -> bool:
        """Check if user is within rate limits."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        key = f"{user_id}:{operation}"
        
        # Remove old entries
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[key]) >= self.security_policies["max_requests_per_minute"]:
            self.logger.log_security_event(
                "RATE_LIMIT_EXCEEDED",
                "WARNING",
                "SecurityManager",
                {"user_id": user_id, "operation": operation}
            )
            return False
        
        # Add current request
        self.rate_limits[key].append(current_time)
        return True
    
    def validate_input(self, input_data: str, data_type: str = "prompt") -> Dict[str, Any]:
        """Comprehensive input validation."""
        validation_result = {
            "valid": True,
            "issues": [],
            "sanitized_data": input_data,
            "risk_level": "LOW"
        }
        
        # Length validation
        max_length = self.security_policies.get("max_prompt_length", 10000)
        if len(input_data) > max_length:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Input exceeds maximum length of {max_length}")
            validation_result["risk_level"] = "HIGH"
        
        # Pattern detection
        for pattern in self.security_policies["blocked_patterns"]:
            if pattern.lower() in input_data.lower():
                validation_result["valid"] = False
                validation_result["issues"].append(f"Blocked pattern detected: {pattern}")
                validation_result["risk_level"] = "CRITICAL"
                
                self.logger.log_security_event(
                    "MALICIOUS_PATTERN_DETECTED",
                    "CRITICAL",
                    "SecurityManager",
                    {"pattern": pattern, "input_preview": input_data[:100]}
                )
        
        # Character encoding validation
        try:
            input_data.encode('utf-8')
        except UnicodeError:
            validation_result["valid"] = False
            validation_result["issues"].append("Invalid character encoding")
            validation_result["risk_level"] = "MEDIUM"
        
        # Sanitization
        if validation_result["valid"]:
            # Basic sanitization - remove potentially dangerous characters
            sanitized = input_data.replace('<', '&lt;').replace('>', '&gt;')
            validation_result["sanitized_data"] = sanitized
        
        return validation_result
    
    def encrypt_sensitive_data(self, data: str, classification: SecurityLevel) -> str:
        """Encrypt sensitive data based on classification."""
        if classification in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
            # Simple encryption for demo (use proper encryption in production)
            encoded = data.encode('utf-8')
            encrypted = hashlib.sha256(encoded).hexdigest()
            return encrypted
        
        return data
    
    def create_secure_session(self, user_id: str) -> str:
        """Create secure session token."""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            "user_id": user_id,
            "created": time.time(),
            "last_activity": time.time(),
            "expires": time.time() + (self.security_policies["session_timeout_minutes"] * 60)
        }
        
        self.session_tokens[session_id] = session_data
        
        self.logger.log_security_event(
            "SESSION_CREATED",
            "INFO",
            "SecurityManager",
            {"user_id": user_id, "session_id": session_id[:8]}
        )
        
        return session_id


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self, logger: SecureLogger):
        self.logger = logger
        self.metrics_queue = queue.Queue()
        self.performance_data = defaultdict(list)
        self.alert_thresholds = {
            "max_duration": 30.0,  # seconds
            "max_memory_mb": 1000,
            "max_cpu_percent": 80.0,
            "error_rate_threshold": 0.1  # 10%
        }
        
        # Start monitoring thread
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start performance monitoring thread."""
        monitor_thread = threading.Thread(
            target=self.process_metrics,
            daemon=True
        )
        monitor_thread.start()
    
    def process_metrics(self):
        """Process performance metrics in background."""
        while True:
            try:
                metric = self.metrics_queue.get(timeout=1.0)
                
                # Store metric
                self.performance_data[metric.operation].append(metric)
                
                # Keep only recent metrics (last 1000 per operation)
                if len(self.performance_data[metric.operation]) > 1000:
                    self.performance_data[metric.operation] = \
                        self.performance_data[metric.operation][-1000:]
                
                # Log performance metric
                self.logger.performance_logger.info(
                    f"PERF | {metric.operation} | {metric.duration:.3f}s | "
                    f"Memory: {metric.memory_usage:.1f}MB | "
                    f"Success: {metric.success}"
                )
                
                # Check for performance issues
                self.check_performance_alerts(metric)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.app_logger.error(f"Error processing performance metric: {e}")
    
    def check_performance_alerts(self, metric: PerformanceMetrics):
        """Check if metric triggers any performance alerts."""
        alerts = []
        
        # Duration alert
        if metric.duration > self.alert_thresholds["max_duration"]:
            alerts.append(f"High duration: {metric.duration:.2f}s")
        
        # Memory alert
        if metric.memory_usage > self.alert_thresholds["max_memory_mb"]:
            alerts.append(f"High memory usage: {metric.memory_usage:.1f}MB")
        
        # CPU alert (if available)
        if metric.cpu_usage > self.alert_thresholds["max_cpu_percent"]:
            alerts.append(f"High CPU usage: {metric.cpu_usage:.1f}%")
        
        # Error alert
        if not metric.success:
            alerts.append(f"Operation failed: {metric.error_message}")
        
        # Send alerts
        for alert in alerts:
            self.logger.log_security_event(
                "PERFORMANCE_ALERT",
                "WARNING",
                "PerformanceMonitor",
                {
                    "operation": metric.operation,
                    "alert": alert,
                    "metric_details": asdict(metric)
                }
            )
    
    def record_operation(self, operation: str, start_time: float, 
                        success: bool = True, error_message: Optional[str] = None):
        """Record operation performance."""
        end_time = time.time()
        duration = end_time - start_time
        
        # Get memory usage (simplified)
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()
        except ImportError:
            memory_usage = 0.0
            cpu_usage = 0.0
        
        metric = PerformanceMetrics(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success=success,
            error_message=error_message
        )
        
        self.metrics_queue.put(metric)
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        if operation:
            metrics = self.performance_data.get(operation, [])
        else:
            metrics = []
            for op_metrics in self.performance_data.values():
                metrics.extend(op_metrics)
        
        if not metrics:
            return {"message": "No performance data available"}
        
        durations = [m.duration for m in metrics]
        memory_usage = [m.memory_usage for m in metrics]
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        
        return {
            "total_operations": len(metrics),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "max_memory_mb": max(memory_usage),
            "success_rate": success_rate,
            "error_rate": 1.0 - success_rate
        }


class RobustEvolutionEngine:
    """Robust evolution engine with comprehensive error handling and monitoring."""
    
    def __init__(self):
        self.logger = SecureLogger()
        self.security_manager = SecurityManager(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.is_initialized = False
        
        try:
            self.initialize()
            self.is_initialized = True
            self.logger.app_logger.info("RobustEvolutionEngine initialized successfully")
        except Exception as e:
            self.logger.app_logger.critical(f"Failed to initialize RobustEvolutionEngine: {e}")
            raise
    
    def initialize(self):
        """Initialize engine with error handling."""
        # Validate environment
        self.validate_environment()
        
        # Setup security
        self.setup_security()
        
        # Initialize monitoring
        self.setup_monitoring()
    
    def validate_environment(self):
        """Validate runtime environment."""
        # Check Python version
        import sys
        if sys.version_info < (3, 7):
            raise RuntimeError("Python 3.7 or higher required")
        
        # Check required directories
        required_dirs = ["logs", "cache", "backups"]
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Validate permissions
        test_file = Path("logs/permission_test.txt")
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise RuntimeError(f"Insufficient file permissions: {e}")
    
    def setup_security(self):
        """Setup security configurations."""
        # Generate system API key
        system_key = self.security_manager.generate_api_key(
            "system", 
            ["evolution", "monitoring", "administration"]
        )
        
        self.logger.app_logger.info("Security manager initialized")
    
    def setup_monitoring(self):
        """Setup monitoring and alerting."""
        self.logger.app_logger.info("Performance monitoring initialized")
    
    async def secure_evolve_prompts(self, 
                                   user_id: str,
                                   api_key: str,
                                   seed_prompts: List[str],
                                   generations: int = 10,
                                   fitness_evaluator: Optional[Callable] = None) -> Dict[str, Any]:
        """Secure prompt evolution with comprehensive error handling."""
        
        operation_start = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            # Security validation
            await self.validate_request(user_id, api_key, seed_prompts)
            
            # Rate limiting
            if not self.security_manager.check_rate_limit(user_id, "evolve_prompts"):
                raise SecurityError("Rate limit exceeded")
            
            # Input validation and sanitization
            validated_prompts = await self.validate_and_sanitize_inputs(seed_prompts)
            
            # Audit logging
            self.logger.log_audit_event(
                operation="evolve_prompts",
                user_id=user_id,
                resource_id=operation_id,
                action="START",
                result="SUCCESS",
                metadata={"generations": generations, "prompt_count": len(seed_prompts)}
            )
            
            # Execute evolution with monitoring
            results = await self.monitored_evolution(
                validated_prompts, 
                generations, 
                fitness_evaluator,
                operation_id
            )
            
            # Success audit
            self.logger.log_audit_event(
                operation="evolve_prompts",
                user_id=user_id,
                resource_id=operation_id,
                action="COMPLETE",
                result="SUCCESS",
                metadata={"final_best_fitness": results.get("best_fitness", 0.0)}
            )
            
            # Record performance
            self.performance_monitor.record_operation(
                "evolve_prompts",
                operation_start,
                success=True
            )
            
            return {
                "status": "success",
                "operation_id": operation_id,
                "results": results,
                "security_classification": SecurityLevel.INTERNAL.value,
                "processing_time": time.time() - operation_start
            }
            
        except Exception as e:
            # Error handling
            error_message = str(e)
            self.logger.app_logger.error(f"Evolution failed for operation {operation_id}: {error_message}")
            
            # Security event for errors
            self.logger.log_security_event(
                "OPERATION_FAILED",
                "ERROR",
                "RobustEvolutionEngine",
                {"operation_id": operation_id, "error": error_message, "user_id": user_id}
            )
            
            # Audit failure
            self.logger.log_audit_event(
                operation="evolve_prompts",
                user_id=user_id,
                resource_id=operation_id,
                action="FAILED",
                result="ERROR",
                metadata={"error": error_message}
            )
            
            # Record performance
            self.performance_monitor.record_operation(
                "evolve_prompts",
                operation_start,
                success=False,
                error_message=error_message
            )
            
            return {
                "status": "error",
                "operation_id": operation_id,
                "error": error_message,
                "error_code": self.classify_error(e),
                "processing_time": time.time() - operation_start
            }
    
    async def validate_request(self, user_id: str, api_key: str, seed_prompts: List[str]):
        """Comprehensive request validation."""
        # API key validation
        key_data = self.security_manager.validate_api_key(api_key)
        if not key_data:
            raise SecurityError("Invalid API key")
        
        if key_data["user_id"] != user_id:
            raise SecurityError("API key does not match user ID")
        
        # User authorization
        if "evolution" not in key_data.get("scope", []):
            raise SecurityError("Insufficient permissions for evolution operations")
        
        # Input validation
        if not seed_prompts:
            raise ValidationError("No seed prompts provided")
        
        if len(seed_prompts) > 100:
            raise ValidationError("Too many seed prompts (max 100)")
    
    async def validate_and_sanitize_inputs(self, seed_prompts: List[str]) -> List[str]:
        """Validate and sanitize input prompts."""
        validated_prompts = []
        
        for i, prompt in enumerate(seed_prompts):
            validation_result = self.security_manager.validate_input(prompt, "prompt")
            
            if not validation_result["valid"]:
                self.logger.log_security_event(
                    "INVALID_INPUT_DETECTED",
                    validation_result["risk_level"],
                    "RobustEvolutionEngine",
                    {
                        "prompt_index": i,
                        "issues": validation_result["issues"],
                        "risk_level": validation_result["risk_level"]
                    }
                )
                
                if validation_result["risk_level"] == "CRITICAL":
                    raise SecurityError(f"Critical security issue in prompt {i}: {validation_result['issues']}")
                
                # Skip invalid prompts
                continue
            
            validated_prompts.append(validation_result["sanitized_data"])
        
        if not validated_prompts:
            raise ValidationError("No valid prompts after security validation")
        
        return validated_prompts
    
    async def monitored_evolution(self, 
                                 prompts: List[str], 
                                 generations: int,
                                 fitness_evaluator: Optional[Callable],
                                 operation_id: str) -> Dict[str, Any]:
        """Execute evolution with comprehensive monitoring."""
        
        evolution_start = time.time()
        
        try:
            # Initialize evolution with error handling
            population = await self.safe_initialize_population(prompts)
            
            best_fitness_history = []
            generation_metrics = []
            
            # Evolution loop with monitoring
            for generation in range(generations):
                gen_start = time.time()
                
                try:
                    # Safe generation evolution
                    population = await self.safe_evolve_generation(
                        population, 
                        fitness_evaluator,
                        generation,
                        operation_id
                    )
                    
                    # Track metrics
                    best_fitness = max(p.get("fitness", 0.0) for p in population)
                    best_fitness_history.append(best_fitness)
                    
                    gen_metrics = {
                        "generation": generation + 1,
                        "best_fitness": best_fitness,
                        "population_size": len(population),
                        "duration": time.time() - gen_start
                    }
                    generation_metrics.append(gen_metrics)
                    
                    self.logger.app_logger.info(
                        f"Generation {generation + 1}/{generations} completed: "
                        f"Best fitness: {best_fitness:.4f}, "
                        f"Duration: {gen_metrics['duration']:.2f}s"
                    )
                    
                    # Check for performance issues
                    if gen_metrics["duration"] > 60.0:  # 1 minute threshold
                        self.logger.log_security_event(
                            "SLOW_GENERATION",
                            "WARNING",
                            "RobustEvolutionEngine",
                            {
                                "operation_id": operation_id,
                                "generation": generation + 1,
                                "duration": gen_metrics["duration"]
                            }
                        )
                
                except Exception as e:
                    self.logger.app_logger.error(f"Generation {generation + 1} failed: {e}")
                    # Continue with next generation if possible
                    continue
            
            # Compile final results
            results = {
                "best_prompts": sorted(population, key=lambda p: p.get("fitness", 0.0), reverse=True)[:10],
                "best_fitness": max(best_fitness_history) if best_fitness_history else 0.0,
                "fitness_history": best_fitness_history,
                "generation_metrics": generation_metrics,
                "total_generations": len(generation_metrics),
                "total_duration": time.time() - evolution_start,
                "final_population_size": len(population)
            }
            
            return results
            
        except Exception as e:
            self.logger.app_logger.error(f"Monitored evolution failed: {e}")
            raise EvolutionError(f"Evolution process failed: {e}")
    
    async def safe_initialize_population(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Safely initialize population with error handling."""
        population = []
        
        for i, prompt_text in enumerate(prompts):
            try:
                prompt = {
                    "id": str(uuid.uuid4()),
                    "text": prompt_text,
                    "fitness": 0.0,
                    "generation": 0,
                    "lineage": [],
                    "created_at": time.time()
                }
                population.append(prompt)
                
            except Exception as e:
                self.logger.app_logger.warning(f"Failed to initialize prompt {i}: {e}")
                continue
        
        if not population:
            raise EvolutionError("Failed to initialize any prompts")
        
        return population
    
    async def safe_evolve_generation(self, 
                                    population: List[Dict[str, Any]], 
                                    fitness_evaluator: Optional[Callable],
                                    generation: int,
                                    operation_id: str) -> List[Dict[str, Any]]:
        """Safely evolve one generation with error handling."""
        
        try:
            # Evaluate fitness with error handling
            await self.safe_evaluate_fitness(population, fitness_evaluator)
            
            # Selection with validation
            selected = self.safe_selection(population)
            
            # Reproduction with error handling
            offspring = await self.safe_reproduction(selected, generation + 1)
            
            # Mutation with validation
            mutated = await self.safe_mutation(offspring)
            
            return mutated
            
        except Exception as e:
            self.logger.app_logger.error(f"Generation evolution failed: {e}")
            # Return original population if evolution fails
            return population
    
    async def safe_evaluate_fitness(self, 
                                   population: List[Dict[str, Any]], 
                                   fitness_evaluator: Optional[Callable]):
        """Safely evaluate fitness with comprehensive error handling."""
        
        if not fitness_evaluator:
            # Default fitness function
            fitness_evaluator = self.default_fitness_function
        
        for prompt in population:
            if prompt.get("fitness", 0.0) > 0.0:
                continue  # Skip already evaluated
            
            try:
                # Safe fitness evaluation with timeout
                fitness_score = await asyncio.wait_for(
                    self.async_fitness_evaluation(prompt["text"], fitness_evaluator),
                    timeout=30.0
                )
                
                prompt["fitness"] = max(0.0, min(1.0, fitness_score))
                
            except asyncio.TimeoutError:
                self.logger.app_logger.warning(f"Fitness evaluation timeout for prompt {prompt['id']}")
                prompt["fitness"] = 0.1  # Assign low fitness
                
            except Exception as e:
                self.logger.app_logger.warning(f"Fitness evaluation error for prompt {prompt['id']}: {e}")
                prompt["fitness"] = 0.1  # Assign low fitness
    
    async def async_fitness_evaluation(self, prompt_text: str, fitness_evaluator: Callable) -> float:
        """Asynchronous fitness evaluation wrapper."""
        try:
            # Check if evaluator is async
            if asyncio.iscoroutinefunction(fitness_evaluator):
                return await fitness_evaluator(prompt_text)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, fitness_evaluator, prompt_text)
        except Exception as e:
            self.logger.app_logger.error(f"Fitness evaluation failed: {e}")
            return 0.1
    
    def default_fitness_function(self, prompt_text: str) -> float:
        """Default robust fitness function."""
        try:
            score = 0.0
            words = prompt_text.split()
            
            # Length component
            optimal_length = 12
            length_score = 1.0 - abs(len(words) - optimal_length) / optimal_length
            score += max(0, length_score) * 0.3
            
            # Quality indicators
            quality_words = ["analyze", "explain", "describe", "help", "solve", "understand"]
            quality_score = sum(1 for word in quality_words if word.lower() in prompt_text.lower())
            score += min(1.0, quality_score / 3) * 0.4
            
            # Structure component
            has_structure = any(marker in prompt_text.lower() for marker in [":", "?", "step", "how"])
            score += 0.3 if has_structure else 0.0
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.app_logger.error(f"Default fitness function error: {e}")
            return 0.1
    
    def safe_selection(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Safe selection with error handling."""
        try:
            if not population:
                return []
            
            # Sort by fitness
            sorted_pop = sorted(population, key=lambda p: p.get("fitness", 0.0), reverse=True)
            
            # Select top 50%
            select_count = max(1, len(sorted_pop) // 2)
            return sorted_pop[:select_count]
            
        except Exception as e:
            self.logger.app_logger.error(f"Selection failed: {e}")
            return population[:len(population)//2] if population else []
    
    async def safe_reproduction(self, 
                               selected: List[Dict[str, Any]], 
                               generation: int) -> List[Dict[str, Any]]:
        """Safe reproduction with error handling."""
        offspring = []
        target_size = len(selected) * 2  # Double the population
        
        try:
            while len(offspring) < target_size:
                if len(selected) >= 2:
                    parent1 = selected[len(offspring) % len(selected)]
                    parent2 = selected[(len(offspring) + 1) % len(selected)]
                    
                    child = await self.safe_crossover(parent1, parent2, generation)
                    offspring.append(child)
                else:
                    # If not enough parents, clone existing
                    if selected:
                        clone = selected[0].copy()
                        clone["id"] = str(uuid.uuid4())
                        clone["generation"] = generation
                        offspring.append(clone)
                    break
            
            return offspring
            
        except Exception as e:
            self.logger.app_logger.error(f"Reproduction failed: {e}")
            return selected  # Return original selection
    
    async def safe_crossover(self, 
                            parent1: Dict[str, Any], 
                            parent2: Dict[str, Any], 
                            generation: int) -> Dict[str, Any]:
        """Safe crossover with error handling."""
        try:
            words1 = parent1["text"].split()
            words2 = parent2["text"].split()
            
            if not words1 and not words2:
                child_text = "help me solve this problem"
            elif not words1:
                child_text = parent2["text"]
            elif not words2:
                child_text = parent1["text"]
            else:
                # Safe crossover
                split_point = len(words1) // 2
                new_words = words1[:split_point] + words2[split_point:]
                child_text = " ".join(new_words)
            
            child = {
                "id": str(uuid.uuid4()),
                "text": child_text,
                "fitness": 0.0,
                "generation": generation,
                "lineage": [parent1["id"], parent2["id"]],
                "created_at": time.time()
            }
            
            return child
            
        except Exception as e:
            self.logger.app_logger.error(f"Crossover failed: {e}")
            # Return a copy of parent1 as fallback
            fallback = parent1.copy()
            fallback["id"] = str(uuid.uuid4())
            fallback["generation"] = generation
            return fallback
    
    async def safe_mutation(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Safe mutation with error handling."""
        mutated = []
        
        for prompt in population:
            try:
                if random.random() < 0.1:  # 10% mutation rate
                    mutated_prompt = await self.safe_mutate_prompt(prompt)
                    mutated.append(mutated_prompt)
                else:
                    mutated.append(prompt)
                    
            except Exception as e:
                self.logger.app_logger.warning(f"Mutation failed for prompt {prompt.get('id', 'unknown')}: {e}")
                mutated.append(prompt)  # Keep original
        
        return mutated
    
    async def safe_mutate_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Safe mutation of individual prompt."""
        try:
            words = prompt["text"].split()
            if not words:
                return prompt
            
            # Safe mutations
            mutation_type = random.choice(["substitute", "insert", "delete"])
            
            if mutation_type == "substitute" and words:
                idx = random.randint(0, len(words) - 1)
                enhancement_words = ["enhanced", "improved", "better", "effective", "optimized"]
                words[idx] = random.choice(enhancement_words)
                
            elif mutation_type == "insert":
                insert_words = ["please", "help", "efficiently", "carefully"]
                idx = random.randint(0, len(words))
                words.insert(idx, random.choice(insert_words))
                
            elif mutation_type == "delete" and len(words) > 1:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
            
            mutated_text = " ".join(words)
            
            mutated = prompt.copy()
            mutated["text"] = mutated_text
            mutated["fitness"] = 0.0  # Reset fitness
            mutated["id"] = str(uuid.uuid4())
            
            return mutated
            
        except Exception as e:
            self.logger.app_logger.error(f"Prompt mutation failed: {e}")
            return prompt
    
    def classify_error(self, error: Exception) -> str:
        """Classify error types for better handling."""
        if isinstance(error, SecurityError):
            return "SECURITY_ERROR"
        elif isinstance(error, ValidationError):
            return "VALIDATION_ERROR"
        elif isinstance(error, EvolutionError):
            return "EVOLUTION_ERROR"
        elif isinstance(error, asyncio.TimeoutError):
            return "TIMEOUT_ERROR"
        elif isinstance(error, MemoryError):
            return "MEMORY_ERROR"
        else:
            return "UNKNOWN_ERROR"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system": {
                "initialized": self.is_initialized,
                "uptime": time.time() - getattr(self, 'start_time', time.time()),
                "version": "2.0.0-robust"
            },
            "security": {
                "active_sessions": len(self.security_manager.session_tokens),
                "api_keys_issued": len(self.security_manager.api_keys),
                "blocked_ips": len(self.security_manager.blocked_ips)
            },
            "performance": self.performance_monitor.get_performance_summary(),
            "logs": {
                "log_directory": str(self.logger.log_dir),
                "security_events_processed": self.logger.security_events.qsize(),
                "audit_logs_processed": self.logger.audit_logs.qsize()
            }
        }


# Custom exception classes
class SecurityError(Exception):
    """Security-related errors."""
    pass


class ValidationError(Exception):
    """Input validation errors."""
    pass


class EvolutionError(Exception):
    """Evolution process errors."""
    pass


# Demonstration function
async def demonstrate_robust_system():
    """Demonstrate robust security and monitoring system."""
    print("🛡️ GENERATION 2: ROBUST SECURITY & MONITORING SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize robust engine
        engine = RobustEvolutionEngine()
        
        # Generate test API key
        test_user = "test_user_001"
        api_key = engine.security_manager.generate_api_key(
            test_user, 
            ["evolution", "monitoring"]
        )
        
        print(f"✅ System initialized successfully")
        print(f"🔑 Test API key generated: {api_key[:16]}...")
        
        # Test seed prompts
        seed_prompts = [
            "Help me analyze this complex problem step by step",
            "Explain the solution clearly and comprehensively",
            "Provide detailed insights into the analysis process",
            "Guide me through understanding this concept"
        ]
        
        print(f"📊 Starting secure evolution with {len(seed_prompts)} seed prompts")
        
        # Execute secure evolution
        results = await engine.secure_evolve_prompts(
            user_id=test_user,
            api_key=api_key,
            seed_prompts=seed_prompts,
            generations=5,
            fitness_evaluator=None  # Use default
        )
        
        print(f"✅ Evolution completed: {results['status']}")
        print(f"⏱️ Processing time: {results['processing_time']:.2f}s")
        
        if results["status"] == "success":
            evolution_results = results["results"]
            print(f"🏆 Best fitness achieved: {evolution_results['best_fitness']:.4f}")
            print(f"📈 Generations completed: {evolution_results['total_generations']}")
            
            print("\n🏅 TOP 3 EVOLVED PROMPTS:")
            for i, prompt in enumerate(evolution_results["best_prompts"][:3], 1):
                print(f"{i}. [{prompt['fitness']:.3f}] {prompt['text']}")
        
        # System status
        status = engine.get_system_status()
        print(f"\n📊 SYSTEM STATUS:")
        print(f"   Uptime: {status['system']['uptime']:.1f}s")
        print(f"   Active Sessions: {status['security']['active_sessions']}")
        print(f"   Performance Operations: {status['performance'].get('total_operations', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Import required modules
    import random
    from collections import defaultdict
    
    # Run demonstration
    results = asyncio.run(demonstrate_robust_system())
    
    # Save results
    timestamp = int(time.time())
    filename = f"robust_security_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to {filename}")
    print("🛡️ Generation 2 Robust System demonstration complete!")