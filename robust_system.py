#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Comprehensive error handling, security, and monitoring.
Implements production-ready reliability features for the evolutionary prompt system.
"""

import time
import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import warnings
import traceback


# Security and Validation Framework
class SecurityLevel(Enum):
    """Security levels for different operational contexts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration for the system."""
    level: SecurityLevel = SecurityLevel.MEDIUM
    max_prompt_length: int = 10000
    max_population_size: int = 1000
    max_generations: int = 1000
    allowed_mutation_types: List[str] = field(default_factory=lambda: [
        "add_modifier", "reorder", "substitute", "extend"
    ])
    input_sanitization: bool = True
    output_validation: bool = True
    audit_logging: bool = True
    rate_limiting: bool = True


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


class EvolutionError(Exception):
    """Custom exception for evolution process errors."""
    pass


class SecurityValidator:
    """Comprehensive security validation system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_patterns = [
            # Potential injection patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            # System command patterns
            r"system\(",
            r"exec\(",
            r"eval\(",
            r"__import__",
            # File system patterns
            r"\.\.\/",
            r"\/etc\/",
            r"\/proc\/",
        ]
    
    def validate_prompt(self, prompt_text: str) -> bool:
        """Validate prompt text for security issues."""
        if not isinstance(prompt_text, str):
            raise ValidationError("Prompt must be a string")
        
        if len(prompt_text) > self.config.max_prompt_length:
            raise SecurityError(f"Prompt exceeds maximum length of {self.config.max_prompt_length}")
        
        if self.config.input_sanitization:
            self._check_malicious_patterns(prompt_text)
        
        return True
    
    def validate_population_parameters(self, population_size: int, generations: int):
        """Validate population parameters."""
        if population_size > self.config.max_population_size:
            raise SecurityError(f"Population size {population_size} exceeds limit of {self.config.max_population_size}")
        
        if generations > self.config.max_generations:
            raise SecurityError(f"Generations {generations} exceeds limit of {self.config.max_generations}")
        
        if population_size <= 0 or generations <= 0:
            raise ValidationError("Population size and generations must be positive")
    
    def _check_malicious_patterns(self, text: str):
        """Check for potentially malicious patterns."""
        import re
        text_lower = text.lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                raise SecurityError(f"Potentially malicious pattern detected: {pattern}")
    
    def sanitize_output(self, text: str) -> str:
        """Sanitize output text."""
        if not self.config.output_validation:
            return text
        
        # Remove potentially dangerous characters
        sanitized = text.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        
        return sanitized


# Comprehensive Logging and Monitoring System
class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    request_id: str = ""


class RobustLogger:
    """Production-ready logging system with structured output."""
    
    def __init__(self, log_file: Optional[str] = None, log_level: LogLevel = LogLevel.INFO):
        self.log_file = log_file
        self.min_level = log_level
        self.session_id = str(uuid.uuid4())[:8]
        self.entries = []
        
        # Setup Python logging
        logging.basicConfig(
            level=getattr(logging, log_level.value),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, level: LogLevel, component: str, message: str, 
            metadata: Optional[Dict[str, Any]] = None, request_id: str = ""):
        """Log a structured entry."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            component=component,
            message=message,
            metadata=metadata or {},
            session_id=self.session_id,
            request_id=request_id
        )
        
        self.entries.append(entry)
        
        # Log to Python logger
        log_method = getattr(self.logger, level.value.lower())
        log_message = f"[{component}] {message}"
        if metadata:
            log_message += f" | Metadata: {json.dumps(metadata, default=str)}"
        
        log_method(log_message)
        
        # Write to file if specified
        if self.log_file:
            self._write_to_file(entry)
    
    def info(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.log(LogLevel.INFO, component, message, metadata)
    
    def warning(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.log(LogLevel.WARNING, component, message, metadata)
    
    def error(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.log(LogLevel.ERROR, component, message, metadata)
    
    def critical(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log critical message.""" 
        self.log(LogLevel.CRITICAL, component, message, metadata)
    
    def _write_to_file(self, entry: LogEntry):
        """Write entry to log file."""
        try:
            log_data = {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "component": entry.component,
                "message": entry.message,
                "metadata": entry.metadata,
                "session_id": entry.session_id,
                "request_id": entry.request_id
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_data) + "\\n")
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {e}")


# Health Monitoring and Metrics System
@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_populations: int = 0
    total_evaluations: int = 0
    error_rate: float = 0.0
    average_response_time: float = 0.0
    successful_operations: int = 0
    failed_operations: int = 0


class HealthMonitor:
    """Health monitoring and alerting system."""
    
    def __init__(self, logger: RobustLogger):
        self.logger = logger
        self.metrics_history = []
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10%
            "memory_usage": 0.8,  # 80%
            "response_time": 5.0   # 5 seconds
        }
        self.alerts_sent = set()
    
    def record_metrics(self, metrics: HealthMetrics):
        """Record health metrics."""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        self._check_alerts(metrics)
        
        self.logger.info("health_monitor", "Metrics recorded", {
            "error_rate": metrics.error_rate,
            "memory_usage": metrics.memory_usage,
            "response_time": metrics.average_response_time
        })
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check for alert conditions."""
        alerts = []
        
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_usage:.2%}")
        
        if metrics.average_response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"High response time: {metrics.average_response_time:.2f}s")
        
        for alert in alerts:
            alert_key = hashlib.md5(alert.encode()).hexdigest()
            if alert_key not in self.alerts_sent:
                self.logger.critical("health_monitor", f"ALERT: {alert}")
                self.alerts_sent.add(alert_key)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 entries
        
        return {
            "status": "healthy" if not self.alerts_sent else "alerts_active",
            "active_alerts": len(self.alerts_sent),
            "average_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "average_response_time": sum(m.average_response_time for m in recent_metrics) / len(recent_metrics),
            "total_operations": sum(m.successful_operations + m.failed_operations for m in recent_metrics),
            "uptime_metrics": len(self.metrics_history)
        }


# Robust Error Handling Framework
class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, logger: RobustLogger, max_retries: int = 3):
        self.logger = logger
        self.max_retries = max_retries
        self.error_counts = {}
    
    def handle_with_retry(self, operation: Callable, operation_name: str, 
                         *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    self.logger.info("error_handler", 
                                   f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                return result
            
            except Exception as e:
                self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
                
                if attempt == self.max_retries:
                    self.logger.error("error_handler", 
                                    f"Operation {operation_name} failed after {self.max_retries + 1} attempts", 
                                    {"error": str(e), "traceback": traceback.format_exc()})
                    raise
                else:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning("error_handler", 
                                      f"Operation {operation_name} failed on attempt {attempt + 1}, retrying in {wait_time}s",
                                      {"error": str(e)})
                    time.sleep(wait_time)
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()


# Robust Evolution System
class RobustEvolutionEngine:
    """Production-ready evolution engine with comprehensive error handling."""
    
    def __init__(self, security_config: Optional[SecurityConfig] = None,
                 log_file: Optional[str] = None):
        self.security_config = security_config or SecurityConfig()
        self.logger = RobustLogger(log_file, LogLevel.INFO)
        self.validator = SecurityValidator(self.security_config)
        self.error_handler = ErrorHandler(self.logger)
        self.health_monitor = HealthMonitor(self.logger)
        
        # Evolution parameters
        self.population_size = 20
        self.generations = 10
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8
        
        # Performance tracking
        self.start_time = None
        self.operations_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
        
        self.logger.info("robust_evolution", "RobustEvolutionEngine initialized", {
            "security_level": self.security_config.level.value,
            "audit_logging": self.security_config.audit_logging
        })
    
    def evolve_safely(self, initial_prompts: List[str], test_scenarios: List[Dict[str, Any]],
                     population_size: int = 20, generations: int = 10) -> Dict[str, Any]:
        """Safely evolve prompts with full error handling and monitoring."""
        request_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(initial_prompts, test_scenarios, population_size, generations)
            
            # Initialize population
            population = self._initialize_population_safely(initial_prompts, population_size)
            
            # Run evolution with monitoring
            results = self._run_evolution_with_monitoring(population, test_scenarios, 
                                                        generations, request_id)
            
            # Record success metrics
            self._record_success_metrics()
            
            self.logger.info("robust_evolution", "Evolution completed successfully", {
                "request_id": request_id,
                "generations": generations,
                "population_size": len(population),
                "execution_time": time.time() - self.start_time
            })
            
            return results
            
        except Exception as e:
            self._record_failure_metrics()
            self.logger.error("robust_evolution", "Evolution failed", {
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            raise EvolutionError(f"Evolution process failed: {e}") from e
    
    def _validate_inputs(self, prompts: List[str], scenarios: List[Dict[str, Any]], 
                        pop_size: int, generations: int):
        """Validate all inputs thoroughly."""
        if not prompts:
            raise ValidationError("Initial prompts list cannot be empty")
        
        if not scenarios:
            raise ValidationError("Test scenarios list cannot be empty")
        
        # Validate each prompt
        for i, prompt in enumerate(prompts):
            try:
                self.validator.validate_prompt(prompt)
            except (ValidationError, SecurityError) as e:
                raise ValidationError(f"Prompt {i} validation failed: {e}")
        
        # Validate parameters
        self.validator.validate_population_parameters(pop_size, generations)
        
        # Validate scenarios structure
        for i, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                raise ValidationError(f"Scenario {i} must be a dictionary")
            if "input" not in scenario or "expected" not in scenario:
                raise ValidationError(f"Scenario {i} must have 'input' and 'expected' fields")
    
    def _initialize_population_safely(self, initial_prompts: List[str], 
                                    target_size: int) -> List[Dict[str, Any]]:
        """Initialize population with error handling."""
        def init_operation():
            population = []
            
            # Add initial prompts
            for prompt in initial_prompts:
                sanitized_prompt = self.validator.sanitize_output(prompt)
                population.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": sanitized_prompt,
                    "fitness": 0.0,
                    "generation": 0,
                    "parent_ids": [],
                    "mutations_applied": []
                })
            
            # Expand population if needed
            while len(population) < target_size:
                base_prompt = population[len(population) % len(initial_prompts)]
                try:
                    mutated = self._safe_mutation(base_prompt)
                    population.append(mutated)
                except Exception as e:
                    self.logger.warning("population_init", f"Mutation failed: {e}")
                    # Fallback: duplicate existing prompt
                    population.append(base_prompt.copy())
            
            return population[:target_size]
        
        return self.error_handler.handle_with_retry(init_operation, "initialize_population")
    
    def _run_evolution_with_monitoring(self, population: List[Dict[str, Any]], 
                                     scenarios: List[Dict[str, Any]], 
                                     generations: int, request_id: str) -> Dict[str, Any]:
        """Run evolution with comprehensive monitoring."""
        evolution_history = []
        
        for generation in range(generations):
            gen_start_time = time.time()
            
            try:
                # Evaluate fitness
                self._evaluate_population_safely(population, scenarios)
                
                # Track best fitness
                best_fitness = max(ind["fitness"] for ind in population)
                
                # Create next generation (except for last generation)
                if generation < generations - 1:
                    population = self._create_next_generation_safely(population)
                
                # Record generation metrics
                gen_time = time.time() - gen_start_time
                diversity = self._calculate_diversity_safely(population)
                
                generation_data = {
                    "generation": generation + 1,
                    "best_fitness": best_fitness,
                    "average_fitness": sum(ind["fitness"] for ind in population) / len(population),
                    "diversity": diversity,
                    "execution_time": gen_time,
                    "population_size": len(population)
                }
                
                evolution_history.append(generation_data)
                
                # Health monitoring
                health_metrics = HealthMetrics(
                    timestamp=datetime.now(timezone.utc),
                    active_populations=1,
                    total_evaluations=len(population),
                    average_response_time=gen_time,
                    successful_operations=self.successful_operations,
                    failed_operations=self.failed_operations,
                    error_rate=self.failed_operations / max(1, self.operations_count)
                )
                
                self.health_monitor.record_metrics(health_metrics)
                
                self.logger.info("evolution_generation", f"Generation {generation + 1} completed", {
                    "request_id": request_id,
                    "best_fitness": best_fitness,
                    "diversity": diversity,
                    "execution_time": gen_time
                })
                
            except Exception as e:
                self.logger.error("evolution_generation", f"Generation {generation + 1} failed", {
                    "request_id": request_id,
                    "error": str(e)
                })
                raise
        
        # Compile final results
        top_prompts = sorted(population, key=lambda x: x["fitness"], reverse=True)[:10]
        
        return {
            "evolution_history": evolution_history,
            "final_population": population,
            "top_prompts": top_prompts,
            "health_summary": self.health_monitor.get_health_summary(),
            "error_statistics": self.error_handler.get_error_statistics(),
            "security_summary": {
                "security_level": self.security_config.level.value,
                "validations_performed": True,
                "sanitization_applied": self.security_config.input_sanitization
            },
            "performance_metrics": {
                "total_time": time.time() - self.start_time,
                "successful_operations": self.successful_operations,
                "failed_operations": self.failed_operations,
                "success_rate": self.successful_operations / max(1, self.operations_count)
            }
        }
    
    def _evaluate_population_safely(self, population: List[Dict[str, Any]], 
                                   scenarios: List[Dict[str, Any]]):
        """Evaluate population fitness with error handling."""
        for individual in population:
            def evaluate_operation():
                # Simple fitness evaluation (can be replaced with more sophisticated methods)
                total_score = 0.0
                for scenario in scenarios:
                    score = self._score_prompt_safely(individual["text"], scenario)
                    weight = scenario.get("weight", 1.0)
                    total_score += score * weight
                
                total_weight = sum(scenario.get("weight", 1.0) for scenario in scenarios)
                individual["fitness"] = total_score / total_weight if total_weight > 0 else 0.0
                return individual["fitness"]
            
            try:
                self.error_handler.handle_with_retry(evaluate_operation, "evaluate_fitness")
                self.successful_operations += 1
            except Exception:
                individual["fitness"] = 0.0  # Fallback fitness
                self.failed_operations += 1
            
            self.operations_count += 1
    
    def _score_prompt_safely(self, prompt_text: str, scenario: Dict[str, Any]) -> float:
        """Score prompt with comprehensive error handling."""
        try:
            # Validate prompt before scoring
            self.validator.validate_prompt(prompt_text)
            
            # Simple heuristic scoring (can be extended)
            score = 0.5  # Base score
            
            # Length analysis
            words = prompt_text.split()
            if 5 <= len(words) <= 30:
                score += 0.2
            
            # Keyword relevance
            expected_lower = scenario["expected"].lower()
            prompt_lower = prompt_text.lower()
            
            relevant_words = {"help", "assist", "provide", "explain", "analyze", "describe"}
            matches = sum(1 for word in relevant_words if word in prompt_lower)
            score += matches * 0.1
            
            # Context relevance
            if "explain" in scenario["input"].lower() and "explain" in prompt_lower:
                score += 0.15
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.warning("fitness_evaluation", f"Scoring failed: {e}")
            return 0.0
    
    def _create_next_generation_safely(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create next generation with comprehensive error handling."""
        def generation_operation():
            # Sort by fitness
            sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
            
            # Keep elite individuals (top 30%)
            elite_count = max(1, int(len(sorted_pop) * 0.3))
            next_generation = sorted_pop[:elite_count].copy()
            
            # Fill remaining with mutations and crossovers
            while len(next_generation) < len(population):
                try:
                    if len(next_generation) < len(population) // 2:
                        # More mutations
                        parent = self._tournament_selection(sorted_pop[:len(sorted_pop)//2])
                        child = self._safe_mutation(parent)
                    else:
                        # Crossovers
                        parent1 = self._tournament_selection(sorted_pop[:len(sorted_pop)//2])
                        parent2 = self._tournament_selection(sorted_pop[:len(sorted_pop)//2])
                        child = self._safe_crossover(parent1, parent2)
                    
                    next_generation.append(child)
                    
                except Exception as e:
                    self.logger.warning("generation_creation", f"Reproduction failed: {e}")
                    # Fallback: duplicate elite individual
                    next_generation.append(sorted_pop[0].copy())
            
            return next_generation[:len(population)]
        
        return self.error_handler.handle_with_retry(generation_operation, "create_generation")
    
    def _safe_mutation(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safe mutation with validation."""
        import random
        
        words = parent["text"].split()
        mutated_words = words.copy()
        
        # Safe mutation operations
        if "add_modifier" in self.security_config.allowed_mutation_types:
            modifiers = ["carefully", "clearly", "precisely", "thoroughly", "effectively"]
            if random.random() < 0.5:
                pos = random.randint(0, len(mutated_words))
                mutated_words.insert(pos, random.choice(modifiers))
        
        mutated_text = " ".join(mutated_words)
        
        # Validate mutated text
        self.validator.validate_prompt(mutated_text)
        sanitized_text = self.validator.sanitize_output(mutated_text)
        
        return {
            "id": str(uuid.uuid4())[:8],
            "text": sanitized_text,
            "fitness": 0.0,
            "generation": parent["generation"] + 1,
            "parent_ids": [parent["id"]],
            "mutations_applied": ["add_modifier"]
        }
    
    def _safe_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safe crossover with validation."""
        import random
        
        words1 = parent1["text"].split()
        words2 = parent2["text"].split()
        
        if len(words1) < 2 or len(words2) < 2:
            return self._safe_mutation(parent1)
        
        crossover_point = random.randint(1, min(len(words1), len(words2)) - 1)
        child_words = words1[:crossover_point] + words2[crossover_point:]
        
        child_text = " ".join(child_words)
        
        # Validate crossover result
        self.validator.validate_prompt(child_text)
        sanitized_text = self.validator.sanitize_output(child_text)
        
        return {
            "id": str(uuid.uuid4())[:8],
            "text": sanitized_text,
            "fitness": 0.0,
            "generation": max(parent1["generation"], parent2["generation"]) + 1,
            "parent_ids": [parent1["id"], parent2["id"]],
            "mutations_applied": ["crossover"]
        }
    
    def _tournament_selection(self, population: List[Dict[str, Any]], 
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection with error handling."""
        import random
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x["fitness"])
    
    def _calculate_diversity_safely(self, population: List[Dict[str, Any]]) -> float:
        """Calculate diversity with error handling."""
        try:
            if len(population) < 2:
                return 0.0
            
            total_distance = 0.0
            comparisons = 0
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    try:
                        words1 = set(population[i]["text"].lower().split())
                        words2 = set(population[j]["text"].lower().split())
                        
                        union = words1.union(words2)
                        intersection = words1.intersection(words2)
                        
                        if union:
                            jaccard_sim = len(intersection) / len(union)
                            distance = 1.0 - jaccard_sim
                            total_distance += distance
                            comparisons += 1
                    except Exception:
                        continue
            
            return total_distance / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.warning("diversity_calculation", f"Diversity calculation failed: {e}")
            return 0.0
    
    def _record_success_metrics(self):
        """Record successful operation metrics."""
        self.successful_operations += 1
        self.operations_count += 1
    
    def _record_failure_metrics(self):
        """Record failed operation metrics."""
        self.failed_operations += 1
        self.operations_count += 1


def main():
    """Demonstrate Generation 2: MAKE IT ROBUST functionality."""
    print("🔐 Meta-Prompt-Evolution-Hub - Generation 2: MAKE IT ROBUST")
    print("🛡️  Comprehensive error handling, security, and monitoring")
    print("=" * 80)
    
    # Initialize robust system
    security_config = SecurityConfig(
        level=SecurityLevel.HIGH,
        max_prompt_length=5000,
        max_population_size=100,
        input_sanitization=True,
        output_validation=True,
        audit_logging=True
    )
    
    log_file = "demo_results/robust_evolution.log"
    Path("demo_results").mkdir(exist_ok=True)
    
    engine = RobustEvolutionEngine(security_config, log_file)
    
    # Test data
    initial_prompts = [
        "You are a helpful assistant. Please help with: {task}",
        "I'll provide comprehensive assistance with your request: {task}",
        "Let me carefully analyze and help you with: {task}",
        "As a professional AI, I'll address your need: {task}"
    ]
    
    test_scenarios = [
        {
            "input": "Write a professional email",
            "expected": "formal tone, clear structure, professional language",
            "weight": 1.0
        },
        {
            "input": "Explain a complex concept simply", 
            "expected": "clear explanations, simple language, good examples",
            "weight": 1.2
        },
        {
            "input": "Analyze data and provide insights",
            "expected": "systematic analysis, actionable insights, evidence-based",
            "weight": 1.1
        }
    ]
    
    try:
        # Run robust evolution
        start_time = time.time()
        results = engine.evolve_safely(
            initial_prompts=initial_prompts,
            test_scenarios=test_scenarios,
            population_size=30,
            generations=15
        )
        total_time = time.time() - start_time
        
        # Display results
        print("\\n" + "=" * 80)
        print("🎉 GENERATION 2 COMPLETE: MAKE IT ROBUST")
        print("=" * 80)
        
        print("\\n🔐 SECURITY SUMMARY:")
        security = results["security_summary"]
        print(f"   Security Level: {security['security_level'].upper()}")
        print(f"   Validations: {'✅ Applied' if security['validations_performed'] else '❌ Skipped'}")
        print(f"   Sanitization: {'✅ Applied' if security['sanitization_applied'] else '❌ Skipped'}")
        
        print("\\n📊 PERFORMANCE METRICS:")
        perf = results["performance_metrics"]
        print(f"   Total Time: {perf['total_time']:.2f}s")
        print(f"   Success Rate: {perf['success_rate']:.2%}")
        print(f"   Successful Operations: {perf['successful_operations']}")
        print(f"   Failed Operations: {perf['failed_operations']}")
        
        print("\\n🏥 HEALTH SUMMARY:")
        health = results["health_summary"]
        print(f"   Status: {health['status'].upper()}")
        print(f"   Active Alerts: {health['active_alerts']}")
        print(f"   Error Rate: {health['average_error_rate']:.2%}")
        print(f"   Avg Response Time: {health['average_response_time']:.3f}s")
        
        print("\\n🥇 TOP 3 ROBUST PROMPTS:")
        for i, prompt in enumerate(results["top_prompts"][:3], 1):
            print(f"   {i}. (Fitness: {prompt['fitness']:.3f}) Gen: {prompt['generation']}")
            print(f"      {prompt['text'][:70]}{'...' if len(prompt['text']) > 70 else ''}")
        
        print("\\n✅ ROBUST FEATURES IMPLEMENTED:")
        print("   🔒 Input validation and sanitization")
        print("   🛡️  Security threat detection")
        print("   📝 Comprehensive audit logging")
        print("   🔄 Automatic retry with exponential backoff")
        print("   🏥 Real-time health monitoring")
        print("   ⚠️  Proactive alerting system")
        print("   🚨 Error classification and tracking")
        print("   📊 Performance metrics collection")
        
        print("\\n🔄 READY FOR GENERATION 3: MAKE IT SCALE")
        print("   Next: Performance optimization, caching, distributed processing")
        
        # Save results
        results_file = "demo_results/generation_2_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📁 Results saved to: {results_file}")
        print(f"📋 Logs available at: {log_file}")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Robust evolution failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)