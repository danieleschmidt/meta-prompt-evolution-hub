#!/usr/bin/env python3
"""
SECURITY HARDENED SYSTEM - Production-Ready Security Implementation
Addressing security violations and implementing hardened defensive measures
"""

import sys
import os
import json
import time
import hashlib
import hmac
import secrets
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
import base64
from contextlib import contextmanager

sys.path.insert(0, '.')


@dataclass
class SecurityConfig:
    """Security configuration for hardened system."""
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    max_input_length: int = 10000
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    audit_log_file: str = "security_audit.log"


class SecureInputValidator:
    """Secure input validation and sanitization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Safe patterns for prompt content
        self.safe_patterns = [
            r'^[a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)\{\}\[\]\'\"]+$',  # Basic safe chars
        ]
        
        # Dangerous patterns to reject
        self.dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
            r'shell=True',
            r'rm\s+-rf',
            r'del\s+/',
            r'DROP\s+TABLE',
            r'SELECT.*FROM.*WHERE',
            r'INSERT\s+INTO',
            r'UPDATE.*SET',
            r'\.\./',  # Path traversal
        ]
    
    def validate_prompt_input(self, prompt_text: str) -> Tuple[bool, Optional[str], str]:
        """Validate and sanitize prompt input."""
        if not isinstance(prompt_text, str):
            return False, "Input must be string", ""
        
        if len(prompt_text) > self.config.max_input_length:
            return False, f"Input too long (max {self.config.max_input_length})", ""
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}", ""
        
        # Sanitize input
        sanitized = self._sanitize_input(prompt_text)
        
        return True, None, sanitized
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        # Remove potential harmful characters
        sanitized = re.sub(r'[<>\"\'\\]', '', text)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit length after sanitization
        if len(sanitized) > self.config.max_input_length:
            sanitized = sanitized[:self.config.max_input_length]
        
        return sanitized


class SecureAuditLogger:
    """Secure audit logging system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = self._setup_logger()
        self._lock = threading.Lock()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup secure audit logger."""
        logger = logging.getLogger("SecurityAudit")
        logger.setLevel(logging.INFO)
        
        # Secure file handler
        handler = logging.FileHandler(self.config.audit_log_file, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events."""
        with self._lock:
            # Create secure log entry
            log_entry = {
                'timestamp': time.time(),
                'event_type': event_type,
                'details': details,
                'session_id': self._get_session_id()
            }
            
            self.logger.info(f"SECURITY_EVENT: {json.dumps(log_entry, separators=(',', ':'))}")
    
    def _get_session_id(self) -> str:
        """Get secure session identifier."""
        return hashlib.sha256(f"{time.time()}_{threading.get_ident()}".encode()).hexdigest()[:16]


class RateLimiter:
    """Secure rate limiting implementation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests = {}
        self._lock = threading.Lock()
    
    def is_rate_limited(self, identifier: str = "default") -> bool:
        """Check if request should be rate limited."""
        current_time = time.time()
        
        with self._lock:
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.config.rate_limit_window
            ]
            
            # Check rate limit
            if len(self.requests[identifier]) >= self.config.rate_limit_requests:
                return True
            
            # Add current request
            self.requests[identifier].append(current_time)
            return False


class SecureEvolutionSystem:
    """Security-hardened evolution system."""
    
    def __init__(self, security_config: Optional[SecurityConfig] = None):
        """Initialize secure evolution system."""
        self.security_config = security_config or SecurityConfig()
        self.validator = SecureInputValidator(self.security_config)
        self.audit_logger = SecureAuditLogger(self.security_config)
        self.rate_limiter = RateLimiter(self.security_config)
        
        self.audit_logger.log_security_event("SYSTEM_INIT", {
            "component": "SecureEvolutionSystem",
            "security_features": {
                "input_validation": self.security_config.enable_input_validation,
                "output_sanitization": self.security_config.enable_output_sanitization,
                "rate_limiting": self.security_config.enable_rate_limiting,
                "audit_logging": self.security_config.enable_audit_logging
            }
        })
    
    def secure_evolve(self, seed_prompts: List[str], request_id: str = None) -> Dict[str, Any]:
        """Perform secure evolution with comprehensive validation."""
        request_id = request_id or f"req_{secrets.token_hex(8)}"
        
        # Rate limiting check
        if self.security_config.enable_rate_limiting:
            if self.rate_limiter.is_rate_limited(request_id):
                self.audit_logger.log_security_event("RATE_LIMIT_EXCEEDED", {
                    "request_id": request_id,
                    "prompts_count": len(seed_prompts)
                })
                return {
                    "status": "RATE_LIMITED",
                    "error": "Rate limit exceeded",
                    "request_id": request_id
                }
        
        start_time = time.time()
        
        try:
            # Validate all input prompts
            validated_prompts = []
            validation_errors = []
            
            for i, prompt in enumerate(seed_prompts):
                is_valid, error, sanitized = self.validator.validate_prompt_input(prompt)
                
                if is_valid:
                    validated_prompts.append(sanitized)
                else:
                    validation_errors.append({
                        "prompt_index": i,
                        "error": error,
                        "original_prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
                    })
                    
                    self.audit_logger.log_security_event("INPUT_VALIDATION_FAILED", {
                        "request_id": request_id,
                        "prompt_index": i,
                        "error": error,
                        "prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt
                    })
            
            if not validated_prompts:
                return {
                    "status": "VALIDATION_FAILED",
                    "error": "No valid prompts after validation",
                    "validation_errors": validation_errors,
                    "request_id": request_id
                }
            
            # Log successful validation
            self.audit_logger.log_security_event("INPUT_VALIDATED", {
                "request_id": request_id,
                "original_count": len(seed_prompts),
                "validated_count": len(validated_prompts),
                "rejected_count": len(validation_errors)
            })
            
            # Perform secure evolution
            evolution_result = self._secure_evolution_process(validated_prompts, request_id)
            
            # Sanitize outputs
            if self.security_config.enable_output_sanitization:
                evolution_result = self._sanitize_evolution_output(evolution_result)
            
            evolution_result["request_id"] = request_id
            evolution_result["execution_time"] = time.time() - start_time
            evolution_result["security_metrics"] = {
                "input_prompts_processed": len(validated_prompts),
                "validation_errors": len(validation_errors),
                "security_features_enabled": {
                    "input_validation": self.security_config.enable_input_validation,
                    "output_sanitization": self.security_config.enable_output_sanitization,
                    "rate_limiting": self.security_config.enable_rate_limiting,
                    "audit_logging": self.security_config.enable_audit_logging
                }
            }
            
            self.audit_logger.log_security_event("EVOLUTION_COMPLETED", {
                "request_id": request_id,
                "status": evolution_result.get("status", "UNKNOWN"),
                "execution_time": evolution_result["execution_time"]
            })
            
            return evolution_result
            
        except Exception as e:
            self.audit_logger.log_security_event("EVOLUTION_ERROR", {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            return {
                "status": "ERROR",
                "error": "Internal processing error",  # Don't expose internal details
                "request_id": request_id,
                "execution_time": time.time() - start_time
            }
    
    def _secure_evolution_process(self, prompts: List[str], request_id: str) -> Dict[str, Any]:
        """Perform the actual evolution process with security controls."""
        try:
            # Import secure components
            from meta_prompt_evolution import PromptPopulation, Prompt
            
            # Create secure population
            population = PromptPopulation.from_seeds(prompts)
            
            # Secure fitness evaluation
            for prompt in population:
                # Use safe fitness calculation
                fitness = self._calculate_secure_fitness(prompt.text)
                prompt.fitness_scores = fitness
            
            # Secure selection and mutation
            top_prompts = population.get_top_k(min(5, len(population)))
            
            evolved_prompts = []
            for prompt in top_prompts:
                mutations = self._secure_mutation(prompt)
                evolved_prompts.extend(mutations)
            
            # Validate evolved prompts
            final_prompts = []
            for prompt in evolved_prompts:
                is_valid, error, sanitized = self.validator.validate_prompt_input(prompt.text)
                if is_valid:
                    prompt.text = sanitized
                    prompt.fitness_scores = self._calculate_secure_fitness(sanitized)
                    final_prompts.append(prompt)
            
            # Return secure results
            return {
                "status": "SUCCESS",
                "evolved_prompts": [
                    {
                        "text": prompt.text,
                        "fitness_score": prompt.fitness_scores.get("overall_fitness", 0.0),
                        "generation": prompt.generation
                    }
                    for prompt in final_prompts[:10]  # Limit output size
                ]
            }
            
        except Exception as e:
            raise Exception("Evolution process failed")  # Don't expose internal details
    
    def _calculate_secure_fitness(self, prompt_text: str) -> Dict[str, float]:
        """Calculate fitness with security considerations."""
        # Basic secure fitness calculation
        text_length = min(len(prompt_text), 1000)  # Cap length for security
        word_count = len(prompt_text.split())
        
        # Security-aware scoring
        security_penalty = 0.0
        
        # Check for potential issues
        if any(char in prompt_text for char in '<>"\'\\/'):
            security_penalty += 0.1
        
        if len(prompt_text) > 500:
            security_penalty += 0.05
        
        base_fitness = min(1.0, (text_length * 0.001 + word_count * 0.01))
        secure_fitness = max(0.0, base_fitness - security_penalty)
        
        return {
            "overall_fitness": secure_fitness,
            "security_score": 1.0 - security_penalty,
            "length_score": min(1.0, text_length / 200.0),
            "word_score": min(1.0, word_count / 50.0)
        }
    
    def _secure_mutation(self, prompt: 'Prompt') -> List['Prompt']:
        """Perform secure mutation operations."""
        from meta_prompt_evolution import Prompt
        
        mutations = []
        base_text = prompt.text
        
        # Safe mutation operations only
        safe_mutations = [
            base_text.replace("help", "assist"),
            base_text + " I'm here to help.",
            "Please: " + base_text,
            base_text.replace("task", "request")
        ]
        
        for i, mutated_text in enumerate(safe_mutations[:3]):  # Limit mutations
            is_valid, error, sanitized = self.validator.validate_prompt_input(mutated_text)
            if is_valid:
                mutation = Prompt(
                    text=sanitized,
                    generation=prompt.generation + 1,
                    parent_ids=[prompt.id]
                )
                mutations.append(mutation)
        
        return mutations
    
    def _sanitize_evolution_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize evolution output for security."""
        if "evolved_prompts" in result:
            sanitized_prompts = []
            for prompt_data in result["evolved_prompts"]:
                if isinstance(prompt_data, dict) and "text" in prompt_data:
                    # Sanitize prompt text
                    sanitized_text = re.sub(r'[<>\"\'\\]', '', prompt_data["text"])
                    sanitized_text = ' '.join(sanitized_text.split())
                    
                    sanitized_prompts.append({
                        "text": sanitized_text[:500],  # Limit output length
                        "fitness_score": prompt_data.get("fitness_score", 0.0),
                        "generation": prompt_data.get("generation", 0)
                    })
            
            result["evolved_prompts"] = sanitized_prompts
        
        return result


def run_security_hardened_demo():
    """Demonstrate security-hardened system."""
    print("🔒 SECURITY HARDENED SYSTEM - Production-Ready Security")
    print("=" * 60)
    
    # Initialize with strict security config
    security_config = SecurityConfig(
        enable_input_validation=True,
        enable_output_sanitization=True,
        enable_rate_limiting=True,
        enable_audit_logging=True,
        max_input_length=1000,
        rate_limit_requests=10
    )
    
    secure_system = SecureEvolutionSystem(security_config)
    
    # Test with various inputs including potentially problematic ones
    test_prompts = [
        "You are a helpful AI assistant. Please help with {task}.",
        "I can assist you with {task}. Let's begin.",
        "As an AI, I'll help you solve {task} step by step.",
        "Let me provide guidance for {task}.",
        "I'm ready to support you with {task}."
    ]
    
    print(f"🚀 Testing secure evolution with {len(test_prompts)} prompts...")
    
    # Run secure evolution
    results = secure_system.secure_evolve(test_prompts)
    
    if results["status"] == "SUCCESS":
        print(f"✅ Secure evolution completed successfully!")
        print(f"✅ Request ID: {results['request_id']}")
        print(f"✅ Execution time: {results['execution_time']:.2f}s")
        print(f"✅ Security features enabled: {len([k for k, v in results['security_metrics']['security_features_enabled'].items() if v])}/4")
        
        print(f"\n🏆 Top Secure Evolved Prompts:")
        for i, prompt in enumerate(results["evolved_prompts"][:3]):
            print(f"   {i+1}. Fitness: {prompt['fitness_score']:.3f}")
            print(f"      Text: {prompt['text']}")
            print()
        
        return results
    else:
        print(f"❌ Secure evolution failed: {results.get('error', 'Unknown error')}")
        return results


if __name__ == "__main__":
    try:
        results = run_security_hardened_demo()
        
        # Save security-compliant results
        timestamp = int(time.time())
        output_file = f"security_hardened_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")
        print("🎯 SECURITY HARDENED SYSTEM OPERATIONAL")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)