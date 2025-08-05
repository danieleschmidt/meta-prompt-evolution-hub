#!/usr/bin/env python3
"""
Generation 2: Input Validation and Data Safety
Comprehensive validation, sanitization, and safety checks.
"""

import re
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass
from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation
from meta_prompt_evolution.evaluation.base import TestCase

@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Optional[Any] = None

class PromptValidator:
    """Comprehensive prompt validation and sanitization."""
    
    def __init__(self):
        self.max_prompt_length = 5000
        self.min_prompt_length = 5
        self.unsafe_patterns = [
            r'<script.*?>.*?</script>',  # Script injection
            r'javascript:',             # JavaScript protocol
            r'data:text/html',          # Data URI HTML
            r'vbscript:',              # VBScript protocol
            r'on\w+\s*=',              # Event handlers
        ]
        self.profanity_words = [
            # Basic profanity filter (simplified)
            'harmful', 'dangerous', 'illegal', 'malicious'
        ]
        
    def validate_prompt(self, prompt: Union[str, Prompt]) -> ValidationResult:
        """Validate a single prompt comprehensively."""
        errors = []
        warnings = []
        
        # Extract text content
        if isinstance(prompt, Prompt):
            text = prompt.text
            prompt_id = prompt.id
        else:
            text = prompt
            prompt_id = "unknown"
            
        # Length validation
        if len(text) < self.min_prompt_length:
            errors.append(f"Prompt too short: {len(text)} < {self.min_prompt_length}")
        elif len(text) > self.max_prompt_length:
            errors.append(f"Prompt too long: {len(text)} > {self.max_prompt_length}")
            
        # Content safety validation
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Unsafe pattern detected: {pattern}")
                
        # Profanity check
        text_lower = text.lower()
        for word in self.profanity_words:
            if word in text_lower:
                warnings.append(f"Potentially inappropriate content: {word}")
                
        # Character encoding validation
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Invalid character encoding")
            
        # Structure validation
        if not text.strip():
            errors.append("Empty or whitespace-only prompt")
            
        # Sanitize if valid
        sanitized = None
        if not errors:
            sanitized = self._sanitize_prompt(text)
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized
        )
        
    def _sanitize_prompt(self, text: str) -> str:
        """Sanitize prompt text."""
        # Remove potentially dangerous patterns
        sanitized = text
        for pattern in self.unsafe_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Ensure reasonable length
        if len(sanitized) > self.max_prompt_length:
            sanitized = sanitized[:self.max_prompt_length-3] + "..."
            
        return sanitized
        
    def validate_population(self, population: PromptPopulation) -> Dict[str, ValidationResult]:
        """Validate entire population."""
        results = {}
        for prompt in population.prompts:
            results[prompt.id] = self.validate_prompt(prompt)
        return results

class TestCaseValidator:
    """Validation for test cases and evaluation data."""
    
    def __init__(self):
        self.max_input_length = 10000
        self.max_output_length = 10000
        
    def validate_test_case(self, test_case: TestCase) -> ValidationResult:
        """Validate a test case."""
        errors = []
        warnings = []
        
        # Input validation
        if not test_case.input_data:
            errors.append("Missing input data")
        elif len(str(test_case.input_data)) > self.max_input_length:
            errors.append(f"Input too long: {len(str(test_case.input_data))}")
            
        # Output validation  
        if not test_case.expected_output:
            warnings.append("Missing expected output")
        elif len(str(test_case.expected_output)) > self.max_output_length:
            errors.append(f"Expected output too long: {len(str(test_case.expected_output))}")
            
        # Weight validation
        if test_case.weight <= 0:
            errors.append("Test case weight must be positive")
        elif test_case.weight > 100:
            warnings.append("Very high test case weight")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_test_suite(self, test_cases: List[TestCase]) -> Dict[int, ValidationResult]:
        """Validate entire test suite."""
        results = {}
        for i, test_case in enumerate(test_cases):
            results[i] = self.validate_test_case(test_case)
        return results

# Global validators
prompt_validator = PromptValidator()
test_validator = TestCaseValidator()