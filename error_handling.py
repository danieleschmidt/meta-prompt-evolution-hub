#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Error Handling and Reliability
Enhanced error handling, validation, logging, and monitoring.
"""

import logging
import traceback
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ErrorMetrics:
    """Track error metrics across the system."""
    total_errors: int = 0
    evaluation_errors: int = 0
    algorithm_errors: int = 0
    timeout_errors: int = 0
    last_error_time: Optional[float] = None
    error_rate_per_hour: float = 0.0

class RobustErrorHandler:
    """Comprehensive error handling with recovery strategies."""
    
    def __init__(self):
        self.error_metrics = ErrorMetrics()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/root/repo/evolution.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def error_context(self, operation_name: str):
        """Context manager for tracking and handling errors."""
        start_time = time.time()
        try:
            self.logger.info(f"Starting operation: {operation_name}")
            yield
            self.logger.info(f"Completed operation: {operation_name} in {time.time() - start_time:.2f}s")
        except Exception as e:
            self._record_error(operation_name, e)
            raise
            
    def _record_error(self, operation: str, error: Exception):
        """Record error metrics and details."""
        self.error_metrics.total_errors += 1
        self.error_metrics.last_error_time = time.time()
        
        if "evaluation" in operation.lower():
            self.error_metrics.evaluation_errors += 1
        elif "algorithm" in operation.lower():
            self.error_metrics.algorithm_errors += 1
        elif "timeout" in str(error).lower():
            self.error_metrics.timeout_errors += 1
            
        self.logger.error(f"Error in {operation}: {str(error)}")
        self.logger.debug(f"Full traceback: {traceback.format_exc()}")
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        return {
            "total_errors": self.error_metrics.total_errors,
            "error_breakdown": {
                "evaluation": self.error_metrics.evaluation_errors,
                "algorithm": self.error_metrics.algorithm_errors,
                "timeout": self.error_metrics.timeout_errors
            },
            "last_error_time": self.error_metrics.last_error_time,
            "system_health": "healthy" if self.error_metrics.total_errors < 5 else "degraded"
        }

# Global error handler instance
error_handler = RobustErrorHandler()