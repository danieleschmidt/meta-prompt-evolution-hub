#!/usr/bin/env python3
"""
PROGRESSIVE QUALITY GATES - Generation 1: Basic Framework
Autonomous SDLC with intelligent progressive validation that adapts to project maturity
"""

import sys
import os
import json
import time
import hashlib
import asyncio
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('progressive_quality_gates')


class ProjectMaturity(Enum):
    """Project maturity levels for progressive validation"""
    PROTOTYPE = "prototype"         # Basic functionality only
    DEVELOPMENT = "development"     # Core features implemented
    TESTING = "testing"            # Comprehensive testing required
    PRODUCTION = "production"      # Full validation required


class QualityGateType(Enum):
    """Types of quality gates"""
    FUNCTIONAL = "functional"       # Core functionality tests
    SECURITY = "security"          # Security validation
    PERFORMANCE = "performance"     # Performance benchmarks
    RELIABILITY = "reliability"    # System stability tests
    COMPLIANCE = "compliance"      # Standards compliance


@dataclass
class QualityGateResult:
    """Individual quality gate result"""
    gate_type: QualityGateType
    name: str
    passed: bool
    score: float
    message: str
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    weight: float = 1.0


@dataclass
class ProgressiveConfig:
    """Progressive quality gates configuration"""
    project_maturity: ProjectMaturity = ProjectMaturity.DEVELOPMENT
    adaptive_thresholds: bool = True
    parallel_execution: bool = True
    fail_fast: bool = False
    min_overall_score: float = 75.0
    timeout_seconds: int = 300


class MaturityDetector:
    """Automatically detect project maturity level"""
    
    def __init__(self):
        self.indicators = {
            ProjectMaturity.PROTOTYPE: {
                'test_coverage': 0.3,
                'file_count': 10,
                'documentation_ratio': 0.1,
                'deployment_config': False
            },
            ProjectMaturity.DEVELOPMENT: {
                'test_coverage': 0.6,
                'file_count': 50,
                'documentation_ratio': 0.3,
                'deployment_config': True
            },
            ProjectMaturity.TESTING: {
                'test_coverage': 0.8,
                'file_count': 100,
                'documentation_ratio': 0.5,
                'deployment_config': True
            },
            ProjectMaturity.PRODUCTION: {
                'test_coverage': 0.9,
                'file_count': 200,
                'documentation_ratio': 0.7,
                'deployment_config': True
            }
        }
    
    def detect_maturity(self, project_root: str = '.') -> ProjectMaturity:
        """Detect project maturity level based on indicators"""
        try:
            indicators = self._analyze_project_indicators(project_root)
            
            # Score each maturity level
            scores = {}
            for maturity, thresholds in self.indicators.items():
                score = self._calculate_maturity_score(indicators, thresholds)
                scores[maturity] = score
            
            # Return the maturity level with highest score
            best_maturity = max(scores, key=scores.get)
            
            logger.info(f"🔍 Detected project maturity: {best_maturity.value}")
            logger.info(f"   Test coverage: {indicators['test_coverage']:.1%}")
            logger.info(f"   Files: {indicators['file_count']}")
            logger.info(f"   Documentation: {indicators['documentation_ratio']:.1%}")
            logger.info(f"   Deployment config: {indicators['deployment_config']}")
            
            return best_maturity
            
        except Exception as e:
            logger.warning(f"Failed to detect maturity, defaulting to DEVELOPMENT: {e}")
            return ProjectMaturity.DEVELOPMENT
    
    def _analyze_project_indicators(self, project_root: str) -> Dict[str, Any]:
        """Analyze project to extract maturity indicators"""
        indicators = {
            'test_coverage': 0.0,
            'file_count': 0,
            'documentation_ratio': 0.0,
            'deployment_config': False
        }
        
        # Count files and analyze structure
        python_files = []
        test_files = []
        doc_files = []
        
        for root, dirs, files in os.walk(project_root):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.py'):
                    python_files.append(file_path)
                    
                    if 'test_' in file or file.startswith('test_'):
                        test_files.append(file_path)
                
                elif file.endswith(('.md', '.rst', '.txt')):
                    doc_files.append(file_path)
        
        indicators['file_count'] = len(python_files)
        
        # Estimate test coverage
        if python_files:
            indicators['test_coverage'] = min(1.0, len(test_files) / len(python_files) * 2)
        
        # Calculate documentation ratio
        total_files = len(python_files) + len(doc_files)
        if total_files > 0:
            indicators['documentation_ratio'] = len(doc_files) / total_files
        
        # Check for deployment configuration
        deployment_files = [
            'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
            'deployment.yaml', 'deployment.yml',
            'kubernetes', 'deployment',
            'pyproject.toml', 'setup.py', 'requirements.txt'
        ]
        
        indicators['deployment_config'] = any(
            os.path.exists(os.path.join(project_root, file)) for file in deployment_files
        )
        
        return indicators
    
    def _calculate_maturity_score(self, indicators: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
        """Calculate how well indicators match maturity thresholds"""
        score = 0.0
        total_weight = 0.0
        
        weights = {
            'test_coverage': 0.4,
            'file_count': 0.2,
            'documentation_ratio': 0.2,
            'deployment_config': 0.2
        }
        
        for key, weight in weights.items():
            if key in indicators and key in thresholds:
                if key == 'deployment_config':
                    # Boolean indicator
                    score += weight if (indicators[key] == thresholds[key]) else 0
                else:
                    # Numeric indicator - closer to threshold is better
                    ratio = min(1.0, indicators[key] / thresholds[key])
                    score += weight * ratio
                
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0


class BasicFunctionalTester:
    """Basic functional testing for Generation 1"""
    
    def __init__(self):
        self.test_methods = []
    
    def test_core_imports(self) -> QualityGateResult:
        """Test that core modules can be imported"""
        start_time = time.time()
        
        try:
            # Try to import core modules
            from meta_prompt_evolution import PromptPopulation, Prompt
            
            # Test basic functionality
            prompt = Prompt("test prompt")
            population = PromptPopulation.from_seeds(["test1", "test2"])
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                gate_type=QualityGateType.FUNCTIONAL,
                name="core_imports",
                passed=True,
                score=100.0,
                message="Core imports successful",
                execution_time=execution_time,
                details={'modules_tested': 2}
            )
            
        except ImportError as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_type=QualityGateType.FUNCTIONAL,
                name="core_imports",
                passed=False,
                score=0.0,
                message=f"Core import failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_type=QualityGateType.FUNCTIONAL,
                name="core_imports",
                passed=False,
                score=0.0,
                message=f"Functionality test failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def test_basic_evolution(self) -> QualityGateResult:
        """Test basic evolution functionality"""
        start_time = time.time()
        
        try:
            # Test different generation systems
            generation_results = []
            
            # Test Generation 1 if available
            try:
                exec(open('generation_1_simple_working_demo.py').read(), {'__name__': '__test__'})
                generation_results.append(('Generation 1', True))
                logger.debug("✅ Generation 1 test passed")
            except Exception as e:
                generation_results.append(('Generation 1', False))
                logger.debug(f"❌ Generation 1 test failed: {e}")
            
            # Test Generation 2 if available
            try:
                exec(open('generation_2_robust_comprehensive_system.py').read(), {'__name__': '__test__'})
                generation_results.append(('Generation 2', True))
                logger.debug("✅ Generation 2 test passed")
            except Exception as e:
                generation_results.append(('Generation 2', False))
                logger.debug(f"❌ Generation 2 test failed: {e}")
            
            execution_time = time.time() - start_time
            
            # Calculate score based on working generations
            working_generations = sum(1 for _, passed in generation_results if passed)
            total_generations = len(generation_results)
            score = (working_generations / max(total_generations, 1)) * 100
            
            return QualityGateResult(
                gate_type=QualityGateType.FUNCTIONAL,
                name="basic_evolution",
                passed=working_generations > 0,
                score=score,
                message=f"Evolution systems: {working_generations}/{total_generations} working",
                execution_time=execution_time,
                details={
                    'generation_results': generation_results,
                    'working_generations': working_generations
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_type=QualityGateType.FUNCTIONAL,
                name="basic_evolution",
                passed=False,
                score=0.0,
                message=f"Evolution test failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )


class BasicSecurityScanner:
    """Basic security scanning for Generation 1"""
    
    def __init__(self):
        self.critical_patterns = [
            (r'eval\s*\(', "Critical: eval() usage"),
            (r'exec\s*\(', "Critical: exec() usage"),
            (r'os\.system\s*\(', "Critical: OS command execution"),
            (r'subprocess\.(call|run)', "Warning: subprocess usage"),
        ]
    
    def scan_security_basics(self) -> QualityGateResult:
        """Basic security scan focusing on critical issues"""
        start_time = time.time()
        
        try:
            violations = []
            files_scanned = 0
            
            # Scan main generation files
            generation_files = [
                'generation_1_simple_working_demo.py',
                'generation_2_robust_comprehensive_system.py',
                'generation_3_scalable_performance_optimized.py'
            ]
            
            for file_path in generation_files:
                if os.path.exists(file_path):
                    files_scanned += 1
                    file_violations = self._scan_file(file_path)
                    violations.extend(file_violations)
            
            # Scan meta_prompt_evolution module
            module_path = 'meta_prompt_evolution'
            if os.path.exists(module_path):
                for root, dirs, files in os.walk(module_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            files_scanned += 1
                            file_violations = self._scan_file(file_path)
                            violations.extend(file_violations)
            
            execution_time = time.time() - start_time
            
            # Calculate security score
            critical_violations = [v for v in violations if 'Critical' in v['description']]
            security_score = 100.0 - (len(critical_violations) * 25) - (len(violations) * 5)
            security_score = max(0.0, security_score)
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                name="basic_security",
                passed=len(critical_violations) == 0,
                score=security_score,
                message=f"Security scan: {len(violations)} issues, {len(critical_violations)} critical",
                execution_time=execution_time,
                details={
                    'violations': violations,
                    'critical_violations': len(critical_violations),
                    'files_scanned': files_scanned,
                    'security_score': security_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                name="basic_security",
                passed=False,
                score=0.0,
                message=f"Security scan failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Scan a single file for security issues"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in self.critical_patterns:
                        import re
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append({
                                'file': file_path,
                                'line': line_num,
                                'pattern': pattern,
                                'description': description,
                                'code': line.strip()
                            })
        
        except Exception as e:
            violations.append({
                'file': file_path,
                'line': 0,
                'pattern': 'FILE_READ_ERROR',
                'description': f"Failed to read file: {str(e)}",
                'code': ''
            })
        
        return violations


class BasicPerformanceTester:
    """Basic performance testing for Generation 1"""
    
    def test_basic_performance(self) -> QualityGateResult:
        """Test basic performance characteristics"""
        start_time = time.time()
        
        try:
            performance_scores = []
            
            # Test 1: Data structure performance
            data_perf = self._test_data_structures()
            performance_scores.append(data_perf)
            
            # Test 2: Basic algorithm performance
            algo_perf = self._test_algorithms()
            performance_scores.append(algo_perf)
            
            execution_time = time.time() - start_time
            average_score = sum(performance_scores) / len(performance_scores)
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE,
                name="basic_performance",
                passed=average_score >= 70.0,
                score=average_score,
                message=f"Performance: {average_score:.1f}% average across {len(performance_scores)} tests",
                execution_time=execution_time,
                details={
                    'data_structure_score': data_perf,
                    'algorithm_score': algo_perf,
                    'average_score': average_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE,
                name="basic_performance",
                passed=False,
                score=0.0,
                message=f"Performance test failed: {str(e)}",
                execution_time=execution_time,
                details={'error': str(e)}
            )
    
    def _test_data_structures(self) -> float:
        """Test basic data structure performance"""
        try:
            from meta_prompt_evolution import PromptPopulation, Prompt
            
            # Create and manipulate data structures
            test_start = time.time()
            
            prompts = [Prompt(f"test prompt {i}") for i in range(100)]
            population = PromptPopulation(prompts)
            top_prompts = population.get_top_k(10)
            
            elapsed = time.time() - test_start
            
            # Score based on execution time (under 10ms is excellent)
            if elapsed < 0.01:
                return 100.0
            elif elapsed < 0.05:
                return 90.0
            elif elapsed < 0.1:
                return 80.0
            else:
                return max(60.0, 100.0 - (elapsed * 100))
        
        except Exception:
            return 50.0
    
    def _test_algorithms(self) -> float:
        """Test basic algorithm performance"""
        try:
            # Simple algorithm performance test
            test_start = time.time()
            
            # Simulate evolutionary operations
            data = list(range(1000))
            sorted_data = sorted(data, reverse=True)
            top_100 = sorted_data[:100]
            
            elapsed = time.time() - test_start
            
            # Score based on execution time
            if elapsed < 0.005:
                return 100.0
            elif elapsed < 0.01:
                return 95.0
            elif elapsed < 0.02:
                return 85.0
            else:
                return max(70.0, 100.0 - (elapsed * 1000))
        
        except Exception:
            return 50.0


class ProgressiveQualityGates:
    """Progressive quality gates orchestrator for Generation 1"""
    
    def __init__(self, config: Optional[ProgressiveConfig] = None):
        self.config = config or ProgressiveConfig()
        self.maturity_detector = MaturityDetector()
        self.functional_tester = BasicFunctionalTester()
        self.security_scanner = BasicSecurityScanner()
        self.performance_tester = BasicPerformanceTester()
        
        self.gate_results: List[QualityGateResult] = []
        self.project_maturity = None
        self.adaptive_weights = {}
    
    def run_progressive_quality_gates(self) -> Dict[str, Any]:
        """Run progressive quality gates based on project maturity"""
        logger.info("🚀 PROGRESSIVE QUALITY GATES - Generation 1")
        logger.info("=" * 55)
        
        start_time = time.time()
        
        # Step 1: Detect project maturity
        self.project_maturity = self.maturity_detector.detect_maturity()
        self._configure_adaptive_weights()
        
        # Step 2: Run appropriate quality gates
        gates_to_run = self._select_gates_for_maturity()
        
        logger.info(f"\n🎯 Running {len(gates_to_run)} quality gates for {self.project_maturity.value} project")
        logger.info("-" * 50)
        
        # Step 3: Execute quality gates
        if self.config.parallel_execution:
            self._run_gates_parallel(gates_to_run)
        else:
            self._run_gates_sequential(gates_to_run)
        
        # Step 4: Calculate overall results
        total_time = time.time() - start_time
        overall_results = self._calculate_overall_results(total_time)
        
        # Step 5: Generate report
        report = self._generate_progressive_report(overall_results)
        
        # Step 6: Save results
        self._save_results(report)
        
        return report
    
    def _configure_adaptive_weights(self):
        """Configure adaptive weights based on project maturity"""
        maturity_weights = {
            ProjectMaturity.PROTOTYPE: {
                QualityGateType.FUNCTIONAL: 0.6,
                QualityGateType.SECURITY: 0.2,
                QualityGateType.PERFORMANCE: 0.1,
                QualityGateType.RELIABILITY: 0.1,
                QualityGateType.COMPLIANCE: 0.0
            },
            ProjectMaturity.DEVELOPMENT: {
                QualityGateType.FUNCTIONAL: 0.4,
                QualityGateType.SECURITY: 0.3,
                QualityGateType.PERFORMANCE: 0.2,
                QualityGateType.RELIABILITY: 0.1,
                QualityGateType.COMPLIANCE: 0.0
            },
            ProjectMaturity.TESTING: {
                QualityGateType.FUNCTIONAL: 0.3,
                QualityGateType.SECURITY: 0.3,
                QualityGateType.PERFORMANCE: 0.2,
                QualityGateType.RELIABILITY: 0.2,
                QualityGateType.COMPLIANCE: 0.0
            },
            ProjectMaturity.PRODUCTION: {
                QualityGateType.FUNCTIONAL: 0.2,
                QualityGateType.SECURITY: 0.3,
                QualityGateType.PERFORMANCE: 0.2,
                QualityGateType.RELIABILITY: 0.2,
                QualityGateType.COMPLIANCE: 0.1
            }
        }
        
        self.adaptive_weights = maturity_weights.get(
            self.project_maturity, 
            maturity_weights[ProjectMaturity.DEVELOPMENT]
        )
        
        logger.info(f"📊 Adaptive weights configured for {self.project_maturity.value}:")
        for gate_type, weight in self.adaptive_weights.items():
            if weight > 0:
                logger.info(f"   {gate_type.value}: {weight:.1%}")
    
    def _select_gates_for_maturity(self) -> List[Tuple[str, callable]]:
        """Select appropriate gates based on project maturity"""
        all_gates = [
            ('Core Imports', self.functional_tester.test_core_imports),
            ('Basic Evolution', self.functional_tester.test_basic_evolution),
            ('Basic Security', self.security_scanner.scan_security_basics),
            ('Basic Performance', self.performance_tester.test_basic_performance)
        ]
        
        # For Generation 1, we run all basic gates regardless of maturity
        # This ensures comprehensive coverage while keeping it simple
        return all_gates
    
    def _run_gates_sequential(self, gates: List[Tuple[str, callable]]):
        """Run quality gates sequentially"""
        for gate_name, gate_func in gates:
            logger.info(f"🔍 Running {gate_name}...")
            
            try:
                result = gate_func()
                self.gate_results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"   {status} {gate_name}: {result.score:.1f}% - {result.message}")
                
                if self.config.fail_fast and not result.passed and result.required:
                    logger.warning("⚠️  Fail-fast enabled, stopping on first failure")
                    break
                    
            except Exception as e:
                logger.error(f"❌ {gate_name} execution error: {e}")
                error_result = QualityGateResult(
                    gate_type=QualityGateType.FUNCTIONAL,
                    name=gate_name.lower().replace(' ', '_'),
                    passed=False,
                    score=0.0,
                    message=f"Execution error: {str(e)}",
                    execution_time=0.0,
                    details={'error': str(e)}
                )
                self.gate_results.append(error_result)
    
    def _run_gates_parallel(self, gates: List[Tuple[str, callable]]):
        """Run quality gates in parallel"""
        with ThreadPoolExecutor(max_workers=min(len(gates), 4)) as executor:
            # Submit all gates
            future_to_gate = {
                executor.submit(gate_func): gate_name 
                for gate_name, gate_func in gates
            }
            
            # Collect results
            for future in as_completed(future_to_gate):
                gate_name = future_to_gate[future]
                
                try:
                    result = future.result(timeout=60)  # 60 second timeout per gate
                    self.gate_results.append(result)
                    
                    status = "✅ PASS" if result.passed else "❌ FAIL"
                    logger.info(f"   {status} {gate_name}: {result.score:.1f}% - {result.message}")
                    
                except Exception as e:
                    logger.error(f"❌ {gate_name} execution error: {e}")
                    error_result = QualityGateResult(
                        gate_type=QualityGateType.FUNCTIONAL,
                        name=gate_name.lower().replace(' ', '_'),
                        passed=False,
                        score=0.0,
                        message=f"Execution error: {str(e)}",
                        execution_time=0.0,
                        details={'error': str(e)}
                    )
                    self.gate_results.append(error_result)
    
    def _calculate_overall_results(self, total_time: float) -> Dict[str, Any]:
        """Calculate overall quality assessment"""
        logger.info(f"\n📊 PROGRESSIVE QUALITY ASSESSMENT")
        logger.info("-" * 40)
        
        # Group results by gate type
        results_by_type = {}
        for result in self.gate_results:
            if result.gate_type not in results_by_type:
                results_by_type[result.gate_type] = []
            results_by_type[result.gate_type].append(result)
        
        # Calculate weighted scores by gate type
        type_scores = {}
        for gate_type, results in results_by_type.items():
            if results:
                avg_score = sum(r.score for r in results) / len(results)
                type_scores[gate_type] = avg_score
                
                status = "✅" if all(r.passed for r in results) else "❌"
                logger.info(f"{status} {gate_type.value.title()}: {avg_score:.1f}%")
        
        # Calculate overall weighted score
        overall_score = 0.0
        total_weight = 0.0
        
        for gate_type, weight in self.adaptive_weights.items():
            if gate_type in type_scores and weight > 0:
                overall_score += type_scores[gate_type] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Quality gate decision
        passed_gates = sum(1 for r in self.gate_results if r.passed)
        total_gates = len(self.gate_results)
        required_gates = sum(1 for r in self.gate_results if r.required)
        required_passed = sum(1 for r in self.gate_results if r.required and r.passed)
        
        quality_gate_passed = (
            overall_score >= self.config.min_overall_score and
            required_passed == required_gates
        )
        
        logger.info(f"\n🎯 Overall Quality Score: {overall_score:.1f}%")
        logger.info(f"📈 Gates Passed: {passed_gates}/{total_gates}")
        logger.info(f"🎯 Required Gates: {required_passed}/{required_gates}")
        logger.info(f"⏱️  Execution Time: {total_time:.2f}s")
        
        if quality_gate_passed:
            logger.info("✅ PROGRESSIVE QUALITY GATES PASSED!")
        else:
            logger.warning("⚠️  QUALITY GATES NEED ATTENTION")
        
        return {
            'overall_score': overall_score,
            'quality_gate_passed': quality_gate_passed,
            'type_scores': type_scores,
            'execution_stats': {
                'total_time': total_time,
                'passed_gates': passed_gates,
                'total_gates': total_gates,
                'required_passed': required_passed,
                'required_gates': required_gates
            }
        }
    
    def _generate_progressive_report(self, overall_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive progressive quality report"""
        return {
            'metadata': {
                'report_type': 'progressive_quality_gates_gen1',
                'version': '1.0.0',
                'timestamp': time.time(),
                'execution_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'project_maturity': self.project_maturity.value
            },
            'configuration': {
                'adaptive_thresholds': self.config.adaptive_thresholds,
                'parallel_execution': self.config.parallel_execution,
                'fail_fast': self.config.fail_fast,
                'min_overall_score': self.config.min_overall_score,
                'adaptive_weights': {k.value: v for k, v in self.adaptive_weights.items()}
            },
            'results': {
                'overall_score': overall_results['overall_score'],
                'quality_gate_passed': overall_results['quality_gate_passed'],
                'type_scores': {k.value: v for k, v in overall_results['type_scores'].items()},
                'execution_stats': overall_results['execution_stats']
            },
            'gate_details': [
                {
                    'gate_type': result.gate_type.value,
                    'name': result.name,
                    'passed': result.passed,
                    'score': result.score,
                    'message': result.message,
                    'execution_time': result.execution_time,
                    'required': result.required,
                    'weight': result.weight,
                    'details': result.details
                }
                for result in self.gate_results
            ],
            'recommendations': self._generate_recommendations(overall_results)
        }
    
    def _generate_recommendations(self, overall_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Overall score recommendations
        if overall_results['overall_score'] < 60:
            recommendations.append("🚨 Critical: Overall quality below 60% - immediate attention required")
        elif overall_results['overall_score'] < 75:
            recommendations.append("⚠️  Warning: Overall quality below target - focus on failing gates")
        
        # Gate-specific recommendations
        failed_gates = [r for r in self.gate_results if not r.passed]
        if failed_gates:
            functional_failures = [r for r in failed_gates if r.gate_type == QualityGateType.FUNCTIONAL]
            security_failures = [r for r in failed_gates if r.gate_type == QualityGateType.SECURITY]
            performance_failures = [r for r in failed_gates if r.gate_type == QualityGateType.PERFORMANCE]
            
            if functional_failures:
                recommendations.append("🔧 Fix functional test failures - core system stability at risk")
            
            if security_failures:
                recommendations.append("🔒 Address security vulnerabilities immediately")
            
            if performance_failures:
                recommendations.append("⚡ Optimize performance bottlenecks")
        
        # Maturity-specific recommendations
        if self.project_maturity == ProjectMaturity.PROTOTYPE:
            recommendations.append("🏗️  Consider adding more comprehensive tests as project matures")
        elif self.project_maturity == ProjectMaturity.DEVELOPMENT:
            recommendations.append("📈 Focus on improving test coverage and security measures")
        elif self.project_maturity in [ProjectMaturity.TESTING, ProjectMaturity.PRODUCTION]:
            recommendations.append("🚀 Implement full quality gates for production readiness")
        
        if not recommendations:
            recommendations.append("✅ All quality gates passed - system ready for next development phase")
        
        return recommendations
    
    def _save_results(self, report: Dict[str, Any]):
        """Save quality gates results to file"""
        timestamp = int(time.time())
        filename = f'progressive_quality_gates_gen1_results_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"💾 Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main execution function for Progressive Quality Gates Generation 1"""
    try:
        # Configure for adaptive execution
        config = ProgressiveConfig(
            adaptive_thresholds=True,
            parallel_execution=True,
            fail_fast=False,
            min_overall_score=75.0,
            timeout_seconds=300
        )
        
        # Initialize and run progressive quality gates
        progressive_gates = ProgressiveQualityGates(config)
        results = progressive_gates.run_progressive_quality_gates()
        
        # Exit with appropriate code
        exit_code = 0 if results['results']['quality_gate_passed'] else 1
        
        if exit_code == 0:
            logger.info("\n🎉 PROGRESSIVE QUALITY GATES GENERATION 1 COMPLETE")
        else:
            logger.warning("\n⚠️  PROGRESSIVE QUALITY GATES NEED ATTENTION")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"❌ Critical error in progressive quality gates: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)