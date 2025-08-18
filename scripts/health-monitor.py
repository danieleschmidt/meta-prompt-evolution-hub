#!/usr/bin/env python3
"""
Repository health monitoring script for Meta-Prompt-Evolution-Hub.
Monitors various health indicators and sends alerts when issues are detected.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class HealthIndicator:
    """Represents a single health indicator."""
    
    def __init__(self, name: str, status: str, message: str, value: Any = None, threshold: Any = None):
        self.name = name
        self.status = status  # "healthy", "warning", "critical"
        self.message = message
        self.value = value
        self.threshold = threshold
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat()
        }


class RepositoryHealthMonitor:
    """Monitors repository health across multiple dimensions."""
    
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or (self.project_root / "scripts" / "health-config.json")
        self.indicators: List[HealthIndicator] = []
        self.load_config()
        
    def load_config(self):
        """Load monitoring configuration."""
        default_config = {
            "thresholds": {
                "test_coverage": 80.0,
                "build_time_seconds": 300,
                "critical_vulnerabilities": 0,
                "high_vulnerabilities": 2,
                "outdated_dependencies": 10,
                "disk_usage_percent": 85.0,
                "response_time_ms": 2000
            },
            "notifications": {
                "enabled": False,
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "slack": {
                    "webhook_url": ""
                }
            },
            "checks": {
                "code_quality": True,
                "security": True,
                "dependencies": True,
                "git_health": True,
                "disk_space": True,
                "api_health": False
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self.config = {**default_config, **loaded_config}
            except Exception as e:
                print(f"⚠️  Error loading config, using defaults: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
            
    def save_config(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def run_command(self, command: List[str]) -> tuple[str, int]:
        """Run a shell command and return output and exit code."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=60
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", 1
        except Exception:
            return "", 1
            
    def check_test_coverage(self):
        """Check test coverage and compare against threshold."""
        if not self.config["checks"]["code_quality"]:
            return
            
        print("🧪 Checking test coverage...")
        
        # Run coverage check
        output, code = self.run_command([
            "python", "-m", "pytest",
            "--cov=meta_prompt_evolution",
            "--cov-report=json:coverage.json",
            "--quiet",
            "tests/"
        ])
        
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    coverage_percent = coverage_data["totals"]["percent_covered"]
                    threshold = self.config["thresholds"]["test_coverage"]
                    
                    if coverage_percent >= threshold:
                        status = "healthy"
                        message = f"Test coverage is {coverage_percent:.1f}%"
                    else:
                        status = "warning"
                        message = f"Test coverage is {coverage_percent:.1f}% (below {threshold}%)"
                        
                    self.indicators.append(HealthIndicator(
                        "test_coverage",
                        status,
                        message,
                        coverage_percent,
                        threshold
                    ))
            except Exception as e:
                self.indicators.append(HealthIndicator(
                    "test_coverage",
                    "critical",
                    f"Failed to read coverage data: {e}"
                ))
        else:
            self.indicators.append(HealthIndicator(
                "test_coverage",
                "critical",
                "Coverage data not found"
            ))
            
    def check_security_vulnerabilities(self):
        """Check for security vulnerabilities."""
        if not self.config["checks"]["security"]:
            return
            
        print("🛡️  Checking security vulnerabilities...")
        
        # Run safety check
        output, code = self.run_command(["safety", "check", "--json"])
        
        try:
            if output:
                safety_data = json.loads(output)
                vulnerabilities = safety_data.get("vulnerabilities", [])
                
                # Count by severity (simplified - safety doesn't provide severity)
                vuln_count = len(vulnerabilities)
                critical_threshold = self.config["thresholds"]["critical_vulnerabilities"]
                high_threshold = self.config["thresholds"]["high_vulnerabilities"]
                
                if vuln_count == 0:
                    status = "healthy"
                    message = "No known vulnerabilities found"
                elif vuln_count <= high_threshold:
                    status = "warning"
                    message = f"{vuln_count} vulnerabilities found"
                else:
                    status = "critical"
                    message = f"{vuln_count} vulnerabilities found (above threshold)"
                    
                self.indicators.append(HealthIndicator(
                    "security_vulnerabilities",
                    status,
                    message,
                    vuln_count,
                    high_threshold
                ))
            else:
                self.indicators.append(HealthIndicator(
                    "security_vulnerabilities",
                    "healthy",
                    "No vulnerabilities detected"
                ))
        except Exception as e:
            self.indicators.append(HealthIndicator(
                "security_vulnerabilities",
                "warning",
                f"Security check failed: {e}"
            ))
            
    def check_dependency_health(self):
        """Check dependency health and outdated packages."""
        if not self.config["checks"]["dependencies"]:
            return
            
        print("📦 Checking dependency health...")
        
        # Check for outdated packages
        output, code = self.run_command(["pip", "list", "--outdated", "--format=json"])
        
        try:
            if output:
                outdated_packages = json.loads(output)
                outdated_count = len(outdated_packages)
                threshold = self.config["thresholds"]["outdated_dependencies"]
                
                if outdated_count == 0:
                    status = "healthy"
                    message = "All dependencies are up to date"
                elif outdated_count <= threshold:
                    status = "warning"
                    message = f"{outdated_count} outdated dependencies"
                else:
                    status = "critical"
                    message = f"{outdated_count} outdated dependencies (above threshold)"
                    
                self.indicators.append(HealthIndicator(
                    "dependency_health",
                    status,
                    message,
                    outdated_count,
                    threshold
                ))
            else:
                self.indicators.append(HealthIndicator(
                    "dependency_health",
                    "healthy",
                    "All dependencies up to date"
                ))
        except Exception as e:
            self.indicators.append(HealthIndicator(
                "dependency_health",
                "warning",
                f"Dependency check failed: {e}"
            ))
            
    def check_git_health(self):
        """Check Git repository health."""
        if not self.config["checks"]["git_health"]:
            return
            
        print("📈 Checking Git repository health...")
        
        indicators = []
        
        # Check for uncommitted changes
        output, code = self.run_command(["git", "status", "--porcelain"])
        if output:
            indicators.append(("warning", f"{len(output.splitlines())} uncommitted changes"))
        else:
            indicators.append(("healthy", "Working directory clean"))
            
        # Check for untracked files
        output, code = self.run_command(["git", "ls-files", "--other", "--exclude-standard"])
        untracked_files = output.splitlines() if output else []
        if len(untracked_files) > 5:  # Threshold for too many untracked files
            indicators.append(("warning", f"{len(untracked_files)} untracked files"))
        else:
            indicators.append(("healthy", f"{len(untracked_files)} untracked files"))
            
        # Check for recent commits (activity indicator)
        output, code = self.run_command(["git", "log", "--since='7 days ago'", "--oneline"])
        recent_commits = len(output.splitlines()) if output else 0
        if recent_commits == 0:
            indicators.append(("warning", "No commits in the last 7 days"))
        else:
            indicators.append(("healthy", f"{recent_commits} commits in the last 7 days"))
            
        # Check for large files in repo
        output, code = self.run_command(["find", ".", "-size", "+50M", "-not", "-path", "./.git/*"])
        large_files = output.splitlines() if output else []
        if large_files:
            indicators.append(("warning", f"{len(large_files)} large files (>50MB) found"))
        else:
            indicators.append(("healthy", "No large files detected"))
            
        # Aggregate git health
        warning_count = sum(1 for status, _ in indicators if status == "warning")
        if warning_count == 0:
            overall_status = "healthy"
            overall_message = "Git repository is healthy"
        elif warning_count <= 2:
            overall_status = "warning"
            overall_message = f"{warning_count} Git health issues detected"
        else:
            overall_status = "critical"
            overall_message = f"{warning_count} Git health issues detected"
            
        self.indicators.append(HealthIndicator(
            "git_health",
            overall_status,
            overall_message,
            warning_count,
            2
        ))
        
    def check_disk_space(self):
        """Check available disk space."""
        if not self.config["checks"]["disk_space"]:
            return
            
        print("💾 Checking disk space...")
        
        try:
            # Use df command to check disk usage
            output, code = self.run_command(["df", "-h", "."])
            
            if code == 0 and output:
                lines = output.strip().split('\n')
                if len(lines) >= 2:
                    # Parse the df output
                    fields = lines[1].split()
                    if len(fields) >= 5:
                        usage_percent = int(fields[4].rstrip('%'))
                        available = fields[3]
                        threshold = self.config["thresholds"]["disk_usage_percent"]
                        
                        if usage_percent < threshold:
                            status = "healthy"
                            message = f"Disk usage: {usage_percent}% ({available} available)"
                        elif usage_percent < threshold + 10:
                            status = "warning"
                            message = f"Disk usage: {usage_percent}% ({available} available)"
                        else:
                            status = "critical"
                            message = f"Disk usage: {usage_percent}% ({available} available)"
                            
                        self.indicators.append(HealthIndicator(
                            "disk_space",
                            status,
                            message,
                            usage_percent,
                            threshold
                        ))
                        return
                        
            # Fallback for systems without df or parsing issues
            self.indicators.append(HealthIndicator(
                "disk_space",
                "warning",
                "Could not determine disk usage"
            ))
            
        except Exception as e:
            self.indicators.append(HealthIndicator(
                "disk_space",
                "warning",
                f"Disk space check failed: {e}"
            ))
            
    def check_api_health(self):
        """Check API endpoint health (if applicable)."""
        if not self.config["checks"]["api_health"]:
            return
            
        print("🌐 Checking API health...")
        
        # This would typically check application health endpoints
        # For now, it's a placeholder that could be configured
        
        api_endpoints = [
            {"url": "http://localhost:8080/health", "name": "Application Health"},
            {"url": "http://localhost:8080/metrics", "name": "Metrics Endpoint"}
        ]
        
        healthy_endpoints = 0
        total_endpoints = len(api_endpoints)
        
        for endpoint in api_endpoints:
            try:
                response = requests.get(endpoint["url"], timeout=10)
                if response.status_code == 200:
                    healthy_endpoints += 1
            except:
                pass  # Endpoint not available, which is fine for development
                
        if total_endpoints == 0:
            # No endpoints configured
            self.indicators.append(HealthIndicator(
                "api_health",
                "healthy",
                "No API endpoints configured for monitoring"
            ))
        elif healthy_endpoints == total_endpoints:
            self.indicators.append(HealthIndicator(
                "api_health",
                "healthy",
                f"All {total_endpoints} API endpoints responding"
            ))
        elif healthy_endpoints > 0:
            self.indicators.append(HealthIndicator(
                "api_health",
                "warning",
                f"{healthy_endpoints}/{total_endpoints} API endpoints responding"
            ))
        else:
            self.indicators.append(HealthIndicator(
                "api_health",
                "critical",
                "No API endpoints responding"
            ))
            
    def run_all_checks(self):
        """Run all health checks."""
        print("🏥 Starting repository health monitoring...")
        
        self.indicators = []  # Reset indicators
        
        try:
            self.check_test_coverage()
        except Exception as e:
            print(f"Error in test coverage check: {e}")
            
        try:
            self.check_security_vulnerabilities()
        except Exception as e:
            print(f"Error in security check: {e}")
            
        try:
            self.check_dependency_health()
        except Exception as e:
            print(f"Error in dependency check: {e}")
            
        try:
            self.check_git_health()
        except Exception as e:
            print(f"Error in git health check: {e}")
            
        try:
            self.check_disk_space()
        except Exception as e:
            print(f"Error in disk space check: {e}")
            
        try:
            self.check_api_health()
        except Exception as e:
            print(f"Error in API health check: {e}")
            
    def get_overall_health(self) -> tuple[str, str]:
        """Get overall health status and summary."""
        if not self.indicators:
            return "unknown", "No health indicators available"
            
        critical_count = sum(1 for i in self.indicators if i.status == "critical")
        warning_count = sum(1 for i in self.indicators if i.status == "warning")
        healthy_count = sum(1 for i in self.indicators if i.status == "healthy")
        
        if critical_count > 0:
            return "critical", f"{critical_count} critical issues, {warning_count} warnings"
        elif warning_count > 0:
            return "warning", f"{warning_count} warnings, {healthy_count} healthy checks"
        else:
            return "healthy", f"All {healthy_count} checks passing"
            
    def generate_report(self) -> str:
        """Generate a human-readable health report."""
        overall_status, overall_message = self.get_overall_health()
        
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓"
        }
        
        report = []
        report.append("# 🏥 Repository Health Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"## Overall Status: {status_emoji[overall_status]} {overall_status.upper()}")
        report.append(f"{overall_message}")
        report.append("")
        
        if self.indicators:
            report.append("## Health Indicators")
            report.append("")
            
            # Group by status
            for status in ["critical", "warning", "healthy"]:
                indicators_for_status = [i for i in self.indicators if i.status == status]
                
                if indicators_for_status:
                    report.append(f"### {status_emoji[status]} {status.title()} ({len(indicators_for_status)})")
                    for indicator in indicators_for_status:
                        report.append(f"- **{indicator.name}**: {indicator.message}")
                        if indicator.value is not None and indicator.threshold is not None:
                            report.append(f"  - Value: {indicator.value}, Threshold: {indicator.threshold}")
                    report.append("")
                    
        report.append("## Recommendations")
        critical_indicators = [i for i in self.indicators if i.status == "critical"]
        warning_indicators = [i for i in self.indicators if i.status == "warning"]
        
        if critical_indicators:
            report.append("### 🚨 Critical Actions Required")
            for indicator in critical_indicators:
                if "vulnerability" in indicator.name:
                    report.append("- Update vulnerable dependencies immediately")
                elif "coverage" in indicator.name:
                    report.append("- Add tests to improve coverage")
                elif "disk" in indicator.name:
                    report.append("- Free up disk space or expand storage")
                else:
                    report.append(f"- Address {indicator.name} issue")
                    
        if warning_indicators:
            report.append("### ⚠️ Recommended Actions")
            for indicator in warning_indicators:
                if "dependency" in indicator.name:
                    report.append("- Update outdated dependencies")
                elif "git" in indicator.name:
                    report.append("- Review Git repository hygiene")
                elif "api" in indicator.name:
                    report.append("- Check API endpoint availability")
                else:
                    report.append(f"- Review {indicator.name} status")
                    
        if not critical_indicators and not warning_indicators:
            report.append("✨ Repository is in excellent health!")
            
        return "\n".join(report)
        
    def save_health_data(self):
        """Save health data to JSON file for trending."""
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_health()[0],
            "indicators": [i.to_dict() for i in self.indicators]
        }
        
        # Append to health history file
        history_file = self.project_root / "health-history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(health_data) + '\n')
            
        # Save latest health status
        status_file = self.project_root / "health-status.json"
        with open(status_file, 'w') as f:
            json.dump(health_data, f, indent=2)
            
    def send_notifications(self, report: str):
        """Send notifications if issues are detected."""
        if not self.config["notifications"]["enabled"]:
            return
            
        overall_status, _ = self.get_overall_health()
        
        # Only send notifications for warnings and critical issues
        if overall_status == "healthy":
            return
            
        print("📧 Sending health notifications...")
        
        # Email notification
        email_config = self.config["notifications"]["email"]
        if email_config["recipients"] and email_config["username"]:
            self._send_email_notification(report, overall_status, email_config)
            
        # Slack notification
        slack_config = self.config["notifications"]["slack"]
        if slack_config["webhook_url"]:
            self._send_slack_notification(report, overall_status, slack_config)
            
    def _send_email_notification(self, report: str, status: str, config: Dict):
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"Repository Health Alert - {status.upper()}"
            
            msg.attach(MIMEText(report, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.sendmail(config['username'], config['recipients'], msg.as_string())
            server.quit()
            
            print("✅ Email notification sent")
        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")
            
    def _send_slack_notification(self, report: str, status: str, config: Dict):
        """Send Slack notification."""
        try:
            color = {"critical": "#ff0000", "warning": "#ffaa00", "healthy": "#00ff00"}
            
            payload = {
                "text": f"Repository Health Alert - {status.upper()}",
                "attachments": [{
                    "color": color.get(status, "#cccccc"),
                    "text": report,
                    "mrkdwn_in": ["text"]
                }]
            }
            
            response = requests.post(config['webhook_url'], json=payload, timeout=10)
            response.raise_for_status()
            
            print("✅ Slack notification sent")
        except Exception as e:
            print(f"❌ Failed to send Slack notification: {e}")


def main():
    """Main function for health monitoring."""
    if "--help" in sys.argv:
        print("Usage: python health-monitor.py [options]")
        print("Monitor repository health across multiple dimensions.")
        print("")
        print("Options:")
        print("  --report-only    Generate report without saving data")
        print("  --no-notify      Skip sending notifications")
        print("  --config FILE    Use custom configuration file")
        return
        
    report_only = "--report-only" in sys.argv
    no_notify = "--no-notify" in sys.argv
    
    config_path = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
            
    monitor = RepositoryHealthMonitor(config_path)
    
    # Run all health checks
    monitor.run_all_checks()
    
    # Generate and display report
    report = monitor.generate_report()
    print("\n" + report)
    
    if not report_only:
        # Save health data for trending
        monitor.save_health_data()
        
        # Save report to file
        report_file = Path("health-report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to {report_file}")
        
    # Send notifications
    if not no_notify:
        monitor.send_notifications(report)
        
    # Exit with appropriate code
    overall_status, _ = monitor.get_overall_health()
    if overall_status == "critical":
        sys.exit(2)
    elif overall_status == "warning":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()