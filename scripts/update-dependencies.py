#!/usr/bin/env python3
"""
Automated dependency update script for Meta-Prompt-Evolution-Hub.
Updates dependencies while ensuring compatibility and security.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import requests
from packaging import version


class DependencyUpdater:
    """Manages automated dependency updates with safety checks."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.requirements_files = [
            "requirements.prod.txt",
            "requirements.txt"
        ]
        self.updated_packages = []
        self.failed_updates = []
        
    def run_command(self, command: List[str], capture_output: bool = True) -> Tuple[str, int]:
        """Run a command and return output and exit code."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root,
                timeout=300
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", 1
        except Exception as e:
            print(f"Error running command {' '.join(command)}: {e}")
            return "", 1
            
    def get_outdated_packages(self) -> List[Dict[str, str]]:
        """Get list of outdated packages with current and latest versions."""
        output, code = self.run_command(["pip", "list", "--outdated", "--format=json"])
        
        if code != 0:
            print("❌ Failed to get outdated packages list")
            return []
            
        try:
            outdated_packages = json.loads(output)
            print(f"📦 Found {len(outdated_packages)} outdated packages")
            return outdated_packages
        except json.JSONDecodeError:
            print("❌ Failed to parse outdated packages JSON")
            return []
            
    def check_security_advisories(self, package_name: str, current_version: str, latest_version: str) -> List[Dict]:
        """Check for security advisories for a package."""
        try:
            # Use PyPI JSON API to check for vulnerabilities
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            response.raise_for_status()
            
            package_data = response.json()
            
            # Check for security advisories (simplified approach)
            # In a real implementation, you'd integrate with services like:
            # - GitHub Security Advisories
            # - OSV (Open Source Vulnerabilities)
            # - PyUp Safety API
            
            vulnerabilities = []
            
            # Placeholder for actual vulnerability checking
            # This would be replaced with real API calls
            
            return vulnerabilities
            
        except Exception as e:
            print(f"⚠️  Could not check security for {package_name}: {e}")
            return []
            
    def is_major_version_update(self, current: str, latest: str) -> bool:
        """Check if update is a major version change."""
        try:
            current_ver = version.parse(current)
            latest_ver = version.parse(latest)
            
            # Consider it major if the major version number changes
            return current_ver.major != latest_ver.major
        except Exception:
            return False
            
    def backup_dependencies(self):
        """Create backup of current dependency files."""
        backup_dir = self.project_root / "dependency_backups"
        backup_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup pyproject.toml
        if self.pyproject_path.exists():
            backup_path = backup_dir / f"pyproject.toml.{timestamp}"
            backup_path.write_text(self.pyproject_path.read_text())
            print(f"📁 Backed up pyproject.toml to {backup_path}")
            
        # Backup requirements files
        for req_file in self.requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                backup_path = backup_dir / f"{req_file}.{timestamp}"
                backup_path.write_text(req_path.read_text())
                print(f"📁 Backed up {req_file} to {backup_path}")
                
    def run_tests(self) -> bool:
        """Run tests to ensure updates don't break functionality."""
        print("🧪 Running tests to verify updates...")
        
        # Run unit tests
        output, code = self.run_command([
            "python", "-m", "pytest", "tests/unit/", 
            "-v", "--tb=short", "-x"  # Stop on first failure
        ])
        
        if code != 0:
            print("❌ Unit tests failed")
            print(output)
            return False
            
        # Run integration tests (if they exist and are fast)
        if (self.project_root / "tests" / "integration").exists():
            output, code = self.run_command([
                "python", "-m", "pytest", "tests/integration/",
                "-v", "--tb=short", "-x", "-m", "not slow"  # Skip slow tests
            ])
            
            if code != 0:
                print("❌ Integration tests failed")
                print(output)
                return False
                
        print("✅ Tests passed")
        return True
        
    def run_security_check(self) -> bool:
        """Run security checks on updated dependencies."""
        print("🛡️  Running security checks...")
        
        # Run safety check
        output, code = self.run_command(["safety", "check", "--short-report"])
        
        if code != 0:
            print("❌ Security vulnerabilities found:")
            print(output)
            return False
            
        # Run bandit on code
        output, code = self.run_command([
            "bandit", "-r", "meta_prompt_evolution/",
            "-f", "txt", "--severity-level", "high"
        ])
        
        if code != 0:
            print("❌ Security issues found in code:")
            print(output)
            return False
            
        print("✅ Security checks passed")
        return True
        
    def update_package(self, package: Dict[str, str], force_update: bool = False) -> bool:
        """Update a single package with safety checks."""
        name = package["name"]
        current_version = package["version"]
        latest_version = package["latest_version"]
        
        print(f"\n📦 Updating {name}: {current_version} → {latest_version}")
        
        # Check if it's a major version update
        is_major = self.is_major_version_update(current_version, latest_version)
        
        if is_major and not force_update:
            print(f"⚠️  Major version update detected for {name}")
            print("    Skipping automatic update (use --force-major to override)")
            self.failed_updates.append({
                "package": name,
                "reason": "Major version update requires manual review",
                "current": current_version,
                "latest": latest_version
            })
            return False
            
        # Check security advisories
        advisories = self.check_security_advisories(name, current_version, latest_version)
        if advisories:
            print(f"🔒 Security advisories found for {name}")
            for advisory in advisories:
                print(f"    - {advisory.get('title', 'Security issue')}")
                
        # Attempt the update
        output, code = self.run_command([
            "pip", "install", f"{name}=={latest_version}"
        ])
        
        if code != 0:
            print(f"❌ Failed to update {name}")
            print(output)
            self.failed_updates.append({
                "package": name,
                "reason": "Installation failed",
                "current": current_version,
                "latest": latest_version,
                "error": output
            })
            return False
            
        self.updated_packages.append({
            "package": name,
            "from_version": current_version,
            "to_version": latest_version,
            "is_major": is_major,
            "security_update": len(advisories) > 0
        })
        
        print(f"✅ Successfully updated {name}")
        return True
        
    def generate_freeze_file(self):
        """Generate updated requirements.txt files."""
        print("📝 Generating updated requirements...")
        
        # Generate full dependency list
        output, code = self.run_command(["pip", "freeze"])
        
        if code == 0:
            # Write to requirements.txt
            requirements_path = self.project_root / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write(output)
            print(f"✅ Updated {requirements_path}")
            
            # Filter for production requirements (exclude dev dependencies)
            prod_requirements = []
            dev_packages = {
                "pytest", "black", "ruff", "mypy", "pre-commit",
                "sphinx", "bandit", "safety", "coverage"
            }
            
            for line in output.split('\n'):
                if line and not any(pkg in line.lower() for pkg in dev_packages):
                    prod_requirements.append(line)
                    
            prod_path = self.project_root / "requirements.prod.txt"
            with open(prod_path, 'w') as f:
                f.write('\n'.join(prod_requirements))
            print(f"✅ Updated {prod_path}")
            
    def rollback_changes(self):
        """Rollback dependency changes if tests fail."""
        print("🔄 Rolling back changes...")
        
        # Find most recent backup
        backup_dir = self.project_root / "dependency_backups"
        if not backup_dir.exists():
            print("❌ No backups found for rollback")
            return False
            
        backup_files = list(backup_dir.glob("*"))
        if not backup_files:
            print("❌ No backup files found")
            return False
            
        # Sort by modification time and get the most recent
        latest_backups = sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Extract timestamp from latest backup
        timestamps = set()
        for backup in latest_backups[:10]:  # Check last 10 backups
            parts = backup.name.split('.')
            if len(parts) >= 2:
                timestamps.add(parts[-1])
                
        if not timestamps:
            print("❌ Could not identify backup timestamp")
            return False
            
        latest_timestamp = sorted(timestamps)[-1]
        
        # Restore files with matching timestamp
        restored = 0
        for backup_file in backup_dir.glob(f"*.{latest_timestamp}"):
            original_name = '.'.join(backup_file.name.split('.')[:-1])
            original_path = self.project_root / original_name
            
            try:
                original_path.write_text(backup_file.read_text())
                print(f"📁 Restored {original_name}")
                restored += 1
            except Exception as e:
                print(f"❌ Failed to restore {original_name}: {e}")
                
        if restored > 0:
            # Reinstall dependencies from restored files
            output, code = self.run_command(["pip", "install", "-r", "requirements.txt"])
            if code == 0:
                print("✅ Dependencies rolled back successfully")
                return True
                
        print("❌ Rollback failed")
        return False
        
    def generate_update_report(self) -> str:
        """Generate a report of the update process."""
        report = []
        report.append("# 🔄 Dependency Update Report")
        report.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if self.updated_packages:
            report.append(f"## ✅ Successfully Updated ({len(self.updated_packages)} packages)")
            report.append("")
            for pkg in self.updated_packages:
                name = pkg['package']
                from_ver = pkg['from_version']
                to_ver = pkg['to_version']
                flags = []
                
                if pkg['is_major']:
                    flags.append("MAJOR")
                if pkg['security_update']:
                    flags.append("SECURITY")
                    
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                report.append(f"- **{name}**: {from_ver} → {to_ver}{flag_str}")
                
            report.append("")
            
        if self.failed_updates:
            report.append(f"## ❌ Failed Updates ({len(self.failed_updates)} packages)")
            report.append("")
            for pkg in self.failed_updates:
                name = pkg['package']
                reason = pkg['reason']
                current = pkg['current']
                latest = pkg['latest']
                
                report.append(f"- **{name}** ({current} → {latest}): {reason}")
                
            report.append("")
            
        report.append("## 🔍 Next Steps")
        if self.failed_updates:
            report.append("- Review failed updates manually")
            report.append("- Consider major version updates carefully")
            report.append("- Check compatibility with existing code")
            
        report.append("- Run full test suite")
        report.append("- Update documentation if needed")
        report.append("- Monitor application after deployment")
        
        return "\n".join(report)
        

def main():
    """Main function for dependency updates."""
    if "--help" in sys.argv:
        print("Usage: python update-dependencies.py [options]")
        print("Update project dependencies with safety checks.")
        print("")
        print("Options:")
        print("  --dry-run         Show what would be updated without making changes")
        print("  --force-major     Allow major version updates")
        print("  --skip-tests      Skip test execution (not recommended)")
        print("  --packages PKG    Only update specific packages (comma-separated)")
        return
        
    dry_run = "--dry-run" in sys.argv
    force_major = "--force-major" in sys.argv
    skip_tests = "--skip-tests" in sys.argv
    
    # Parse specific packages to update
    specific_packages = set()
    if "--packages" in sys.argv:
        idx = sys.argv.index("--packages")
        if idx + 1 < len(sys.argv):
            specific_packages = set(sys.argv[idx + 1].split(','))
            
    updater = DependencyUpdater()
    
    print("🔄 Starting dependency update process...")
    
    # Get outdated packages
    outdated = updater.get_outdated_packages()
    if not outdated:
        print("✅ All dependencies are up to date!")
        return
        
    # Filter by specific packages if requested
    if specific_packages:
        outdated = [pkg for pkg in outdated if pkg['name'] in specific_packages]
        print(f"📦 Filtering to {len(outdated)} specific packages")
        
    if dry_run:
        print("\n🔍 DRY RUN - Would update the following packages:")
        for pkg in outdated:
            name = pkg['name']
            current = pkg['version']
            latest = pkg['latest_version']
            is_major = updater.is_major_version_update(current, latest)
            flag = " [MAJOR]" if is_major else ""
            print(f"  - {name}: {current} → {latest}{flag}")
        return
        
    # Backup current state
    updater.backup_dependencies()
    
    # Update packages
    for pkg in outdated:
        success = updater.update_package(pkg, force_update=force_major)
        
        # If update successful and tests are enabled, run them
        if success and not skip_tests:
            if not updater.run_tests():
                print("❌ Tests failed after update, rolling back...")
                updater.rollback_changes()
                break
                
            if not updater.run_security_check():
                print("❌ Security check failed after update, rolling back...")
                updater.rollback_changes()
                break
                
    # Generate updated requirements files
    if updater.updated_packages:
        updater.generate_freeze_file()
        
    # Generate and display report
    report = updater.generate_update_report()
    print("\n" + report)
    
    # Save report to file
    report_path = Path("dependency-update-report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to {report_path}")
    
    # Exit with appropriate code
    if updater.failed_updates:
        print(f"\n⚠️  {len(updater.failed_updates)} packages failed to update")
        sys.exit(1)
    elif updater.updated_packages:
        print(f"\n✅ Successfully updated {len(updater.updated_packages)} packages")
    else:
        print("\n✨ No updates were needed")


if __name__ == "__main__":
    main()