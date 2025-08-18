#!/usr/bin/env python3
"""
Automation scheduler for Meta-Prompt-Evolution-Hub.
Orchestrates automated maintenance tasks on a schedule.
"""

import json
import subprocess
import sys
import time
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Callable
import logging


class AutomationScheduler:
    """Manages scheduled automation tasks for repository maintenance."""
    
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or (self.project_root / "scripts" / "automation-config.json")
        self.scripts_dir = Path(__file__).parent
        self.load_config()
        self.setup_logging()
        self.task_history = []
        
    def setup_logging(self):
        """Setup logging for automation tasks."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "automation.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """Load automation configuration."""
        default_config = {
            "schedules": {
                "metrics_collection": {
                    "enabled": True,
                    "schedule": "daily",
                    "time": "02:00",
                    "script": "collect-metrics.py",
                    "description": "Collect and update project metrics"
                },
                "health_monitoring": {
                    "enabled": True,
                    "schedule": "hourly",
                    "time": None,
                    "script": "health-monitor.py",
                    "description": "Monitor repository health"
                },
                "dependency_updates": {
                    "enabled": True,
                    "schedule": "weekly",
                    "day": "monday",
                    "time": "03:00",
                    "script": "update-dependencies.py",
                    "args": ["--skip-tests"],
                    "description": "Update project dependencies"
                },
                "security_scan": {
                    "enabled": True,
                    "schedule": "daily",
                    "time": "01:00",
                    "script": "health-monitor.py",
                    "args": ["--config", "security-config.json"],
                    "description": "Run security health checks"
                },
                "cleanup_logs": {
                    "enabled": True,
                    "schedule": "weekly",
                    "day": "sunday",
                    "time": "04:00",
                    "script": "cleanup-logs.py",
                    "description": "Clean up old log files"
                }
            },
            "notifications": {
                "enabled": True,
                "on_failure": True,
                "on_success": False,
                "channels": ["log"]
            },
            "retention": {
                "task_history_days": 30,
                "log_files_days": 14
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self.config = {**default_config, **loaded_config}
            except Exception as e:
                self.logger.error(f"Error loading config, using defaults: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
            
    def save_config(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def run_script(self, script_name: str, args: List[str] = None) -> Dict:
        """Run a automation script and return result."""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script {script_name} not found",
                "output": "",
                "duration": 0
            }
            
        command = ["python3", str(script_path)]
        if args:
            command.extend(args)
            
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800  # 30 minute timeout
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "duration": duration,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timeout",
                "output": "",
                "duration": time.time() - start_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "duration": time.time() - start_time
            }
            
    def execute_task(self, task_name: str, task_config: Dict):
        """Execute a scheduled task."""
        if not task_config.get("enabled", True):
            self.logger.info(f"Task {task_name} is disabled, skipping")
            return
            
        self.logger.info(f"Starting task: {task_name}")
        
        script = task_config["script"]
        args = task_config.get("args", [])
        
        result = self.run_script(script, args)
        
        # Record task execution
        task_record = {
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            "script": script,
            "args": args,
            "success": result["success"],
            "duration": result["duration"],
            "output": result["output"][:1000] if result["output"] else "",  # Truncate long output
            "error": result.get("error")
        }
        
        self.task_history.append(task_record)
        
        # Log result
        if result["success"]:
            self.logger.info(f"Task {task_name} completed successfully in {result['duration']:.1f}s")
            if self.config["notifications"]["on_success"]:
                self.send_notification(task_name, "SUCCESS", task_record)
        else:
            self.logger.error(f"Task {task_name} failed: {result.get('error', 'Unknown error')}")
            if self.config["notifications"]["on_failure"]:
                self.send_notification(task_name, "FAILURE", task_record)
                
        # Save task history
        self.save_task_history()
        
    def send_notification(self, task_name: str, status: str, task_record: Dict):
        """Send notification about task execution."""
        if not self.config["notifications"]["enabled"]:
            return
            
        message = f"Task {task_name} - {status}"
        if status == "FAILURE":
            message += f"\nError: {task_record['error']}"
        else:
            message += f"\nDuration: {task_record['duration']:.1f}s"
            
        # For now, just log the notification
        # In a real implementation, this would send to configured channels
        self.logger.info(f"NOTIFICATION: {message}")
        
    def save_task_history(self):
        """Save task execution history."""
        history_file = self.project_root / "task-history.jsonl"
        
        # Append latest records
        with open(history_file, 'a') as f:
            for record in self.task_history:
                f.write(json.dumps(record) + '\n')
                
        self.task_history = []  # Clear after saving
        
        # Clean up old history
        self.cleanup_old_history()
        
    def cleanup_old_history(self):
        """Clean up old task history records."""
        history_file = self.project_root / "task-history.jsonl"
        if not history_file.exists():
            return
            
        cutoff_date = datetime.now() - timedelta(days=self.config["retention"]["task_history_days"])
        
        try:
            # Read all records
            records = []
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_date = datetime.fromisoformat(record["timestamp"])
                        if record_date > cutoff_date:
                            records.append(record)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
                        
            # Write back only recent records
            with open(history_file, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
                    
            self.logger.info(f"Cleaned up task history, kept {len(records)} recent records")
            
        except Exception as e:
            self.logger.error(f"Failed to clean up task history: {e}")
            
    def setup_schedules(self):
        """Setup all scheduled tasks."""
        for task_name, task_config in self.config["schedules"].items():
            if not task_config.get("enabled", True):
                continue
                
            schedule_type = task_config["schedule"]
            
            if schedule_type == "hourly":
                schedule.every().hour.do(self.execute_task, task_name, task_config)
                self.logger.info(f"Scheduled {task_name} to run hourly")
                
            elif schedule_type == "daily":
                time_str = task_config.get("time", "02:00")
                schedule.every().day.at(time_str).do(self.execute_task, task_name, task_config)
                self.logger.info(f"Scheduled {task_name} to run daily at {time_str}")
                
            elif schedule_type == "weekly":
                day = task_config.get("day", "monday")
                time_str = task_config.get("time", "02:00")
                
                if day == "monday":
                    schedule.every().monday.at(time_str).do(self.execute_task, task_name, task_config)
                elif day == "tuesday":
                    schedule.every().tuesday.at(time_str).do(self.execute_task, task_name, task_config)
                elif day == "wednesday":
                    schedule.every().wednesday.at(time_str).do(self.execute_task, task_name, task_config)
                elif day == "thursday":
                    schedule.every().thursday.at(time_str).do(self.execute_task, task_name, task_config)
                elif day == "friday":
                    schedule.every().friday.at(time_str).do(self.execute_task, task_name, task_config)
                elif day == "saturday":
                    schedule.every().saturday.at(time_str).do(self.execute_task, task_name, task_config)
                elif day == "sunday":
                    schedule.every().sunday.at(time_str).do(self.execute_task, task_name, task_config)
                    
                self.logger.info(f"Scheduled {task_name} to run weekly on {day} at {time_str}")
                
            elif schedule_type == "interval":
                minutes = task_config.get("minutes", 60)
                schedule.every(minutes).minutes.do(self.execute_task, task_name, task_config)
                self.logger.info(f"Scheduled {task_name} to run every {minutes} minutes")
                
    def run_scheduler(self):
        """Run the scheduler continuously."""
        self.logger.info("🤖 Starting automation scheduler...")
        self.setup_schedules()
        
        # Run initial health check
        self.logger.info("Running initial health check...")
        self.execute_task("startup_health_check", {
            "script": "health-monitor.py",
            "args": ["--no-notify"],
            "enabled": True
        })
        
        self.logger.info("Scheduler is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
            
    def run_task_now(self, task_name: str):
        """Run a specific task immediately."""
        if task_name not in self.config["schedules"]:
            self.logger.error(f"Unknown task: {task_name}")
            return False
            
        task_config = self.config["schedules"][task_name]
        self.execute_task(task_name, task_config)
        return True
        
    def list_tasks(self):
        """List all configured tasks."""
        print("📋 Configured automation tasks:")
        print()
        
        for task_name, task_config in self.config["schedules"].items():
            status = "✅ Enabled" if task_config.get("enabled", True) else "❌ Disabled"
            schedule_info = task_config["schedule"]
            
            if "time" in task_config:
                schedule_info += f" at {task_config['time']}"
            if "day" in task_config:
                schedule_info += f" on {task_config['day']}"
                
            print(f"**{task_name}** ({status})")
            print(f"  Schedule: {schedule_info}")
            print(f"  Script: {task_config['script']}")
            print(f"  Description: {task_config.get('description', 'No description')}")
            print()
            
    def show_task_history(self, days: int = 7):
        """Show recent task execution history."""
        history_file = self.project_root / "task-history.jsonl"
        if not history_file.exists():
            print("No task history found")
            return
            
        cutoff_date = datetime.now() - timedelta(days=days)
        records = []
        
        try:
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_date = datetime.fromisoformat(record["timestamp"])
                        if record_date > cutoff_date:
                            records.append(record)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
                        
            if not records:
                print(f"No task history found for the last {days} days")
                return
                
            print(f"📊 Task history for the last {days} days:")
            print()
            
            for record in sorted(records, key=lambda x: x["timestamp"], reverse=True):
                timestamp = datetime.fromisoformat(record["timestamp"]).strftime("%Y-%m-%d %H:%M")
                status = "✅ SUCCESS" if record["success"] else "❌ FAILURE"
                duration = f"{record['duration']:.1f}s"
                
                print(f"{timestamp} - {record['task_name']} - {status} ({duration})")
                
                if not record["success"] and record.get("error"):
                    print(f"  Error: {record['error']}")
                print()
                
        except Exception as e:
            print(f"Error reading task history: {e}")


def main():
    """Main function for automation scheduler."""
    if "--help" in sys.argv:
        print("Usage: python automation-scheduler.py [command] [options]")
        print("Manage and run automated repository maintenance tasks.")
        print("")
        print("Commands:")
        print("  run                    Run the scheduler (default)")
        print("  list                   List all configured tasks")
        print("  history [days]         Show task execution history")
        print("  execute TASK           Run a specific task now")
        print("")
        print("Options:")
        print("  --config FILE          Use custom configuration file")
        return
        
    config_path = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
            
    scheduler = AutomationScheduler(config_path)
    
    # Parse command
    command = "run"  # default
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        command = sys.argv[1]
        
    if command == "list":
        scheduler.list_tasks()
        
    elif command == "history":
        days = 7
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            days = int(sys.argv[2])
        scheduler.show_task_history(days)
        
    elif command == "execute":
        if len(sys.argv) < 3:
            print("Error: Please specify a task name")
            print("Use 'python automation-scheduler.py list' to see available tasks")
            sys.exit(1)
            
        task_name = sys.argv[2]
        success = scheduler.run_task_now(task_name)
        sys.exit(0 if success else 1)
        
    elif command == "run":
        scheduler.run_scheduler()
        
    else:
        print(f"Unknown command: {command}")
        print("Use --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()