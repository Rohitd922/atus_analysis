#!/usr/bin/env python
"""
Enhanced ATUS experiment runner with resource monitoring, resumable runs, and CLI automation.

This script addresses key reliability issues:
1. CLI arguments to replace interactive prompts (enables batch/cron jobs)
2. Resource monitoring to ensure system stability between rungs
3. Persistent logging for troubleshooting
4. Resume capability with progress tracking
5. Robust error handling and recovery

Usage Examples:
    # Interactive mode (original behavior)
    python run_single_experiments.py
    
    # Automated mode - routing ladder only
    python run_single_experiments.py --mode routing --delay 60 --max-memory 75
    
    # Automated mode - full ladder with hazard
    python run_single_experiments.py --mode full --delay 120 --log-file experiments.log
    
    # Single rung
    python run_single_experiments.py --mode single --rung R3 --hazard
    
    # Resume interrupted run
    python run_single_experiments.py --mode routing --resume --progress-file progress.json
"""

import argparse
import json
import logging
import psutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import safety configuration
try:
    from .safety_config import (
        MAX_MEMORY_PERCENT, MAX_CPU_PERCENT, DEFAULT_DELAY,
        RESOURCE_CHECK_INTERVAL, SETTLE_TIME, DEFAULT_TIMEOUT_MINUTES,
        RESOURCE_WAIT_TIMEOUT, SAFETY_NOTES
    )
except ImportError:
    # Fallback values if safety_config.py is not available
    MAX_MEMORY_PERCENT = 70.0
    MAX_CPU_PERCENT = 80.0
    DEFAULT_DELAY = 120
    RESOURCE_CHECK_INTERVAL = 15
    SETTLE_TIME = 60
    DEFAULT_TIMEOUT_MINUTES = 180
    RESOURCE_WAIT_TIMEOUT = 45
    SAFETY_NOTES = "Use conservative settings for safe execution."

class ResourceMonitor:
    """Monitor system resources and wait for availability."""
    
    def __init__(self, max_memory_percent: float = MAX_MEMORY_PERCENT, 
                 max_cpu_percent: float = MAX_CPU_PERCENT, 
                 check_interval: int = RESOURCE_CHECK_INTERVAL, 
                 settle_time: int = SETTLE_TIME):
        self.max_memory = max_memory_percent
        self.max_cpu = max_cpu_percent
        self.check_interval = check_interval
        self.settle_time = settle_time
        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")
    
    def get_resource_status(self) -> Dict:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'cpu_percent': cpu,
            'memory_ok': memory.percent < self.max_memory,
            'cpu_ok': cpu < self.max_cpu,
            'system_ok': memory.percent < self.max_memory and cpu < self.max_cpu
        }
    
    def wait_for_resources(self, timeout_minutes: int = RESOURCE_WAIT_TIMEOUT) -> bool:
        """
        Wait until system resources are available.
        
        Args:
            timeout_minutes: Maximum time to wait for resources
            
        Returns:
            True if resources became available, False if timeout
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        self.logger.info(f"Waiting for system resources (max memory: {self.max_memory}%, max CPU: {self.max_cpu}%)")
        
        consecutive_good_checks = 0
        required_consecutive = max(1, self.settle_time // self.check_interval)
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_resource_status()
            
            self.logger.debug(f"Resource check - Memory: {status['memory_percent']:.1f}%, "
                            f"CPU: {status['cpu_percent']:.1f}%, "
                            f"Available RAM: {status['memory_available_gb']:.1f}GB")
            
            if status['system_ok']:
                consecutive_good_checks += 1
                if consecutive_good_checks >= required_consecutive:
                    self.logger.info(f"✓ System resources stable for {self.settle_time}s, proceeding...")
                    return True
            else:
                consecutive_good_checks = 0
                if not status['memory_ok']:
                    self.logger.warning(f"High memory usage: {status['memory_percent']:.1f}% (limit: {self.max_memory}%)")
                if not status['cpu_ok']:
                    self.logger.warning(f"High CPU usage: {status['cpu_percent']:.1f}% (limit: {self.max_cpu}%)")
            
            time.sleep(self.check_interval)
        
        self.logger.error(f"Timeout waiting for resources after {timeout_minutes} minutes")
        return False

class ProgressTracker:
    """Track experiment progress for resumable runs."""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.logger = logging.getLogger(f"{__name__}.ProgressTracker")
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load existing progress or create new."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                self.logger.info(f"Loaded existing progress from {self.progress_file}")
                return progress
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}, starting fresh")
        
        return {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'started_at': datetime.now().isoformat(),
            'completed_rungs': [],
            'failed_rungs': [],
            'current_rung': None,
            'total_rungs': 0,
            'mode': None
        }
    
    def save_progress(self):
        """Save current progress to file."""
        try:
            self.progress['last_updated'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")
    
    def start_experiment(self, mode: str, rungs: List[str]):
        """Initialize new experiment."""
        self.progress.update({
            'mode': mode,
            'total_rungs': len(rungs),
            'planned_rungs': rungs,
            'started_at': datetime.now().isoformat()
        })
        self.save_progress()
    
    def start_rung(self, rung: str):
        """Mark rung as started."""
        self.progress['current_rung'] = rung
        self.progress[f'{rung}_started_at'] = datetime.now().isoformat()
        self.save_progress()
        self.logger.info(f"Started rung {rung}")
    
    def complete_rung(self, rung: str, success: bool):
        """Mark rung as completed."""
        self.progress['current_rung'] = None
        self.progress[f'{rung}_completed_at'] = datetime.now().isoformat()
        self.progress[f'{rung}_success'] = success
        
        if success:
            if rung not in self.progress['completed_rungs']:
                self.progress['completed_rungs'].append(rung)
        else:
            if rung not in self.progress['failed_rungs']:
                self.progress['failed_rungs'].append(rung)
        
        self.save_progress()
        status = "✓ completed" if success else "✗ failed"
        self.logger.info(f"Rung {rung} {status}")
    
    def get_remaining_rungs(self, planned_rungs: List[str]) -> List[str]:
        """Get list of rungs that still need to be run."""
        completed = set(self.progress['completed_rungs'])
        return [rung for rung in planned_rungs if rung not in completed]
    
    def get_summary(self) -> str:
        """Get progress summary."""
        completed = len(self.progress['completed_rungs'])
        failed = len(self.progress['failed_rungs'])
        total = self.progress.get('total_rungs', 0)
        
        return (f"Progress: {completed}/{total} completed, {failed} failed, "
                f"Current: {self.progress.get('current_rung', 'None')}")

class ExperimentRunner:
    """Enhanced experiment runner with all improvements."""
    
    RUNG_SPECS = {
        "R1": "region",
        "R2": "region,sex", 
        "R3": "region,employment",
        "R4": "region,day_type",
        "R5": "region,hh_size_band",
        "R6": "employment,day_type,hh_size_band,sex,region,quarter",
        "R7": "employment,day_type,hh_size_band,sex,region,quarter"
    }
    
    def __init__(self, monitor: ResourceMonitor, tracker: ProgressTracker, 
                 delay: int = DEFAULT_DELAY, timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES):
        self.monitor = monitor
        self.tracker = tracker
        self.delay = delay
        self.timeout_minutes = timeout_minutes
        self.logger = logging.getLogger(f"{__name__}.ExperimentRunner")
    
    def run_single_rung(self, rung: str, include_hazard: bool = False) -> bool:
        """
        Run a single rung with full error handling and resource monitoring.
        
        Args:
            rung: Rung identifier (R1-R7)
            include_hazard: Whether to include hazard model
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"=== Preparing rung {rung} ===")
        self.tracker.start_rung(rung)
        
        # Wait for resources before starting
        if not self.monitor.wait_for_resources(self.timeout_minutes // 2):
            self.logger.error(f"Could not obtain sufficient resources for {rung}")
            self.tracker.complete_rung(rung, False)
            return False
        
        # Build command
        cmd = [
            sys.executable, "-m", "atus_analysis.scripts.run_ladders",
            "single", "--rung", rung, "--gc"
        ]
        
        if include_hazard and rung != "R7":  # R7 includes hazard by default
            cmd.append("--also_b2")
        
        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            start_time = time.time()
            
            # Run with timeout
            result = subprocess.run(
                cmd, 
                timeout=self.timeout_minutes * 60,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"✓ {rung} completed successfully in {elapsed:.1f} seconds")
            
            # Log any warnings from subprocess
            if result.stderr:
                self.logger.warning(f"Subprocess stderr: {result.stderr}")
            
            self.tracker.complete_rung(rung, True)
            
            # Wait for system to settle after completion
            if self.delay > 0:
                self.logger.info(f"Waiting {self.delay}s for system to settle...")
                time.sleep(self.delay)
                
                # Ensure resources are actually available before continuing
                self.monitor.wait_for_resources(5)  # Quick 5-minute check
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ {rung} timed out after {self.timeout_minutes} minutes")
            self.tracker.complete_rung(rung, False)
            return False
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ {rung} failed with return code {e.returncode}")
            if e.stdout:
                self.logger.error(f"Subprocess stdout: {e.stdout}")
            if e.stderr:
                self.logger.error(f"Subprocess stderr: {e.stderr}")
            self.tracker.complete_rung(rung, False)
            return False
            
        except Exception as e:
            self.logger.error(f"✗ {rung} failed with exception: {e}")
            self.tracker.complete_rung(rung, False)
            return False
    
    def run_experiment_batch(self, rungs: List[str], include_hazard: bool = False, 
                           resume: bool = False) -> Tuple[int, int]:
        """
        Run a batch of experiments.
        
        Args:
            rungs: List of rung names to run
            include_hazard: Whether to include hazard models
            resume: Whether to resume from previous progress
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if resume:
            remaining_rungs = self.tracker.get_remaining_rungs(rungs)
            if len(remaining_rungs) < len(rungs):
                self.logger.info(f"Resuming: {len(remaining_rungs)} rungs remaining")
                self.logger.info(f"Already completed: {self.tracker.progress['completed_rungs']}")
            rungs = remaining_rungs
        else:
            self.tracker.start_experiment(f"batch_{len(rungs)}_rungs", rungs)
        
        if not rungs:
            self.logger.info("No rungs to process")
            return 0, 0
        
        successful = 0
        failed = 0
        
        for i, rung in enumerate(rungs, 1):
            self.logger.info(f"=== Starting rung {i}/{len(rungs)}: {rung} ===")
            
            # For R7, hazard is automatic; for others, use the include_hazard flag
            rung_hazard = include_hazard or (rung == "R7")
            
            if self.run_single_rung(rung, rung_hazard):
                successful += 1
            else:
                failed += 1
                
                # Only ask about continuing if there are actually more rungs to run
                remaining_rungs = len(rungs) - i
                if remaining_rungs > 0:
                    # Ask user whether to continue on failure (if interactive)
                    if sys.stdin.isatty():
                        try:
                            response = input(f"\n{rung} failed. Continue with remaining {remaining_rungs} rungs? (y/n): ")
                            if response.lower() != 'y':
                                self.logger.info("User chose to stop after failure")
                                break
                        except (EOFError, KeyboardInterrupt):
                            self.logger.info("User interrupted, stopping")
                            break
                    else:
                        # In non-interactive mode, log and continue
                        self.logger.warning(f"{rung} failed, continuing with remaining {remaining_rungs} rungs")
                else:
                    # This is the last/only rung, just report the failure
                    self.logger.error(f"{rung} failed - no more rungs to process")
        
        return successful, failed

def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """Set up logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

def get_rungs_for_mode(mode: str) -> List[str]:
    """Get list of rungs for a given mode."""
    if mode == "routing":
        return ["R1", "R2", "R3", "R4", "R5", "R6"]
    elif mode == "full":
        return ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def perform_safety_checks() -> bool:
    """Perform initial safety checks before starting experiments."""
    print("Performing initial safety checks...")
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 4.0:
        print(f"⚠️  WARNING: Only {available_gb:.1f}GB memory available")
        print("   Recommended: At least 4GB free memory")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    else:
        print(f"✓ Memory check passed: {available_gb:.1f}GB available")
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=2)
    if cpu_percent > 50:
        print(f"⚠️  WARNING: Current CPU usage is {cpu_percent:.1f}%")
        print("   Consider closing other applications before starting")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    else:
        print(f"✓ CPU check passed: {cpu_percent:.1f}% usage")
    
    # Check disk space in output directory
    try:
        output_path = Path("atus_analysis/data/models")
        if output_path.exists():
            disk_usage = psutil.disk_usage(str(output_path))
            available_gb = disk_usage.free / (1024**3)
            
            if available_gb < 10.0:
                print(f"⚠️  WARNING: Only {available_gb:.1f}GB disk space available")
                print("   Recommended: At least 10GB free space")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    return False
            else:
                print(f"✓ Disk space check passed: {available_gb:.1f}GB available")
    except Exception:
        print("⚠️  Could not check disk space")
    
    print("✓ Safety checks completed")
    return True

def interactive_mode() -> Tuple[str, List[str], bool, int]:
    """Original interactive mode for backward compatibility."""
    print("ATUS Hierarchical Baseline Experiments - Safe Mode")
    print("Running one rung at a time to avoid system overload")
    print()
    print("SAFETY REMINDERS:")
    print("- Ensure at least 8GB free memory before starting")
    print("- Close unnecessary programs to reduce system load")
    print("- Use conservative delays between rungs")
    print("- Monitor system performance during execution")
    
    print("\nOptions:")
    print("1. Run routing ladder only (R1-R6, B1-H models)")
    print("2. Run routing ladder + R7 with hazard (R1-R7)")
    print("3. Run routing ladder + all rungs with hazard (slower)")
    print("4. Run single rung (specify which one)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    delay = DEFAULT_DELAY
    try:
        custom_delay = input(f"Delay between rungs in seconds (default {delay}): ").strip()
        if custom_delay:
            delay = int(custom_delay)
            if delay < 60:
                print("Warning: Delay less than 60 seconds may cause system overload")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    delay = 60
    except ValueError:
        print(f"Using default delay of {delay} seconds")
    
    if choice == "1":
        return "routing", get_rungs_for_mode("routing"), False, delay
    elif choice == "2":
        return "full", get_rungs_for_mode("full"), False, delay
    elif choice == "3":
        return "full", get_rungs_for_mode("full"), True, delay
    elif choice == "4":
        rung = input("Enter rung name (R1, R2, R3, R4, R5, R6, R7): ").strip().upper()
        if rung in ExperimentRunner.RUNG_SPECS:
            hazard = input("Include hazard model? (y/n): ").strip().lower() == 'y'
            return "single", [rung], hazard, delay
        else:
            raise ValueError("Invalid rung name")
    else:
        raise ValueError("Invalid choice")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced ATUS experiment runner with resource monitoring and resumable runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['routing', 'full', 'single', 'interactive'],
                       default='interactive',
                       help='Experiment mode (default: interactive)')
    
    parser.add_argument('--rung', choices=list(ExperimentRunner.RUNG_SPECS.keys()),
                       help='Single rung to run (required if mode=single)')
    
    parser.add_argument('--hazard', action='store_true',
                       help='Include hazard models for all rungs')
    
    # Resource management
    parser.add_argument('--max-memory', type=float, default=MAX_MEMORY_PERCENT,
                       help=f'Maximum memory usage percentage (default: {MAX_MEMORY_PERCENT})')
    
    parser.add_argument('--max-cpu', type=float, default=MAX_CPU_PERCENT,
                       help=f'Maximum CPU usage percentage (default: {MAX_CPU_PERCENT})')
    
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY,
                       help=f'Delay between rungs in seconds (default: {DEFAULT_DELAY})')
    
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT_MINUTES,
                       help=f'Timeout per rung in minutes (default: {DEFAULT_TIMEOUT_MINUTES})')
    
    # Progress and logging
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous progress')
    
    parser.add_argument('--progress-file', type=Path, 
                       default=Path('experiment_progress.json'),
                       help='Progress tracking file (default: experiment_progress.json)')
    
    parser.add_argument('--log-file', type=Path,
                       help='Optional log file for persistent logging')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_file, args.verbose)
    
    # Perform initial safety checks
    if not perform_safety_checks():
        logger.info("Safety checks failed, aborting")
        return 1
    
    try:
        # Initialize components
        monitor = ResourceMonitor(args.max_memory, args.max_cpu)
        tracker = ProgressTracker(args.progress_file)
        runner = ExperimentRunner(monitor, tracker, args.delay, args.timeout)
        
        # Determine what to run
        if args.mode == 'interactive':
            mode, rungs, include_hazard, delay = interactive_mode()
            runner.delay = delay
        elif args.mode == 'single':
            if not args.rung:
                parser.error("--rung is required when mode=single")
            mode = 'single'
            rungs = [args.rung]
            include_hazard = args.hazard
        else:
            mode = args.mode
            rungs = get_rungs_for_mode(mode)
            include_hazard = args.hazard
        
        logger.info(f"=== Starting ATUS Experiments ===")
        logger.info(f"Mode: {mode}")
        logger.info(f"Rungs: {rungs}")
        logger.info(f"Include hazard: {include_hazard}")
        logger.info(f"Resource limits: {args.max_memory}% memory, {args.max_cpu}% CPU")
        logger.info(f"Delay between rungs: {args.delay}s")
        
        if args.resume:
            logger.info(f"Resume mode: {tracker.get_summary()}")
        
        # Run experiments
        start_time = time.time()
        successful, failed = runner.run_experiment_batch(rungs, include_hazard, args.resume)
        total_time = time.time() - start_time
        
        # Final summary
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT BATCH COMPLETED")
        logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
        logger.info("=" * 60)
        
        # Print final progress
        print(f"\n{tracker.get_summary()}")
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
