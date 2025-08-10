#!/usr/bin/env python
"""
Lightweight ATUS experiment runner with resource monitoring.
This script monitors system resources and automatically pauses/resumes experiments.
"""

import psutil
import subprocess
import time
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ResourceMonitor:
    def __init__(self, max_memory_percent=80, max_cpu_percent=90):
        self.max_memory = max_memory_percent
        self.max_cpu = max_cpu_percent
        
    def check_resources(self):
        """Check current system resource usage."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu,
            'memory_ok': memory.percent < self.max_memory,
            'cpu_ok': cpu < self.max_cpu
        }
    
    def wait_for_resources(self, check_interval=30):
        """Wait until system resources are available."""
        while True:
            status = self.check_resources()
            logging.info(f"Resource check - Memory: {status['memory_percent']:.1f}%, "
                        f"CPU: {status['cpu_percent']:.1f}%, "
                        f"Available RAM: {status['memory_available_gb']:.1f}GB")
            
            if status['memory_ok'] and status['cpu_ok']:
                logging.info("✓ Resources available, proceeding...")
                return True
            
            logging.warning(f"⚠ System resources high - waiting {check_interval}s...")
            if not status['memory_ok']:
                logging.warning(f"  Memory usage: {status['memory_percent']:.1f}% (max: {self.max_memory}%)")
            if not status['cpu_ok']:
                logging.warning(f"  CPU usage: {status['cpu_percent']:.1f}% (max: {self.max_cpu}%)")
            
            time.sleep(check_interval)

def run_safe_experiment(rung, monitor, also_b2=False):
    """Run a single experiment with resource monitoring."""
    logging.info(f"Preparing to run rung {rung}")
    
    # Check resources before starting
    monitor.wait_for_resources()
    
    # Build command
    cmd = [
        sys.executable, "-m", "atus_analysis.scripts.run_ladders",
        "single", "--rung", rung, "--gc"  # Enable garbage collection
    ]
    
    if also_b2:
        cmd.append("--also_b2")
    
    try:
        logging.info(f"Starting {rung}...")
        start_time = time.time()
        
        # Run with resource monitoring
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor process
        while process.poll() is None:
            status = monitor.check_resources()
            if not status['memory_ok']:
                logging.warning(f"High memory usage during {rung}: {status['memory_percent']:.1f}%")
            time.sleep(10)  # Check every 10 seconds
        
        # Get results
        stdout, stderr = process.communicate()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            logging.info(f"✓ {rung} completed successfully in {elapsed:.1f} seconds")
            return True
        else:
            logging.error(f"✗ {rung} failed with return code {process.returncode}")
            if stderr:
                logging.error(f"Error output: {stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Exception during {rung}: {e}")
        return False

def main():
    print("ATUS Safe Experiment Runner")
    print("="*50)
    
    # Get system info
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"System Info:")
    print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"  CPU cores: {cpu_count}")
    print(f"  Current memory usage: {memory.percent:.1f}%")
    
    # Ask for resource limits
    try:
        max_mem = input(f"\nMax memory usage % (default 75): ").strip()
        max_mem = int(max_mem) if max_mem else 75
        
        max_cpu = input(f"Max CPU usage % (default 85): ").strip()
        max_cpu = int(max_cpu) if max_cpu else 85
    except ValueError:
        max_mem, max_cpu = 75, 85
    
    monitor = ResourceMonitor(max_mem, max_cpu)
    
    # Choose experiment type
    print(f"\nExperiment options:")
    print(f"1. Single rung (safest)")
    print(f"2. Routing ladder (R1-R6)")
    print(f"3. Full ladder (R1-R7)")
    
    choice = input("Choose option (1-3): ").strip()
    
    rungs_to_run = []
    include_hazard = False
    
    if choice == "1":
        rung = input("Enter rung (R1-R7): ").strip().upper()
        if rung in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
            rungs_to_run = [rung]
            include_hazard = input("Include hazard model? (y/n): ").strip().lower() == 'y'
        else:
            print("Invalid rung")
            return
            
    elif choice == "2":
        rungs_to_run = ["R1", "R2", "R3", "R4", "R5", "R6"]
        
    elif choice == "3":
        rungs_to_run = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
        
    else:
        print("Invalid choice")
        return
    
    # Run experiments
    success_count = 0
    for i, rung in enumerate(rungs_to_run, 1):
        logging.info(f"Starting rung {i}/{len(rungs_to_run)}: {rung}")
        
        # For R7, hazard is included by default
        run_hazard = include_hazard or (rung == "R7")
        
        if run_safe_experiment(rung, monitor, run_hazard):
            success_count += 1
            
            # Wait between runs
            if i < len(rungs_to_run):
                logging.info("Waiting 60 seconds between experiments...")
                time.sleep(60)
        else:
            print(f"\n⚠ {rung} failed. Continue with remaining experiments? (y/n)")
            if input().strip().lower() != 'y':
                break
    
    print(f"\n" + "="*50)
    print(f"Completed {success_count}/{len(rungs_to_run)} experiments successfully")
    print("="*50)

if __name__ == "__main__":
    main()
