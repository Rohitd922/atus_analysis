# Safety Configuration for ATUS Experiments
# This file contains recommended settings for safe experiment execution

# Resource Monitoring Limits
MAX_MEMORY_PERCENT = 70.0      # Conservative memory limit
MAX_CPU_PERCENT = 80.0         # Conservative CPU limit
DEFAULT_DELAY = 120            # 2 minutes between rungs
RESOURCE_CHECK_INTERVAL = 15   # Check every 15 seconds
SETTLE_TIME = 60              # Wait 60 seconds for stable resources

# Timeout Settings
DEFAULT_TIMEOUT_MINUTES = 180  # 3 hours per rung (conservative)
RESOURCE_WAIT_TIMEOUT = 45     # 45 minutes to wait for resources

# Progress Tracking
DEFAULT_PROGRESS_FILE = "experiment_progress.json"
BACKUP_PROGRESS_EVERY = 1      # Backup progress after every rung

# Logging
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Safety Recommendations
SAFETY_NOTES = """
SAFETY RECOMMENDATIONS:
1. Always start with a single rung test: --mode single --rung R1
2. Monitor system resources during execution
3. Use conservative delays (≥120s) between rungs
4. Ensure at least 8GB free memory before starting
5. Avoid running during heavy system usage
6. Keep progress files for resuming interrupted runs
7. Run routing ladder first before full experiments
"""
