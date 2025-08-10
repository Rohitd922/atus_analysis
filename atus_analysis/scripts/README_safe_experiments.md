# Safe ATUS Experiment Runner

## Overview

The `run_single_experiments.py` script is designed to safely run ATUS hierarchical baseline experiments (rungs R1-R7) without creating computational hazards. It includes comprehensive resource monitoring, progress tracking, and safety features.

## Safety Features

1. **Resource Monitoring**: Continuously monitors CPU and memory usage
2. **Progress Tracking**: Saves progress to resume interrupted runs
3. **Configurable Delays**: Prevents system overload between rungs
4. **Timeout Protection**: Prevents runaway processes
5. **Memory Management**: Forces garbage collection between rungs
6. **Initial Safety Checks**: Verifies system resources before starting

## Prerequisites

Make sure you have installed the required dependency:

```bash
# If using pip:
pip install psutil

# If using conda:
conda install psutil
```

## Quick Start

### Test the Script First
```bash
# Run the test script to verify everything is working
python atus_analysis/scripts/test_single_experiments.py
```

### Safe Usage Examples

```bash
# 1. Interactive mode (recommended for first-time users)
python atus_analysis/scripts/run_single_experiments.py

# 2. Test with a single rung first
python atus_analysis/scripts/run_single_experiments.py --mode single --rung R1

# 3. Run routing ladder only (conservative)
python atus_analysis/scripts/run_single_experiments.py --mode routing --delay 120

# 4. Run full experiment with hazard models
python atus_analysis/scripts/run_single_experiments.py --mode full --delay 180 --max-memory 65
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-memory` | 70% | Maximum memory usage before waiting |
| `--max-cpu` | 80% | Maximum CPU usage before waiting |
| `--delay` | 120s | Delay between rungs |
| `--timeout` | 180min | Timeout per rung |

## Progress Tracking

The script automatically saves progress to `experiment_progress.json`. If interrupted, you can resume with:

```bash
python atus_analysis/scripts/run_single_experiments.py --mode routing --resume
```

## Safety Recommendations

1. **Start Small**: Always test with a single rung first
2. **Monitor Resources**: Keep an eye on system performance
3. **Use Conservative Settings**: Prefer longer delays over shorter ones
4. **Free Up Memory**: Close unnecessary applications before starting
5. **Check Disk Space**: Ensure at least 10GB free space
6. **Backup Progress**: The script automatically saves progress files

## Troubleshooting

### High Memory Usage
- Increase `--delay` to allow more time for memory cleanup
- Reduce `--max-memory` threshold
- Close other applications

### Process Timeouts
- Increase `--timeout` value
- Check system resources during execution
- Consider running individual rungs separately

### Script Fails to Start
- Run the test script: `python atus_analysis/scripts/test_single_experiments.py`
- Check that `psutil` is installed
- Verify all baseline scripts are present

## File Outputs

The script generates the same outputs as the original `run_ladders.py` script:
- Model files in `atus_analysis/data/models/R*/`
- Progress tracking in `experiment_progress.json`
- Optional log files with `--log-file`

## Emergency Stop

You can safely interrupt the script with Ctrl+C. Progress will be saved and you can resume later.

## Advanced Usage

### Custom Resource Limits
```bash
# Very conservative settings for shared systems
python atus_analysis/scripts/run_single_experiments.py \
  --mode routing \
  --max-memory 60 \
  --max-cpu 70 \
  --delay 300
```

### Automated Execution
```bash
# Non-interactive mode for scheduled runs
python atus_analysis/scripts/run_single_experiments.py \
  --mode routing \
  --delay 120 \
  --log-file experiments.log
```

### Resume Interrupted Run
```bash
# Resume from where you left off
python atus_analysis/scripts/run_single_experiments.py \
  --mode full \
  --resume \
  --progress-file my_progress.json
```
