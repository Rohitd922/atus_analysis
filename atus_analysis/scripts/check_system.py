#!/usr/bin/env python
"""
System requirements check for ATUS experiments.
"""

import psutil
import pandas as pd
import numpy as np
from pathlib import Path

def check_system():
    print("ATUS Experiment System Check")
    print("="*40)
    
    # Memory check
    memory = psutil.virtual_memory()
    print(f"Memory:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.percent:.1f}%")
    
    # CPU check
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"\nCPU:")
    print(f"  Cores: {cpu_count}")
    if cpu_freq:
        print(f"  Frequency: {cpu_freq.current:.0f} MHz")
    
    # Disk space check
    disk = psutil.disk_usage('.')
    print(f"\nDisk Space:")
    print(f"  Total: {disk.total / (1024**3):.1f} GB")
    print(f"  Free: {disk.free / (1024**3):.1f} GB")
    print(f"  Used: {(disk.used / disk.total) * 100:.1f}%")
    
    # Data size estimate
    try:
        seq_path = Path("atus_analysis/data/sequences/markov_sequences.parquet")
        if seq_path.exists():
            size_mb = seq_path.stat().st_size / (1024**2)
            print(f"\nData Files:")
            print(f"  markov_sequences.parquet: {size_mb:.1f} MB")
            
            # Quick memory estimate
            df = pd.read_parquet(seq_path, columns=['TUCASEID'])
            n_sequences = len(df['TUCASEID'].unique())
            print(f"  Number of sequences: {n_sequences:,}")
            
            # Rough memory estimate per experiment
            est_memory_gb = (size_mb * 3) / 1024  # Conservative estimate
            print(f"  Estimated memory per experiment: {est_memory_gb:.1f} GB")
    except Exception as e:
        print(f"\nCould not check data files: {e}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if memory.available < 4 * (1024**3):  # Less than 4GB available
        print("  ⚠ Limited memory - use single rung mode only")
        print("  ⚠ Close other applications before running experiments")
    elif memory.available < 8 * (1024**3):  # Less than 8GB available
        print("  ✓ Sufficient for single rungs with delays")
        print("  ⚠ Use resource monitoring for full ladder")
    else:
        print("  ✓ Good memory for most experiments")
    
    if cpu_count < 4:
        print("  ⚠ Limited CPU cores - expect longer run times")
    else:
        print("  ✓ Sufficient CPU cores")
    
    if disk.free < 5 * (1024**3):  # Less than 5GB free
        print("  ⚠ Low disk space - models may fail to save")
    else:
        print("  ✓ Sufficient disk space")

if __name__ == "__main__":
    check_system()
