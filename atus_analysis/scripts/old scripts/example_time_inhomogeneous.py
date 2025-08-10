#!/usr/bin/env python3
"""
Example usage of the time-inhomogeneous Markov chain module for ATUS analysis.

This script demonstrates how to:
1. Load your ATUS data 
2. Learn time-inhomogeneous Markov models for different demographic groups
3. Generate synthetic sequences as baseline comparisons
4. Validate the models using occupancy curve comparisons

Run from project root:
    python atus_analysis/scripts/example_time_inhomogeneous.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our time-inhomogeneous Markov module
from time_inhomogeneous_markov import (
    load_atus_data,
    learn_ti_markov_models_for_subgroups,
    simulate_ti_sequences,
    validate_markov_model,
    compute_transition_matrix_statistics
)

def main():
    # Set up paths (relative to project root)
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "atus_analysis" / "data"
    
    states_path = DATA_DIR / "sequences" / "states_10min.npy"
    grid_ids_path = DATA_DIR / "sequences" / "grid_ids.csv"
    catalog_path = DATA_DIR / "sequences" / "state_catalog.json"
    subgroups_path = DATA_DIR / "processed" / "subgroups.parquet"
    
    print("🔄 Loading ATUS data...")
    
    # Check if files exist
    missing_files = []
    for name, path in [
        ("States", states_path),
        ("Grid IDs", grid_ids_path), 
        ("State catalog", catalog_path),
        ("Subgroups", subgroups_path)
    ]:
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("\nPlease run the data preprocessing scripts first:")
        print("   python atus_analysis/scripts/build_markov_sequences_14state.py")
        print("   python atus_analysis/scripts/make_subgroups.py")
        return
    
    # Load the data
    states, weights, K, id_to_sub, demographic_masks = load_atus_data(
        str(states_path),
        str(grid_ids_path),
        str(catalog_path), 
        str(subgroups_path)
    )
    
    print(f"\n📊 Data loaded successfully!")
    print(f"   Sequences: {states.shape[0]:,}")
    print(f"   Time slots: {states.shape[1]} (10-min intervals)")
    print(f"   States: {K} activity categories")
    print(f"   Total weight: {weights.sum():,.0f}")
    
    # Show state labels
    print(f"\n🏷️  Activity states:")
    for state_id, label in id_to_sub.items():
        print(f"   {state_id:2d}: {label}")
    
    print(f"\n🔬 Learning time-inhomogeneous Markov models...")
    
    # Learn models for each demographic group
    ti_models = learn_ti_markov_models_for_subgroups(states, weights, demographic_masks, K)
    
    print(f"\n📈 Model statistics:")
    for group_name, model in ti_models.items():
        stats = compute_transition_matrix_statistics(model['transition_matrices'])
        print(f"   {group_name:10s}: entropy={stats['mean_entropy_overall']:.3f}, "
              f"persistence={stats['mean_persistence']:.3f}")
    
    print(f"\n🎲 Generating synthetic sequences for validation...")
    
    # Generate synthetic sequences for each group
    n_synthetic = min(1000, states.shape[0])  # Don't generate more than we have
    synthetic_results = {}
    
    for group_name, model in ti_models.items():
        print(f"   Generating {n_synthetic} synthetic sequences for {group_name}...")
        
        synthetic_seq = simulate_ti_sequences(
            model['transition_matrices'],
            model['initial_distribution'],
            n_synthetic,
            random_seed=42
        )
        
        # Get the actual data for this group
        mask = demographic_masks[group_name]
        original_data = states[mask] if group_name != 'all' else states
        original_weights = weights[mask] if group_name != 'all' else weights
        
        # Validate the model
        validation = validate_markov_model(
            original_data,
            synthetic_seq,
            original_weights,
            K
        )
        
        synthetic_results[group_name] = {
            'sequences': synthetic_seq,
            'validation': validation
        }
        
        print(f"      Frobenius distance: {validation['frobenius_distance']:.4f}")
    
    print(f"\n📋 Validation Summary:")
    print(f"{'Group':<12} {'Frobenius Dist':<15} {'Max State Diff':<15} {'Mean State Diff':<15}")
    print("-" * 65)
    
    for group_name, result in synthetic_results.items():
        val = result['validation']
        print(f"{group_name:<12} {val['frobenius_distance']:<15.4f} "
              f"{val['max_state_difference']:<15.4f} {val['mean_state_difference']:<15.4f}")
    
    # Create a simple visualization
    print(f"\n📊 Creating occupancy comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    groups_to_plot = ['all', 'male', 'female', 'employed']
    times = np.arange(144) * 10 / 60  # Convert to hours
    
    for i, group_name in enumerate(groups_to_plot):
        if group_name not in synthetic_results:
            continue
            
        ax = axes[i]
        val = synthetic_results[group_name]['validation']
        
        # Plot a few representative states
        states_to_plot = [8, 5, 14]  # SLEEP, SCREENS_LEISURE, OUT_OF_HOME
        
        for state_idx in states_to_plot:
            if state_idx <= K:
                original_occ = val['occupancy_original'][state_idx-1, :]  # Convert to 0-indexed
                synthetic_occ = val['occupancy_synthetic'][state_idx-1, :]
                
                state_name = id_to_sub.get(state_idx, f"State_{state_idx}")
                
                ax.plot(times, original_occ, '-', label=f'{state_name} (original)', alpha=0.7)
                ax.plot(times, synthetic_occ, '--', label=f'{state_name} (synthetic)', alpha=0.7)
        
        ax.set_title(f'{group_name.title()} Group')
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('Occupancy probability')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 24)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = ROOT / "atus_analysis" / "assets" / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "time_inhomogeneous_validation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved plot to: {plot_path}")
    
    plt.show()
    
    print(f"\n✅ Analysis complete!")
    print(f"\nNext steps:")
    print(f"   - Use the learned models as baselines for comparison")
    print(f"   - Generate larger synthetic datasets for robustness testing")
    print(f"   - Analyze temporal patterns in the transition matrices")
    print(f"   - Compare time-inhomogeneous vs. time-homogeneous models")

if __name__ == "__main__":
    main()
