"""
Time-Inhomogeneous Markov Chain Analysis

This module provides functions to learn time-inhomogeneous Markov transition matrices
from ATUS activity sequences and generate synthetic sequences from these models.
The time-inhomogeneous approach captures how transition probabilities change throughout
the day, providing a more realistic baseline model for activity sequences.

DATA STRUCTURE EXPLANATION:
- states: np.ndarray of shape (N, T) where:
  * N = number of people/respondents 
  * T = 144 time slots (10-minute intervals covering 24 hours: 0:00-23:50)
  * states[i, t] = activity state ID for person i at time slot t
  * State IDs are integers 1, 2, 3, ..., K representing different activities (1-indexed)
  
- weights: np.ndarray of shape (N,) containing survey weights for each person
  * Used to make results representative of the population
  * Typically derived from ATUS survey weights (TUFNWGTP)
  
- K: int, total number of activity states (14 in the current taxonomy)
  * Determined from the state catalog: K = max(state_ids)
  
"""

import numpy as np
import pandas as pd
import json
from typing import Tuple, Optional, Dict, Union
import warnings


def load_atus_data(
    states_npy_path: str,
    grid_ids_csv_path: str, 
    catalog_json_path: str,
    subgroups_parquet_path: str
) -> Tuple[np.ndarray, np.ndarray, int, Dict, Dict]:
    """
    Load ATUS data for time-inhomogeneous Markov analysis.
    
    Args:
        states_npy_path: Path to states_10min.npy file
        grid_ids_csv_path: Path to grid_ids.csv file  
        catalog_json_path: Path to state_catalog.json file
        subgroups_parquet_path: Path to subgroups.parquet file
        
    Returns:
        tuple: (states, weights, K, id_to_sub, demographic_masks)
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    # Load core data
    states = np.load(states_npy_path)  # Shape: (N, 144)
    ids_df = pd.read_csv(grid_ids_csv_path, dtype={"TUCASEID": "int64"})
    
    with open(catalog_json_path, "r") as f:
        catalog = json.load(f)
    
    # Create state mappings
    id_to_sub = {int(k): v for k, v in catalog["id_to_sub"].items()}
    sub_to_id = {k: int(v) for k, v in catalog["sub_to_id"].items()}
    K = 1 + max(id_to_sub.keys())
    
    # Load demographic data with fallback for parquet engines
    try:
        meta = pd.read_parquet(subgroups_parquet_path, engine="fastparquet")
    except ImportError:
        try:
            meta = pd.read_parquet(subgroups_parquet_path, engine="pyarrow")
        except ImportError:
            meta = pd.read_parquet(subgroups_parquet_path)  # Use default engine
    
    # Handle weights
    if "weight" not in meta.columns:
        if "TUFNWGTP" in meta.columns:
            w = meta["TUFNWGTP"].astype("float64")
            # Handle potential negative or missing weights
            w = np.where(w > 0, w, 1.0)
            meta["weight"] = np.where(np.median(w) > 1e6, w / 1e4, w)
        else:
            meta["weight"] = 1.0
    else:
        # Ensure existing weights are positive
        meta["weight"] = np.where(meta["weight"] > 0, meta["weight"], 1.0)
    
    # Align metadata to states order
    meta_aligned = ids_df.merge(meta, on="TUCASEID", how="left")
    meta_aligned["weight"] = meta_aligned["weight"].fillna(1.0)
    
    # Ensure all weights are positive
    weights = meta_aligned["weight"].to_numpy("float64")
    weights = np.where(weights > 0, weights, 1.0)
    
    # Create demographic masks
    def normalize_column(col):
        return col.str.lower().str.strip().fillna("unknown")
    
    sex_norm = normalize_column(meta_aligned.get("sex", pd.Series(["Unknown"]*len(meta_aligned))))
    employment_norm = normalize_column(meta_aligned.get("employment", pd.Series(["Unknown"]*len(meta_aligned))))  
    day_type_norm = normalize_column(meta_aligned.get("day_type", pd.Series(["Unknown"]*len(meta_aligned))))
    
    demographic_masks = {
        'all': np.ones(len(states), dtype=bool),
        'male': (sex_norm == "male").to_numpy(dtype=bool),
        'female': (sex_norm == "female").to_numpy(dtype=bool),
        'employed': (employment_norm == "employed").to_numpy(dtype=bool),
        'unemployed': (employment_norm == "unemployed").to_numpy(dtype=bool),
        'weekday': (day_type_norm == "weekday").to_numpy(dtype=bool),
        'weekend': (day_type_norm == "weekend").to_numpy(dtype=bool)
    }
    
    print(f"Loaded data: {states.shape} state sequences, K={K} states")
    print(f"Weight summary: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
    print(f"Demographic groups:")
    for name, mask in demographic_masks.items():
        print(f"  {name}: {mask.sum()} people ({100*mask.mean():.1f}%)")
    
    return states, weights, K, id_to_sub, demographic_masks


def validate_inputs(states: np.ndarray, weights: np.ndarray, K: int) -> None:
    """
    Validate that inputs are properly formatted for Markov analysis.
    
    Args:
        states: State sequence array
        weights: Weight array  
        K: Number of states
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(states, np.ndarray):
        raise ValueError("states must be a numpy array")
    
    if states.ndim != 2:
        raise ValueError(f"states must be 2D array, got shape {states.shape}")
    
    N, T = states.shape
    if T != 144:
        warnings.warn(f"Expected T=144 time slots, got T={T}")
    
    if not isinstance(weights, np.ndarray):
        raise ValueError("weights must be a numpy array")
    
    if weights.shape != (N,):
        raise ValueError(f"weights shape {weights.shape} doesn't match states shape {states.shape}")
    
    if not isinstance(K, int) or K <= 0:
        raise ValueError(f"K must be positive integer, got {K}")
    
    # Check state values (1-indexed in your data)
    if np.any(states < 1) or np.any(states > K):
        raise ValueError(f"State values must be in [1, {K}], got range [{states.min()}, {states.max()}]")
    
    # Check weights
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    
    if np.sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero")
    
    print(f"Input validation passed: {N} sequences x {T} time slots, {K} states")


def weighted_time_inhomogeneous_transition_matrices(
    states: np.ndarray, 
    weights: np.ndarray, 
    K: int
) -> np.ndarray:
    """
    Compute weighted time-inhomogeneous transition matrices P_t(s, s').
    
    This function learns a separate transition matrix for each time step in the day,
    capturing how activity transition probabilities change over time. Each matrix
    P_t[i, j] represents P(X_{t+1}=j | X_t=i) for time step t.
    
    Args:
        states (np.ndarray): Shape (N, T) array of state sequences.
                           N = number of sequences, T = number of time steps (144 for 10-min intervals).
                           states[i, t] = activity state ID for person i at time slot t.
        weights (np.ndarray): Shape (N,) array of survey weights for each sequence.
        K (int): Number of activity states in the Markov chain.
        
    Returns:
        np.ndarray: Shape (T-1, K, K) array of transition matrices.
                   mats[t, i, j] = P(X_{t+1}=j | X_t=i) at time step t.
    """
    # Validate inputs
    validate_inputs(states, weights, K)
    
    N, T = states.shape
    
    if N == 0:
        warnings.warn("Empty state sequences provided. Returning zero matrices.")
        return np.zeros((T - 1, K, K), dtype="float64")
    
    # Initialize transition matrices for each time step
    mats = np.zeros((T - 1, K, K), dtype="float64")

    for t in range(T - 1):
        src_states = states[:, t] - 1  # Convert to 0-indexed for matrix indexing
        dst_states = states[:, t + 1] - 1  # Convert to 0-indexed for matrix indexing
        
        # Accumulate weighted counts for this time step
        counts_t = np.zeros((K, K), dtype="float64")
        np.add.at(counts_t, (src_states, dst_states), weights)
        
        # Row-normalize to get probabilities
        row_sums = counts_t.sum(axis=1, keepdims=True)
        probs_t = np.divide(counts_t, row_sums, where=row_sums > 0)
        
        # Handle rows with no transitions (set to uniform distribution)
        zero_rows = (row_sums == 0).flatten()
        if np.any(zero_rows):
            probs_t[zero_rows, :] = 1.0 / K
            
        mats[t] = probs_t
        
    return mats


def learn_initial_state_distribution(
    states: np.ndarray, 
    weights: np.ndarray, 
    K: int
) -> np.ndarray:
    """
    Learn the initial state distribution from data.
    
    Args:
        states (np.ndarray): Shape (N, T) array of state sequences.
        weights (np.ndarray): Shape (N,) array of survey weights for each sequence.
        K (int): Number of activity states in the Markov chain.
        
    Returns:
        np.ndarray: Shape (K,) array of initial state probabilities.
                   p0[s] = P(X_0 = s) = probability of starting the day in state s.
    """
    validate_inputs(states, weights, K)
    
    N = states.shape[0]
    p0 = np.zeros(K, dtype="float64")
    initial_states = states[:, 0] - 1  # Convert to 0-indexed for array indexing
    
    np.add.at(p0, initial_states, weights)
    
    # Normalize to get probabilities
    if p0.sum() == 0:
        warnings.warn("No valid initial states found. Using uniform distribution.")
        p0 = np.ones(K) / K
    else:
        p0 /= p0.sum()
    
    return p0


def simulate_ti_sequences(
    ti_mats: np.ndarray, 
    p0: np.ndarray, 
    n_seq: int,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic sequences from time-inhomogeneous transition matrices.
    
    Args:
        ti_mats (np.ndarray): Shape (T-1, K, K) transition matrices.
        p0 (np.ndarray): Shape (K,) initial state distribution.
        n_seq (int): Number of sequences to generate.
        random_seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Shape (n_seq, T) of generated state sequences.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    T_minus_1, K, K_check = ti_mats.shape
    if K != K_check:
        raise ValueError(f"Transition matrices must be square. Got shape {ti_mats.shape}")
    
    if len(p0) != K:
        raise ValueError(f"Initial distribution length {len(p0)} doesn't match K={K}")
    
    if not np.isclose(p0.sum(), 1.0):
        raise ValueError(f"Initial distribution doesn't sum to 1. Sum = {p0.sum()}")
    
    T = T_minus_1 + 1
    sequences = np.zeros((n_seq, T), dtype=np.int32)
    
    # Draw initial states
    initial_states = np.random.choice(K, size=n_seq, p=p0)
    sequences[:, 0] = initial_states + 1  # Convert back to 1-indexed
    
    # Simulate transitions step-by-step
    for t in range(T_minus_1):
        current_states = sequences[:, t] - 1  # Convert to 0-indexed for matrix lookup
        probs_t = ti_mats[t]  # KxK matrix for this time step
        
        # Validate transition matrix
        if not np.allclose(probs_t.sum(axis=1), 1.0):
            warnings.warn(f"Transition matrix at time {t} rows don't sum to 1")
        
        # Draw next state for each sequence
        next_states = np.zeros(n_seq, dtype=np.int32)
        for i in range(n_seq):
            s_current = current_states[i]
            if 0 <= s_current < K:
                try:
                    next_states[i] = np.random.choice(K, p=probs_t[s_current])
                except ValueError:
                    # Handle edge case where probabilities don't sum to 1
                    normalized_probs = probs_t[s_current] / probs_t[s_current].sum()
                    next_states[i] = np.random.choice(K, p=normalized_probs)
            else:
                # Invalid state, choose randomly
                next_states[i] = np.random.choice(K)
                
        sequences[:, t + 1] = next_states + 1  # Convert back to 1-indexed
        
    return sequences


def compute_transition_matrix_statistics(ti_mats: np.ndarray) -> dict:
    """
    Compute various statistics about the time-inhomogeneous transition matrices.
    
    Args:
        ti_mats (np.ndarray): Shape (T-1, K, K) transition matrices.
        
    Returns:
        dict: Dictionary containing various statistics.
    """
    T_minus_1, K, _ = ti_mats.shape
    
    stats = {
        'n_time_steps': T_minus_1,
        'n_states': K,
        'mean_entropy': np.zeros(T_minus_1),
        'temporal_variance': np.zeros((K, K)),
        'persistence_probability': np.zeros((T_minus_1, K)),
        'off_diagonal_strength': np.zeros(T_minus_1)
    }
    
    # Compute entropy for each time step
    for t in range(T_minus_1):
        P_t = ti_mats[t]
        # Entropy of each row (state)
        row_entropies = -np.sum(P_t * np.log(P_t + 1e-12), axis=1)
        stats['mean_entropy'][t] = np.mean(row_entropies)
        
        # Persistence probabilities (diagonal elements)
        stats['persistence_probability'][t] = np.diag(P_t)
        
        # Off-diagonal strength (sum of off-diagonal elements)
        stats['off_diagonal_strength'][t] = np.sum(P_t) - np.trace(P_t)
    
    # Temporal variance across time steps
    for i in range(K):
        for j in range(K):
            time_series = ti_mats[:, i, j]
            stats['temporal_variance'][i, j] = np.var(time_series)
    
    # Add summary statistics
    stats['mean_persistence'] = np.mean(stats['persistence_probability'])
    stats['max_temporal_variance'] = np.max(stats['temporal_variance'])
    stats['mean_entropy_overall'] = np.mean(stats['mean_entropy'])
    
    return stats


def validate_markov_model(
    original_states: np.ndarray,
    synthetic_states: np.ndarray,
    weights: np.ndarray,
    K: int
) -> dict:
    """
    Validate the time-inhomogeneous Markov model by comparing statistics
    between original and synthetic sequences.
    
    Args:
        original_states (np.ndarray): Original state sequences.
        synthetic_states (np.ndarray): Synthetic state sequences.
        weights (np.ndarray): Weights for original sequences.
        K (int): Number of states.
        
    Returns:
        dict: Validation metrics including Frobenius distance and occupancy comparisons.
    """
    # Define occupancy function locally if not available
    def weighted_occupancy(states, weights, K):
        """Compute weighted occupancy p_s(t): Kx144"""
        N, T = states.shape
        out = np.zeros((K, T), dtype="float64")
        for t in range(T):
            col = states[:, t] - 1  # Convert to 0-indexed for array indexing
            np.add.at(out[:, t], col, weights)
        denom = weights.sum()
        if denom > 0:
            out /= denom
        return out
    
    def frob(A, B):
        """Frobenius distance between matrices A and B"""
        return np.sqrt(np.sum((A - B) ** 2))
    
    # Compute occupancy curves
    occ_original = weighted_occupancy(original_states, weights, K)
    occ_synthetic = weighted_occupancy(synthetic_states, np.ones(len(synthetic_states)), K)
    
    # Compute Frobenius distance
    frob_distance = frob(occ_original, occ_synthetic)
    
    # Compute state-wise differences
    state_differences = np.abs(occ_original - occ_synthetic)
    max_state_diff = np.max(state_differences)
    mean_state_diff = np.mean(state_differences)
    
    # Compute temporal differences
    temporal_diffs = np.abs(occ_original - occ_synthetic)
    max_temporal_diff = np.max(np.mean(temporal_diffs, axis=0))
    
    return {
        'frobenius_distance': frob_distance,
        'max_state_difference': max_state_diff,
        'mean_state_difference': mean_state_diff,
        'max_temporal_difference': max_temporal_diff,
        'occupancy_original': occ_original,
        'occupancy_synthetic': occ_synthetic,
        'state_differences': state_differences
    }


def learn_ti_markov_models_for_subgroups(
    states: np.ndarray,
    weights: np.ndarray,
    masks: dict,
    K: int
) -> dict:
    """
    Learn time-inhomogeneous Markov models for multiple demographic subgroups.
    
    Args:
        states (np.ndarray): Shape (N, T) array of state sequences.
        weights (np.ndarray): Shape (N,) array of weights.
        masks (dict): Dictionary mapping group names to boolean masks.
        K (int): Number of states.
        
    Returns:
        dict: Dictionary mapping group names to (ti_mats, p0) tuples.
    """
    models = {}
    
    for group_name, mask in masks.items():
        if np.any(mask):
            group_states = states[mask]
            group_weights = weights[mask]
            
            # Learn transition matrices and initial distribution
            ti_mats = weighted_time_inhomogeneous_transition_matrices(
                group_states, group_weights, K
            )
            p0 = learn_initial_state_distribution(group_states, group_weights, K)
            
            models[group_name] = {
                'transition_matrices': ti_mats,
                'initial_distribution': p0,
                'n_sequences': len(group_states),
                'total_weight': np.sum(group_weights)
            }
            
            print(f"Learned model for {group_name}: {len(group_states)} sequences, "
                  f"total weight: {np.sum(group_weights):.2f}")
        else:
            print(f"Warning: No sequences found for group {group_name}")
            
    return models


if __name__ == "__main__":
    # Example usage and testing
    print("Time-Inhomogeneous Markov Chain Module")
    print("This module provides functions for learning and simulating time-varying Markov models.")
    print("\nExample usage:")
    print("""
    # 1. Load your ATUS data
    states, weights, K, id_to_sub, masks = load_atus_data(
        'data/sequences/states_10min.npy',
        'data/sequences/grid_ids.csv', 
        'data/sequences/state_catalog.json',
        'data/processed/subgroups.parquet'
    )
    
    # 2. Learn time-inhomogeneous models for demographic groups
    ti_models = learn_ti_markov_models_for_subgroups(states, weights, masks, K)
    
    # 3. Generate synthetic sequences
    synthetic_all = simulate_ti_sequences(
        ti_models['all']['transition_matrices'], 
        ti_models['all']['initial_distribution'], 
        n_seq=1000,
        random_seed=42
    )
    
    # 4. Validate the model
    validation = validate_markov_model(states, synthetic_all, weights, K)
    print(f"Frobenius distance: {validation['frobenius_distance']:.4f}")
    """)
    
    # Simple test with synthetic data
    print("\nRunning test with synthetic data...")
    np.random.seed(42)
    N, T, K = 100, 144, 5
    
    # Generate test data that mimics ATUS structure (1-indexed states)
    test_states = np.random.randint(1, K+1, size=(N, T))  # 1-indexed states
    test_weights = np.random.uniform(0.5, 2.0, size=N)
    
    print(f"Test data: {N} sequences x {T} time slots, {K} states")
    
    try:
        # Test input validation
        validate_inputs(test_states, test_weights, K)
        
        # Learn transition matrices
        ti_mats = weighted_time_inhomogeneous_transition_matrices(test_states, test_weights, K)
        print(f"✓ Learned transition matrices shape: {ti_mats.shape}")
        
        # Learn initial distribution
        p0 = learn_initial_state_distribution(test_states, test_weights, K)
        print(f"✓ Initial distribution sums to: {p0.sum():.4f}")
        
        # Generate synthetic sequences
        synthetic = simulate_ti_sequences(ti_mats, p0, N//2, random_seed=123)
        print(f"✓ Generated synthetic sequences shape: {synthetic.shape}")
        print(f"✓ Synthetic state range: [{synthetic.min()}, {synthetic.max()}]")
        
        # Compute statistics
        stats = compute_transition_matrix_statistics(ti_mats)
        print(f"✓ Mean entropy: {stats['mean_entropy_overall']:.4f}")
        print(f"✓ Mean persistence: {stats['mean_persistence']:.4f}")
        
        # Validate model
        validation = validate_markov_model(test_states, synthetic, test_weights, K)
        print(f"✓ Validation Frobenius distance: {validation['frobenius_distance']:.4f}")
        
        print("\n✅ All tests passed! Module is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
