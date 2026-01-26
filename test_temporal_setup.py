"""
Quick validation script to test temporal embeddings setup.

Run this to verify:
1. All imports work correctly
2. Basic temporal embedding functionality
3. Delta time computation
4. Log scaling
"""
import torch
import numpy as np
from src.models.temporal_embeddings import TemporalEmbedding
from src.data.temporal_utils import compute_delta_times, normalize_delta_times


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 80)
    print("TEST 1: Imports")
    print("=" * 80)
    
    try:
        from src.models.temporal_embeddings import TemporalEmbedding
        from src.models.temporal_model_wrapper import TemporalModelWrapper
        from src.data.temporal_utils import compute_delta_times, align_single_sequence
        from src.data.pretraining_collator import PretrainingCollator
        from src.data.classification_collator import ClassificationCollator
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_delta_time_computation():
    """Test delta time computation."""
    print("\n" + "=" * 80)
    print("TEST 2: Delta Time Computation")
    print("=" * 80)
    
    # Example timestamps (Unix timestamps in seconds)
    timestamps = [
        0,          # Special token
        1609459200, # 2021-01-01 00:00:00
        1609545600, # 2021-01-02 00:00:00 (1 day later)
        1609632000, # 2021-01-03 00:00:00 (1 day later)
        0,          # Special token
        1612224000, # 2021-02-02 00:00:00 (30 days later)
    ]
    
    delta_times = compute_delta_times(timestamps)
    
    print(f"Timestamps: {timestamps}")
    print(f"Delta times: {delta_times}")
    
    # Verify results
    assert delta_times[0] == 0.0, "First event should have delta=0"
    assert delta_times[1] == 0.0, "After special token should have delta=0"
    assert abs(delta_times[2] - 86400) < 1, "1 day should be ~86400 seconds"
    assert abs(delta_times[3] - 86400) < 1, "1 day should be ~86400 seconds"
    assert delta_times[4] == 0.0, "Special token should have delta=0"
    
    print("✓ Delta time computation correct!")
    return True


def test_log_scaling():
    """Test log scaling of delta times."""
    print("\n" + "=" * 80)
    print("TEST 3: Log Scaling")
    print("=" * 80)
    
    # Example delta times (in seconds)
    delta_times = torch.tensor([
        [0.0, 86400.0, 3600.0, 2592000.0],  # 0, 1 day, 1 hour, 30 days
    ])
    
    time_scale = 86400.0  # Days
    
    scaled = normalize_delta_times(delta_times, time_scale)
    
    print(f"Original delta times (seconds): {delta_times[0].tolist()}")
    print(f"Scaled (time_scale={time_scale}): {scaled[0].tolist()}")
    
    # Verify scaling
    expected = [
        np.log1p(0.0 / 86400),         # 0
        np.log1p(86400.0 / 86400),     # log(2) ≈ 0.69
        np.log1p(3600.0 / 86400),      # log(1.042) ≈ 0.04
        np.log1p(2592000.0 / 86400),   # log(31) ≈ 3.43
    ]
    
    print(f"Expected: {expected}")
    
    assert abs(scaled[0, 0] - expected[0]) < 0.01, "Zero should stay zero"
    assert abs(scaled[0, 1] - expected[1]) < 0.01, "1 day scaling incorrect"
    
    print("✓ Log scaling correct!")
    return True


def test_temporal_embedding_layer():
    """Test temporal embedding layer."""
    print("\n" + "=" * 80)
    print("TEST 4: Temporal Embedding Layer")
    print("=" * 80)
    
    hidden_size = 128
    time_scale = 86400.0
    batch_size = 2
    seq_len = 10
    
    # Create temporal embedding layer
    temporal_embedder = TemporalEmbedding(
        hidden_size=hidden_size,
        time_scale=time_scale,
        dropout=0.1
    )
    
    # Create dummy delta times (already log-scaled)
    delta_times = torch.rand(batch_size, seq_len) * 5  # Random values 0-5
    
    # Forward pass
    temporal_embeds = temporal_embedder(delta_times)
    
    print(f"Input shape: {delta_times.shape}")
    print(f"Output shape: {temporal_embeds.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {hidden_size})")
    
    assert temporal_embeds.shape == (batch_size, seq_len, hidden_size), "Shape mismatch"
    assert not torch.isnan(temporal_embeds).any(), "NaN values in embeddings"
    
    print("✓ Temporal embedding layer works!")
    return True


def test_element_wise_addition():
    """Test that temporal embeddings can be added to token embeddings."""
    print("\n" + "=" * 80)
    print("TEST 5: Element-wise Addition with Token Embeddings")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    
    # Simulate token embeddings
    token_embeds = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create temporal embeddings
    temporal_embedder = TemporalEmbedding(hidden_size=hidden_size)
    delta_times = torch.rand(batch_size, seq_len) * 5
    temporal_embeds = temporal_embedder(delta_times)
    
    # Add element-wise
    combined = token_embeds + temporal_embeds
    
    print(f"Token embeddings shape: {token_embeds.shape}")
    print(f"Temporal embeddings shape: {temporal_embeds.shape}")
    print(f"Combined shape: {combined.shape}")
    
    assert combined.shape == token_embeds.shape, "Shape should be preserved"
    assert not torch.isnan(combined).any(), "No NaNs after addition"
    
    print("✓ Element-wise addition works!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TEMPORAL EMBEDDINGS VALIDATION")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Delta Time Computation", test_delta_time_computation()))
    results.append(("Log Scaling", test_log_scaling()))
    results.append(("Temporal Embedding Layer", test_temporal_embedding_layer()))
    results.append(("Element-wise Addition", test_element_wise_addition()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nTemporal embeddings implementation is ready to use!")
        print("To enable, set 'use_temporal_embeddings: true' in your config.")
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED ✗")
        print("=" * 80)
        print("\nPlease fix the issues before using temporal embeddings.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
