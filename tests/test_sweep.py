#!/usr/bin/env python3
"""
Test script to validate sweep functionality.

Run with: python -m src.tests.test_sweep
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf

from src.utils.sweep import (
    register_sweep_resolvers,
    set_sweep_index,
    get_sweep_count,
    get_sweep_job_name,
    validate_sweep_config,
)


def test_basic_sweep():
    """Test basic sweep resolution."""
    print("=" * 60)
    print("Test: Basic sweep resolution")
    print("=" * 60)
    
    # Register resolvers
    register_sweep_resolvers()
    
    # Create a test config
    cfg = OmegaConf.create({
        "sweep": {
            "learning_rate": [0.001, 0.01, 0.1],
            "penalty_weight": [-0.1, -0.2, -0.3],
            "seed": [42, 42, 43],
        },
        "config_name": "test_lr${sz:learning_rate}_pen${sz:penalty_weight}_seed${sz:seed}",
        "train": {
            "learning_rate": "${sz:learning_rate}",
            "seed": "${sz:seed}",
        },
        "reward": {
            "penalty_weight": "${sz:penalty_weight}",
        },
    })
    
    # Test sweep count
    count = get_sweep_count(cfg)
    print(f"Sweep count: {count}")
    assert count == 3, f"Expected 3, got {count}"
    print("✓ Sweep count correct")
    
    # Test resolution for each index
    expected = [
        {"lr": 0.001, "pen": -0.1, "seed": 42, "name": "test_lr0.001_pen-0.1_seed42"},
        {"lr": 0.01, "pen": -0.2, "seed": 42, "name": "test_lr0.01_pen-0.2_seed42"},
        {"lr": 0.1, "pen": -0.3, "seed": 43, "name": "test_lr0.1_pen-0.3_seed43"},
    ]
    
    for i, exp in enumerate(expected):
        set_sweep_index(i)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        
        print(f"\nIndex {i}:")
        print(f"  config_name: {resolved['config_name']}")
        print(f"  train.learning_rate: {resolved['train']['learning_rate']}")
        print(f"  train.seed: {resolved['train']['seed']}")
        print(f"  reward.penalty_weight: {resolved['reward']['penalty_weight']}")
        
        assert resolved["config_name"] == exp["name"], f"Name mismatch at index {i}"
        assert resolved["train"]["learning_rate"] == exp["lr"], f"LR mismatch at index {i}"
        assert resolved["train"]["seed"] == exp["seed"], f"Seed mismatch at index {i}"
        assert resolved["reward"]["penalty_weight"] == exp["pen"], f"Penalty mismatch at index {i}"
        
        print(f"  ✓ All values correct")
    
    print("\n✓ Basic sweep test passed!")


def test_job_names():
    """Test job name generation."""
    print("\n" + "=" * 60)
    print("Test: Job name generation")
    print("=" * 60)
    
    register_sweep_resolvers()
    
    cfg = OmegaConf.create({
        "sweep": {
            "penalty_weight": [-0.01, -0.03, -0.01],
            "learning_rate": [1e-4, 1e-4, 5e-3],
            "seed": [42, 42, 42],
        },
        "config_name": "pen${sz:penalty_weight}_lr${sz:learning_rate}_seed${sz:seed}",
    })
    
    for i in range(get_sweep_count(cfg)):
        name = get_sweep_job_name(cfg, i)
        print(f"  [{i}] {name}")
    
    print("\n✓ Job name generation passed!")


def test_validation():
    """Test sweep validation."""
    print("\n" + "=" * 60)
    print("Test: Sweep validation")
    print("=" * 60)
    
    register_sweep_resolvers()
    
    # Test valid config
    valid_cfg = OmegaConf.create({
        "sweep": {
            "lr": [0.001, 0.01],
            "seed": [42, 43],
        },
        "config_name": "lr${sz:lr}_seed${sz:seed}",
    })
    
    print("Testing valid config...")
    validate_sweep_config(valid_cfg)
    print("✓ Valid config accepted")
    
    # Test mismatched lengths
    invalid_cfg = OmegaConf.create({
        "sweep": {
            "lr": [0.001, 0.01, 0.1],  # 3 elements
            "seed": [42, 43],          # 2 elements
        },
        "config_name": "lr${sz:lr}_seed${sz:seed}",
    })
    
    print("\nTesting invalid config (mismatched lengths)...")
    try:
        validate_sweep_config(invalid_cfg)
        print("✗ Should have raised ValueError!")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    # Test duplicate names
    duplicate_cfg = OmegaConf.create({
        "sweep": {
            "lr": [0.001, 0.001],  # Same values = same names
            "seed": [42, 42],
        },
        "config_name": "lr${sz:lr}_seed${sz:seed}",
    })
    
    print("\nTesting invalid config (duplicate names)...")
    try:
        validate_sweep_config(duplicate_cfg)
        print("✗ Should have raised ValueError!")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    print("\n✓ Validation tests passed!")


def test_full_config():
    """Test with a realistic config structure."""
    print("\n" + "=" * 60)
    print("Test: Realistic config structure")
    print("=" * 60)
    
    register_sweep_resolvers()
    
    # Simulate your actual config structure
    cfg = OmegaConf.create({
        "sweep": {
            "penalty_weight": [-0.01, -0.03, -0.01, -0.03, -0.01, -0.01],
            "learning_rate": [1e-4, 1e-4, 5e-3, 5e-3, 1e-4, 1e-4],
            "seed": [42, 42, 42, 42, 43, 44],
        },
        "config_name": "monitor_informed_pen${sz:penalty_weight}_lr${sz:learning_rate}_seed${sz:seed}",
        "wandb": {
            "project": "obfuscation_generalization",
            "entity": "geodesic",
        },
        "model": {
            "base_model_id": "Qwen/Qwen3-4B",
        },
        "data": {
            "hf_dataset": "geodesic-puria/obf_gen_leave_out_sycophancy_full_xml_tags_seed_42",
        },
        "train": {
            "learning_rate": "${sz:learning_rate}",
            "seed": "${sz:seed}",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
        },
        "reward": {
            "funcs": {
                "correctness_reward_func": {},
                "api_overseer_penalty_func": {
                    "model_name": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
                    "penalty_weight": "${sz:penalty_weight}",
                    "max_tokens": 5,
                },
            },
        },
    })
    
    print("Validating sweep...")
    validate_sweep_config(cfg)
    
    print("\nResolved configs:")
    n_jobs = get_sweep_count(cfg)
    for i in range(n_jobs):
        set_sweep_index(i)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        
        print(f"\n[{i}] {resolved['config_name']}")
        print(f"    train.learning_rate: {resolved['train']['learning_rate']}")
        print(f"    train.seed: {resolved['train']['seed']}")
        print(f"    reward.funcs.api_overseer_penalty_func.penalty_weight: "
              f"{resolved['reward']['funcs']['api_overseer_penalty_func']['penalty_weight']}")
    
    print("\n✓ Realistic config test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SWEEP FUNCTIONALITY TESTS")
    print("=" * 60 + "\n")
    
    test_basic_sweep()
    test_job_names()
    test_validation()
    test_full_config()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
