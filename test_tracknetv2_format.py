#!/usr/bin/env python3
"""
Test script to verify TrackNetV2 format dataset loading
"""

import sys
import os
sys.path.append('src')

from config.config import parse_configs
import argparse

def test_tracknetv2_format():
    # Override sys.argv to simulate command line arguments
    sys.argv = [
        'test_tracknetv2_format.py',
        '--dataset_choice', 'tracknetv2',
        '--dataset_dir', '../data/badminton/TrackNetV2',  # Custom path using unified option
        '--model_choice', 'tracknet',
        '--num_frames', '9',
        '--batch_size', '2',
        '--num_samples', '10',  # Load only 10 samples for testing
        '--no_val'
    ]
    
    # Parse configurations
    configs = parse_configs()
    
    print(f"Dataset choice: {configs.dataset_choice}")
    print(f"TrackNetV2 dataset directory: {configs.tracknetv2_dataset_dir}")
    
    # Import data utils after configs are set
    from data_process.data_utils import get_all_detection_infor_tracknetv2
    
    # Test loading training data
    print("\n=== Testing TrackNetV2 format loader ===")
    print("Loading training data...")
    train_events_infor, train_events_labels = get_all_detection_infor_tracknetv2('train', configs)
    
    print(f"Number of training sequences loaded: {len(train_events_infor)}")
    if len(train_events_infor) > 0:
        print(f"First sequence has {len(train_events_infor[0])} frames")
        print(f"First frame path: {train_events_infor[0][0]}")
        print(f"First label (ball_position, visibility, status): {train_events_labels[0]}")
    
    # Test loading test data
    print("\nLoading test data...")
    test_events_infor, test_events_labels = get_all_detection_infor_tracknetv2('test', configs)
    
    print(f"Number of test sequences loaded: {len(test_events_infor)}")
    if len(test_events_infor) > 0:
        print(f"First sequence has {len(test_events_infor[0])} frames")
        print(f"First frame path: {test_events_infor[0][0]}")
        print(f"First label (ball_position, visibility, status): {test_events_labels[0]}")
    
    # Test with dataloader
    print("\n=== Testing with DataLoader ===")
    from data_process.dataloader import create_occlusion_test_dataloader
    
    test_dataloader = create_occlusion_test_dataloader(configs)
    print(f"Test dataloader created with {len(test_dataloader.dataset)} samples")
    
    # Load one batch
    for i, (imgs, labels) in enumerate(test_dataloader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {imgs.shape}")
        print(f"  Labels: {labels}")
        if i == 0:  # Only show first batch
            break
    
    print("\n✓ TrackNetV2 format loading test completed successfully!")
    print("\n=== Summary ===")
    print(f"- Dataset choice: {configs.dataset_choice}")
    print(f"- TrackNetV2 directory: {configs.tracknetv2_dataset_dir}")
    print(f"- Training sequences: {len(train_events_infor) if train_events_infor else 0}")
    print(f"- Test sequences: {len(test_events_infor) if test_events_infor else 0}")
    print(f"- Usage examples:")
    print(f"  Default: --dataset_choice tracknetv2")
    print(f"  Custom:  --dataset_choice tracknetv2 --dataset_dir /path/to/data")
    print(f"  Other:   --dataset_choice badminton --dataset_dir /path/to/badminton")

def test_unified_dataset_dir():
    """Test that --dataset_dir works for all dataset choices"""
    print("\n=== Testing Unified --dataset_dir Option ===")
    
    test_cases = [
        ('tt', '/custom/tt/path'),
        ('tennis', '/custom/tennis/path'),
        ('badminton', '/custom/badminton/path'),
        ('tta', '/custom/tta/path'),
        ('tracknetv2', '/custom/tracknetv2/path')
    ]
    
    for dataset_choice, custom_path in test_cases:
        # Override sys.argv for each test case
        sys.argv = [
            'test_tracknetv2_format.py',
            '--dataset_choice', dataset_choice,
            '--dataset_dir', custom_path,
            '--model_choice', 'tracknet',
            '--num_frames', '9',
            '--no_val'
        ]
        
        # Parse configurations
        configs = parse_configs()
        
        # Check if the custom path was applied correctly
        if dataset_choice == 'tt':
            actual_path = configs.dataset_dir
        elif dataset_choice == 'tennis':
            actual_path = configs.tennis_dataset_dir
        elif dataset_choice == 'badminton':
            actual_path = configs.badminton_dataset_dir
        elif dataset_choice == 'tta':
            actual_path = configs.tta_dataset_dir
        elif dataset_choice == 'tracknetv2':
            actual_path = configs.tracknetv2_dataset_dir
            
        if actual_path == custom_path:
            print(f"✓ {dataset_choice}: Custom path applied correctly -> {actual_path}")
        else:
            print(f"✗ {dataset_choice}: Expected {custom_path}, got {actual_path}")
    
    print("Unified --dataset_dir option test completed!")

if __name__ == "__main__":
    test_tracknetv2_format()
    test_unified_dataset_dir()