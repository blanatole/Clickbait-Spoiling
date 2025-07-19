#!/usr/bin/env python3
"""
Script to explore and understand the clickbait spoiling dataset
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def explore_dataset():
    """Explore the clickbait spoiling dataset"""
    
    print("🔍 Exploring Clickbait Spoiling Dataset")
    print("=" * 50)
    
    # Load datasets
    data_dir = Path("data")
    
    datasets = {}
    for split in ['train', 'validation', 'test']:
        file_path = data_dir / f"{split}.jsonl"
        if file_path.exists():
            datasets[split] = load_jsonl(file_path)
            print(f"✓ Loaded {split} set: {len(datasets[split])} samples")
        else:
            print(f"✗ {split} set not found")
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Analyze the structure
    print("\n📊 Dataset Statistics:")
    for split, data in datasets.items():
        print(f"\n{split.upper()} SET:")
        print(f"  Total samples: {len(data)}")
        
        # Analyze spoiler types
        spoiler_tags = [item.get('tags', []) for item in data]
        flat_tags = [tag for tags in spoiler_tags for tag in tags if tags]
        tag_counts = Counter(flat_tags)
        print(f"  Spoiler types: {dict(tag_counts)}")
        
        # Analyze platforms
        platforms = [item.get('postPlatform', 'Unknown') for item in data]
        platform_counts = Counter(platforms)
        print(f"  Platforms: {dict(platform_counts)}")
        
        # Analyze spoiler lengths
        spoiler_lengths = []
        for item in data:
            spoilers = item.get('spoiler', [])
            if spoilers:
                total_length = sum(len(spoiler) for spoiler in spoilers)
                spoiler_lengths.append(total_length)
        
        if spoiler_lengths:
            print(f"  Spoiler length stats:")
            print(f"    Mean: {np.mean(spoiler_lengths):.1f}")
            print(f"    Median: {np.median(spoiler_lengths):.1f}")
            print(f"    Min: {min(spoiler_lengths)}")
            print(f"    Max: {max(spoiler_lengths)}")
    
    # Display sample data
    print("\n📝 Sample Data Structure:")
    if 'train' in datasets:
        sample = datasets['train'][0]
        print(f"Post Text: {sample.get('postText', 'N/A')}")
        print(f"Spoiler: {sample.get('spoiler', 'N/A')}")
        print(f"Tags: {sample.get('tags', 'N/A')}")
        print(f"Platform: {sample.get('postPlatform', 'N/A')}")
        print(f"Target Title: {sample.get('targetTitle', 'N/A')[:100]}...")
        
    # Create visualizations
    create_visualizations(datasets)

def create_visualizations(datasets):
    """Create visualizations for the dataset"""
    
    print("\n📈 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clickbait Spoiling Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spoiler types distribution
    all_tags = []
    for split, data in datasets.items():
        for item in data:
            tags = item.get('tags', [])
            all_tags.extend(tags)
    
    tag_counts = Counter(all_tags)
    
    axes[0, 0].bar(tag_counts.keys(), tag_counts.values(), color='skyblue')
    axes[0, 0].set_title('Spoiler Types Distribution')
    axes[0, 0].set_xlabel('Spoiler Type')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Platform distribution
    all_platforms = []
    for split, data in datasets.items():
        for item in data:
            platform = item.get('postPlatform', 'Unknown')
            all_platforms.append(platform)
    
    platform_counts = Counter(all_platforms)
    
    axes[0, 1].pie(platform_counts.values(), labels=platform_counts.keys(), 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Platform Distribution')
    
    # 3. Spoiler length distribution
    spoiler_lengths = []
    for split, data in datasets.items():
        for item in data:
            spoilers = item.get('spoiler', [])
            if spoilers:
                total_length = sum(len(spoiler) for spoiler in spoilers)
                spoiler_lengths.append(total_length)
    
    axes[1, 0].hist(spoiler_lengths, bins=30, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Spoiler Length Distribution')
    axes[1, 0].set_xlabel('Character Count')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Dataset size comparison
    dataset_sizes = {split: len(data) for split, data in datasets.items()}
    
    axes[1, 1].bar(dataset_sizes.keys(), dataset_sizes.values(), color='lightgreen')
    axes[1, 1].set_title('Dataset Sizes')
    axes[1, 1].set_xlabel('Dataset Split')
    axes[1, 1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to 'data_analysis.png'")
    
    # Show if in interactive mode
    try:
        plt.show()
    except:
        pass

def analyze_spoiler_types():
    """Analyze different spoiler types in detail"""
    
    print("\n🔍 Detailed Spoiler Type Analysis:")
    print("=" * 40)
    
    # Load training data
    train_data = load_jsonl("data/train.jsonl")
    
    # Group by spoiler type
    spoiler_types = {}
    for item in train_data:
        tags = item.get('tags', [])
        if tags:
            tag = tags[0]  # Take first tag
            if tag not in spoiler_types:
                spoiler_types[tag] = []
            spoiler_types[tag].append(item)
    
    for spoiler_type, items in spoiler_types.items():
        print(f"\n{spoiler_type.upper()} SPOILERS ({len(items)} samples):")
        
        # Sample examples
        print("  Examples:")
        for i, item in enumerate(items[:3]):  # Show first 3 examples
            post_text = item.get('postText', [''])[0]
            spoiler = item.get('spoiler', [''])[0]
            print(f"    {i+1}. Post: {post_text}")
            print(f"       Spoiler: {spoiler}")
            print()

if __name__ == "__main__":
    explore_dataset()
    analyze_spoiler_types()
