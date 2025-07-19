#!/usr/bin/env python3
"""
Script to verify and test processed data
"""

import json
import pickle
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_generation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load generation data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_classification_data(file_path: str) -> List[Dict[str, Any]]:
    """Load classification data from pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def verify_generation_data(data: List[Dict[str, Any]], split_name: str):
    """Verify generation data structure and content"""
    print(f"\n🔍 Verifying {split_name} Generation Data:")
    print(f"  Total samples: {len(data)}")
    
    # Check required fields
    required_fields = ['uuid', 'input_text', 'target_text', 'input_tokens', 'target_tokens', 
                      'spoiler_type', 'platform']
    
    sample = data[0]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f"  ❌ Missing fields: {missing_fields}")
        return False
    else:
        print(f"  ✅ All required fields present")
    
    # Check token structures
    input_tokens = sample['input_tokens']
    target_tokens = sample['target_tokens']
    
    if 'input_ids' in input_tokens and 'attention_mask' in input_tokens:
        print(f"  ✅ Input tokens structure valid")
        print(f"    Input sequence length: {len(input_tokens['input_ids'])}")
    else:
        print(f"  ❌ Invalid input token structure")
        return False
    
    if 'input_ids' in target_tokens and 'attention_mask' in target_tokens:
        print(f"  ✅ Target tokens structure valid")
        print(f"    Target sequence length: {len(target_tokens['input_ids'])}")
    else:
        print(f"  ❌ Invalid target token structure")
        return False
    
    # Check spoiler types distribution
    spoiler_types = {}
    platforms = {}
    text_lengths = []
    
    for item in data:
        # Count spoiler types
        spoiler_type = item.get('spoiler_type', 'unknown')
        spoiler_types[spoiler_type] = spoiler_types.get(spoiler_type, 0) + 1
        
        # Count platforms
        platform = item.get('platform', 'unknown')
        platforms[platform] = platforms.get(platform, 0) + 1
        
        # Text lengths
        text_lengths.append(len(item['input_text']))
    
    print(f"  Spoiler types: {spoiler_types}")
    print(f"  Platforms: {platforms}")
    print(f"  Text length - Mean: {np.mean(text_lengths):.1f}, Max: {max(text_lengths)}, Min: {min(text_lengths)}")
    
    # Show sample
    print(f"\n  Sample data:")
    print(f"    Input (first 100 chars): {sample['input_text'][:100]}...")
    print(f"    Target: {sample['target_text']}")
    print(f"    Spoiler type: {sample['spoiler_type']}")
    print(f"    Platform: {sample['platform']}")
    
    return True

def verify_classification_data(data: List[Dict[str, Any]], split_name: str):
    """Verify classification data structure and content"""
    print(f"\n🔍 Verifying {split_name} Classification Data:")
    print(f"  Total samples: {len(data)}")
    
    # Check required fields
    required_fields = ['uuid', 'features', 'spoiler_type', 'platform', 'individual_embeddings']
    
    sample = data[0]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f"  ❌ Missing fields: {missing_fields}")
        return False
    else:
        print(f"  ✅ All required fields present")
    
    # Check feature dimensions
    features = sample['features']
    if isinstance(features, np.ndarray) and features.shape == (1920,):
        print(f"  ✅ Feature vector shape is correct: {features.shape}")
    else:
        print(f"  ❌ Invalid feature vector shape: {features.shape if hasattr(features, 'shape') else type(features)}")
        return False
    
    # Check individual embeddings
    individual_embeddings = sample['individual_embeddings']
    expected_keys = ['post', 'title', 'paragraph', 'description', 'keywords']
    
    if all(key in individual_embeddings for key in expected_keys):
        print(f"  ✅ Individual embeddings structure valid")
        
        # Check dimensions
        for key in expected_keys:
            emb = individual_embeddings[key]
            if isinstance(emb, np.ndarray) and emb.shape == (384,):
                print(f"    {key}: {emb.shape} ✅")
            else:
                print(f"    {key}: {emb.shape if hasattr(emb, 'shape') else type(emb)} ❌")
                return False
    else:
        print(f"  ❌ Invalid individual embeddings structure")
        return False
    
    # Check spoiler types distribution
    spoiler_types = {}
    platforms = {}
    
    for item in data:
        # Count spoiler types
        spoiler_type = item.get('spoiler_type', 'unknown')
        spoiler_types[spoiler_type] = spoiler_types.get(spoiler_type, 0) + 1
        
        # Count platforms
        platform = item.get('platform', 'unknown')
        platforms[platform] = platforms.get(platform, 0) + 1
    
    print(f"  Spoiler types: {spoiler_types}")
    print(f"  Platforms: {platforms}")
    
    # Show sample
    print(f"\n  Sample data:")
    print(f"    UUID: {sample['uuid']}")
    print(f"    Feature shape: {sample['features'].shape}")
    print(f"    Spoiler type: {sample['spoiler_type']}")
    print(f"    Platform: {sample['platform']}")
    print(f"    Post text (first 100 chars): {sample.get('post_text', '')[:100]}...")
    
    return True

def main():
    """Main verification function"""
    
    print("🔍 Processed Data Verification")
    print("=" * 50)
    
    processed_dir = Path("processed_data")
    if not processed_dir.exists():
        print("❌ processed_data directory not found!")
        return
    
    splits = ['train', 'validation', 'test']
    all_passed = True
    
    for split in splits:
        # Verify generation data
        gen_file = processed_dir / f"{split}_generation.jsonl"
        if gen_file.exists():
            try:
                gen_data = load_generation_data(gen_file)
                if not verify_generation_data(gen_data, split):
                    all_passed = False
            except Exception as e:
                print(f"❌ Error loading {split} generation data: {e}")
                all_passed = False
        else:
            print(f"❌ {split} generation file not found")
            all_passed = False
        
        # Verify classification data
        class_file = processed_dir / f"{split}_classification.pkl"
        if class_file.exists():
            try:
                class_data = load_classification_data(class_file)
                if not verify_classification_data(class_data, split):
                    all_passed = False
            except Exception as e:
                print(f"❌ Error loading {split} classification data: {e}")
                all_passed = False
        else:
            print(f"❌ {split} classification file not found")
            all_passed = False
    
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY:")
    
    if all_passed:
        print("✅ All data verification passed!")
        print("🎉 Data is ready for model training!")
    else:
        print("❌ Some verification failed!")
        print("🔧 Please check the issues above")
    
    # File size summary
    print("\n📊 File Sizes:")
    for file in sorted(processed_dir.glob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")
    
    print("\n🚀 Next Steps:")
    print("1. Load processed data in model training scripts")
    print("2. Implement baseline models")
    print("3. Train and evaluate models")
    print("4. Compare different approaches")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
