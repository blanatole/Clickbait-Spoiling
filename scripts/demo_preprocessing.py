#!/usr/bin/env python3
"""
Demo script to test data preprocessing functionality
"""

import json
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_preprocessor import DataPreprocessor

def demo_preprocessing():
    """Demo the preprocessing functionality"""
    
    print("🔧 Data Preprocessing Demo")
    print("=" * 50)
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = DataPreprocessor()
    print("✅ Preprocessor initialized successfully")
    
    # Load a small sample of data
    print("\n📊 Loading sample data...")
    train_data = preprocessor.load_jsonl("data/train.jsonl")
    sample_data = train_data[:5]  # Take first 5 samples
    print(f"✅ Loaded {len(sample_data)} samples for demo")
    
    # Demo text cleaning
    print("\n🧹 Text Cleaning Demo:")
    sample_text = "This is a <b>HTML</b> text with 😀 emojis and special chars ™"
    cleaned_text = preprocessor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned:  {cleaned_text}")
    
    # Demo input combination
    print("\n🔗 Input Combination Demo:")
    post_text = ["This is a clickbait title"]
    target_paragraphs = ["This is the full article content that explains the clickbait."]
    combined = preprocessor.combine_post_and_target(post_text, target_paragraphs)
    print(f"Post: {post_text}")
    print(f"Target: {target_paragraphs}")
    print(f"Combined: {combined}")
    
    # Demo GPT-2 tokenization
    print("\n🔤 GPT-2 Tokenization Demo:")
    tokenized = preprocessor.tokenize_with_gpt2(combined)
    print(f"Token count: {tokenized['token_count']}")
    print(f"First 10 tokens: {tokenized['input_ids'][:10].tolist()}")
    
    # Demo SBERT embeddings
    print("\n🧠 SBERT Embedding Demo:")
    sample_texts = ["This is a test sentence", "Another example text"]
    embeddings = preprocessor.create_sbert_embeddings(sample_texts)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Demo spoiler generation preprocessing
    print("\n🎯 Spoiler Generation Preprocessing Demo:")
    gen_data = preprocessor.prepare_spoiler_generation_data(sample_data)
    print(f"Processed {len(gen_data)} samples")
    
    # Show sample
    sample_gen = gen_data[0]
    print("\nSample processed data:")
    print(f"  UUID: {sample_gen['uuid']}")
    print(f"  Input (first 100 chars): {sample_gen['input_text'][:100]}...")
    print(f"  Target: {sample_gen['target_text']}")
    print(f"  Spoiler type: {sample_gen['spoiler_type']}")
    print(f"  Platform: {sample_gen['platform']}")
    print(f"  Input tokens: {sample_gen['input_tokens']['token_count']}")
    print(f"  Target tokens: {sample_gen['target_tokens']['token_count']}")
    
    # Demo spoiler classification preprocessing
    print("\n📊 Spoiler Classification Preprocessing Demo:")
    class_data = preprocessor.prepare_spoiler_classification_data(sample_data)
    print(f"Processed {len(class_data)} samples")
    
    # Show sample
    sample_class = class_data[0]
    print("\nSample processed data:")
    print(f"  UUID: {sample_class['uuid']}")
    print(f"  Feature dimension: {sample_class['features'].shape}")
    print(f"  Spoiler type: {sample_class['spoiler_type']}")
    print(f"  Platform: {sample_class['platform']}")
    print(f"  Post text (first 100 chars): {sample_class['post_text'][:100]}...")
    print(f"  Individual embeddings available: {list(sample_class['individual_embeddings'].keys())}")
    
    # Show statistics
    print("\n📈 Data Statistics:")
    gen_stats = preprocessor.get_data_statistics(gen_data)
    print(f"Generation data:")
    print(f"  Total samples: {gen_stats['total_samples']}")
    print(f"  Spoiler types: {gen_stats['spoiler_types']}")
    print(f"  Platforms: {gen_stats['platforms']}")
    print(f"  Avg text length: {gen_stats.get('avg_text_length', 0):.1f}")
    
    class_stats = preprocessor.get_data_statistics(class_data)
    print(f"\nClassification data:")
    print(f"  Total samples: {class_stats['total_samples']}")
    print(f"  Spoiler types: {class_stats['spoiler_types']}")
    print(f"  Platforms: {class_stats['platforms']}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run full preprocessing:")
    print("  python preprocess_data.py --task both")

if __name__ == "__main__":
    demo_preprocessing()
