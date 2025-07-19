#!/usr/bin/env python3
"""
Script to download and setup NLTK data for clickbait spoiling project
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data"""
    
    print("Downloading NLTK data...")
    
    # List of NLTK data to download
    nltk_data = [
        'punkt',           # For tokenization
        'stopwords',       # For stop words
        'wordnet',         # For lemmatization
        'averaged_perceptron_tagger',  # For POS tagging
        'vader_lexicon',   # For sentiment analysis
        'omw-1.4',         # For wordnet
        'punkt_tab'        # For sentence tokenization
    ]
    
    for data in nltk_data:
        try:
            print(f"Downloading {data}...")
            nltk.download(data, quiet=True)
            print(f"✓ Successfully downloaded {data}")
        except Exception as e:
            print(f"✗ Failed to download {data}: {e}")
    
    print("\nNLTK data setup complete!")

def test_imports():
    """Test if all major libraries can be imported"""
    
    print("\nTesting imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        import sentence_transformers
        print(f"✓ Sentence Transformers version: {sentence_transformers.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
        
        import spacy
        print(f"✓ SpaCy version: {spacy.__version__}")
        
        # Test spaCy model
        nlp = spacy.load('en_core_web_sm')
        print("✓ SpaCy English model loaded successfully")
        
        import nltk
        print(f"✓ NLTK version: {nltk.__version__}")
        
        # Test evaluation libraries
        import sacrebleu
        print(f"✓ SacreBLEU version: {sacrebleu.__version__}")
        
        import rouge_score
        print("✓ ROUGE score library imported")
        
        import bert_score
        print("✓ BERTScore library imported")
        
        # Test data processing libraries
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        print("\n✅ All imports successful!")
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_nltk_data()
    test_imports()
