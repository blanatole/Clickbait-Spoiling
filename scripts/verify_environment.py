#!/usr/bin/env python3
"""
Environment verification script for clickbait spoiling project
"""

import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("   ✅ Python version is compatible (>=3.7)")
        return True
    else:
        print("   ❌ Python version is too old (requires >=3.7)")
        return False

def check_conda_environment():
    """Check if conda environment is activated"""
    print("\n🔧 Conda Environment Check:")
    conda_env = subprocess.run(['conda', 'info', '--envs'], 
                               capture_output=True, text=True)
    
    if conda_env.returncode == 0:
        current_env = subprocess.run(['conda', 'info', '--json'], 
                                    capture_output=True, text=True)
        if 'clickbait' in current_env.stdout:
            print("   ✅ Conda environment 'clickbait' is active")
            return True
        else:
            print("   ⚠️  Conda environment may not be 'clickbait'")
            return False
    else:
        print("   ❌ Conda not available")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    print("\n📦 Package Installation Check:")
    
    required_packages = [
        'torch',
        'transformers',
        'sentence-transformers',
        'scikit-learn',
        'nltk',
        'spacy',
        'sacrebleu',
        'rouge-score',
        'bert-score',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'jupyter'
    ]
    
    installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
    
    all_installed = True
    for package in required_packages:
        # Handle package name variations
        package_variants = [package, package.replace('-', '_')]
        
        if any(variant in installed_packages for variant in package_variants):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_spacy_model():
    """Check if spaCy English model is installed"""
    print("\n🔤 SpaCy Model Check:")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("   ✅ SpaCy English model (en_core_web_sm) is installed")
        return True
    except OSError:
        print("   ❌ SpaCy English model (en_core_web_sm) is NOT installed")
        print("   Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"   ❌ Error loading SpaCy model: {e}")
        return False

def check_nltk_data():
    """Check if NLTK data is downloaded"""
    print("\n📚 NLTK Data Check:")
    
    try:
        import nltk
        
        required_data = [
            'tokenizers/punkt',
            'corpora/stopwords',
            'corpora/wordnet',
            'taggers/averaged_perceptron_tagger',
            'vader_lexicon'
        ]
        
        all_available = True
        for data_name in required_data:
            try:
                nltk.data.find(data_name)
                print(f"   ✅ {data_name.split('/')[-1]}")
            except LookupError:
                print(f"   ❌ {data_name.split('/')[-1]} - NOT DOWNLOADED")
                all_available = False
        
        return all_available
        
    except ImportError:
        print("   ❌ NLTK not available")
        return False

def check_cuda_support():
    """Check CUDA support"""
    print("\n🚀 CUDA Support Check:")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA is available")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   ⚠️  CUDA not available (will use CPU)")
        return True
    except ImportError:
        print("   ❌ PyTorch not available")
        return False

def check_data_files():
    """Check if data files exist"""
    print("\n📁 Data Files Check:")
    
    data_dir = Path("data")
    required_files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
    
    all_exist = True
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   ✅ {file_name} ({size:,} bytes)")
        else:
            print(f"   ❌ {file_name} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Main verification function"""
    print("🔍 Clickbait Spoiling Environment Verification")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_conda_environment(),
        check_required_packages(),
        check_spacy_model(),
        check_nltk_data(),
        check_cuda_support(),
        check_data_files()
    ]
    
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY:")
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"   ✅ All checks passed ({passed}/{total})")
        print("   🎉 Environment is ready for development!")
    else:
        print(f"   ⚠️  {passed}/{total} checks passed")
        print("   🔧 Please fix the issues above before proceeding")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
