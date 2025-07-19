"""
Configuration file for Clickbait Spoiling Project
Contains all project settings, paths, and hyperparameters
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Raw data files
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Processed data files
TRAIN_GENERATION_FILE = PROCESSED_DATA_DIR / "train_generation.jsonl"
VALIDATION_GENERATION_FILE = PROCESSED_DATA_DIR / "validation_generation.jsonl"
TEST_GENERATION_FILE = PROCESSED_DATA_DIR / "test_generation.jsonl"

TRAIN_CLASSIFICATION_FILE = PROCESSED_DATA_DIR / "train_classification.pkl"
VALIDATION_CLASSIFICATION_FILE = PROCESSED_DATA_DIR / "validation_classification.pkl"
TEST_CLASSIFICATION_FILE = PROCESSED_DATA_DIR / "test_classification.pkl"

# Model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "max_length": 512,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "warmup_steps": 500,
        "save_steps": 1000,
        "eval_steps": 500,
        "output_dir": MODELS_DIR / "gpt2_spoiler"
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "max_length": 256,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_steps": 100,
        "save_steps": 500,
        "eval_steps": 250,
        "output_dir": MODELS_DIR / "bert_classifier"
    },
    "t5": {
        "model_name": "t5-small",
        "max_length": 512,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "warmup_steps": 500,
        "save_steps": 1000,
        "eval_steps": 500,
        "output_dir": MODELS_DIR / "t5_generator"
    }
}

# Preprocessing settings
PREPROCESSING_CONFIG = {
    "max_input_length": 512,
    "max_target_length": 128,
    "min_spoiler_length": 5,
    "max_spoiler_length": 500,
    "remove_html": True,
    "remove_emojis": True,
    "normalize_whitespace": True,
    "lowercase": False
}

# Evaluation settings
EVALUATION_CONFIG = {
    "metrics": ["bleu", "rouge", "bertscore", "meteor"],
    "bleu_weights": [0.25, 0.25, 0.25, 0.25],  # 4-gram BLEU
    "rouge_types": ["rouge1", "rouge2", "rougeL"],
    "bertscore_model": "bert-base-uncased",
    "sample_size": None,  # None for full evaluation
    "save_predictions": True,
    "save_metrics": True
}

# Training settings
TRAINING_CONFIG = {
    "seed": 42,
    "fp16": True,  # Mixed precision training
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "logging_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False
}

# Generation settings
GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "min_length": 10,
    "do_sample": True,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "num_return_sequences": 1,
    "pad_token_id": 50256,  # GPT-2 pad token
    "eos_token_id": 50256   # GPT-2 end token
}

# Dataset statistics (from exploration)
DATASET_STATS = {
    "total_samples": 5000,
    "train_samples": 3200,
    "validation_samples": 800,
    "test_samples": 1000,
    "spoiler_types": {
        "phrase": 1367,
        "passage": 1274,
        "multi": 559
    },
    "platforms": {
        "twitter": 1530,
        "reddit": 1150,
        "facebook": 520
    },
    "avg_spoiler_length": 84.6,
    "median_spoiler_length": 43.0
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": PROJECT_ROOT / "logs" / "project.log"
}

# Environment settings
ENVIRONMENT = {
    "cuda_available": True,  # Will be checked at runtime
    "device": "auto",  # auto, cpu, cuda
    "num_workers": 4,  # For data loading
    "pin_memory": True
}

# Special tokens
SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "sep_token": "<sep>",
    "spoiler_token": "<spoiler>"
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project"""
    directories = [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        EVALUATION_DIR,
        NOTEBOOKS_DIR,
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Create model subdirectories
    for model_name in MODEL_CONFIGS:
        model_dir = MODELS_DIR / f"{model_name}_spoiler"
        model_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
