#!/usr/bin/env python3
"""
Main script to preprocess clickbait data for both tasks
"""

import argparse
import json
import pickle
from pathlib import Path
import logging
from scripts.data_preprocessor import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Preprocess clickbait data')
    parser.add_argument('--task', choices=['generation', 'classification', 'both'], 
                       default='both', help='Task type to preprocess for')
    parser.add_argument('--input_dir', default='data', help='Input data directory')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                       help='Data splits to process')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Process each split
    for split in args.splits:
        logger.info(f"Processing {split} split...")
        
        # Load data
        input_file = Path(args.input_dir) / f"{split}.jsonl"
        if not input_file.exists():
            logger.warning(f"File {input_file} not found, skipping...")
            continue
        
        data = preprocessor.load_jsonl(input_file)
        logger.info(f"Loaded {len(data)} samples from {input_file}")
        
        # Process for spoiler generation
        if args.task in ['generation', 'both']:
            logger.info("Processing for spoiler generation task...")
            gen_data = preprocessor.prepare_spoiler_generation_data(data)
            
            # Save generation data
            gen_output_file = output_dir / f"{split}_generation.jsonl"
            preprocessor.save_processed_data(gen_data, gen_output_file, 'generation')
            
            # Print statistics
            gen_stats = preprocessor.get_data_statistics(gen_data)
            logger.info(f"Generation data statistics for {split}:")
            logger.info(f"  Total samples: {gen_stats['total_samples']}")
            logger.info(f"  Spoiler types: {gen_stats['spoiler_types']}")
            logger.info(f"  Platforms: {gen_stats['platforms']}")
            logger.info(f"  Avg text length: {gen_stats.get('avg_text_length', 0):.1f}")
        
        # Process for spoiler type classification
        if args.task in ['classification', 'both']:
            logger.info("Processing for spoiler type classification task...")
            class_data = preprocessor.prepare_spoiler_classification_data(data)
            
            # Save classification data
            class_output_file = output_dir / f"{split}_classification.pkl"
            preprocessor.save_processed_data(class_data, class_output_file, 'classification')
            
            # Print statistics
            class_stats = preprocessor.get_data_statistics(class_data)
            logger.info(f"Classification data statistics for {split}:")
            logger.info(f"  Total samples: {class_stats['total_samples']}")
            logger.info(f"  Spoiler types: {class_stats['spoiler_types']}")
            logger.info(f"  Platforms: {class_stats['platforms']}")
            logger.info(f"  Feature dimension: {class_data[0]['features'].shape[0] if class_data else 0}")
    
    # Create summary report
    create_preprocessing_report(output_dir, args.task)
    
    logger.info("Preprocessing completed successfully!")

def create_preprocessing_report(output_dir: Path, task_type: str):
    """Create a preprocessing report"""
    report_file = output_dir / "preprocessing_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Data Preprocessing Report\n\n")
        f.write(f"**Task Type:** {task_type}\n")
        f.write(f"**Output Directory:** {output_dir}\n\n")
        
        f.write("## Processing Steps\n\n")
        
        if task_type in ['generation', 'both']:
            f.write("### Spoiler Generation Task\n")
            f.write("1. **Input Combination**: Combined `postText` and `targetParagraphs` with [SEP] token\n")
            f.write("2. **Text Cleaning**: Removed HTML, emojis, special characters\n")
            f.write("3. **Tokenization**: Used GPT-2 Byte-Pair Tokenizer\n")
            f.write("4. **Output**: Tokenized input-target pairs for sequence-to-sequence training\n\n")
        
        if task_type in ['classification', 'both']:
            f.write("### Spoiler Type Classification Task\n")
            f.write("1. **Text Normalization**: Cleaned 5 text fields\n")
            f.write("2. **Embedding Generation**: Used SBERT (all-MiniLM-L6-v2) for 384-dim embeddings\n")
            f.write("3. **Feature Concatenation**: Combined embeddings → 1920-dim feature vector\n")
            f.write("4. **Output**: Feature vectors with spoiler type labels\n\n")
        
        f.write("## Files Generated\n\n")
        
        for file in output_dir.glob("*"):
            if file.is_file():
                f.write(f"- `{file.name}`: {file.stat().st_size:,} bytes\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Load processed data for model training\n")
        f.write("2. Implement baseline models\n")
        f.write("3. Train and evaluate models\n")
        f.write("4. Hyperparameter tuning\n")

if __name__ == "__main__":
    main()
