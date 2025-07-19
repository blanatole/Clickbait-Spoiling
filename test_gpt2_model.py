#!/usr/bin/env python3
"""
Script to test and evaluate the trained GPT-2 spoiler model
"""

import torch
import json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scripts.evaluator import SpoilerEvaluator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2SpoilerTester:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = Path(model_path)
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded successfully")
    
    def generate_spoiler(self, input_text: str, max_length: int = 150, 
                        temperature: float = 0.7, top_k: int = 50, 
                        top_p: float = 0.95, num_return_sequences: int = 1) -> str:
        """Generate spoiler for given input text"""
        
        # Format input with separator
        formatted_input = f"{input_text} -> "
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors='pt',
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                early_stopping=True
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract spoiler (everything after " -> ")
        if " -> " in generated_text:
            spoiler = generated_text.split(" -> ", 1)[1]
            # Remove any trailing <|endoftext|> or similar tokens
            spoiler = spoiler.replace("<|endoftext|>", "").strip()
            return spoiler
        
        return generated_text
    
    def test_on_dataset(self, test_file: str, output_file: str = "test_predictions.txt", 
                       num_samples: int = None) -> tuple:
        """Test model on dataset and save predictions"""
        
        logger.info(f"Testing on {test_file}")
        
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        predictions = []
        references = []
        
        logger.info(f"Generating predictions for {len(test_data)} samples...")
        
        for i, item in enumerate(test_data):
            if i % 50 == 0:
                logger.info(f"Processing sample {i+1}/{len(test_data)}")
            
            input_text = item['input_text']
            reference = item['target_text']
            
            # Generate prediction
            prediction = self.generate_spoiler(input_text)
            
            predictions.append(prediction)
            references.append(reference)
            
            # Save sample results
            if i < 10:  # Show first 10 examples
                logger.info(f"Example {i+1}:")
                logger.info(f"Input: {input_text[:100]}...")
                logger.info(f"Reference: {reference}")
                logger.info(f"Prediction: {prediction}")
                logger.info("-" * 50)
        
        # Save predictions to file
        logger.info(f"Saving predictions to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        
        # Save references to file
        ref_file = output_file.replace('.txt', '_references.txt')
        with open(ref_file, 'w', encoding='utf-8') as f:
            for ref in references:
                f.write(f"{ref}\n")
        
        return predictions, references
    
    def evaluate_model(self, test_file: str, output_dir: str = "evaluation_results"):
        """Full evaluation pipeline"""
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate predictions
        predictions, references = self.test_on_dataset(
            test_file, 
            output_file=str(output_dir / "predictions.txt")
        )
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = SpoilerEvaluator()
        
        # Run comprehensive evaluation
        logger.info("Running comprehensive evaluation...")
        results = evaluator.evaluate_comprehensive(references, predictions)
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        evaluator.save_results(results, str(results_file))
        
        # Print results
        print(evaluator.format_results(results))
        
        return results

def main():
    """Main testing function"""
    
    # Configuration
    model_path = "models/gpt2-spoiler/final_model"
    test_file = "processed_data/test_generation.jsonl"
    output_dir = "evaluation_results"
    
    # Test specific samples (set to None for full test)
    num_samples = 100  # Test on first 100 samples for quick evaluation
    
    # Create tester
    tester = GPT2SpoilerTester(model_path)
    
    # Test interactive generation
    print("=== Interactive Testing ===")
    sample_inputs = [
        "This clickbait article claims that celebrities are doing something shocking",
        "You won't believe what happened when scientists discovered this",
        "The secret ingredient that chefs don't want you to know about"
    ]
    
    for i, input_text in enumerate(sample_inputs, 1):
        print(f"\nExample {i}:")
        print(f"Input: {input_text}")
        spoiler = tester.generate_spoiler(input_text)
        print(f"Generated Spoiler: {spoiler}")
    
    # Full evaluation
    print("\n=== Full Evaluation ===")
    results = tester.evaluate_model(test_file, output_dir)
    
    # Print key metrics
    print("\n=== Key Metrics Summary ===")
    print(f"BLEU-4: {results['bleu_scores']['bleu-4']:.4f}")
    print(f"ROUGE-L: {results['rouge_scores']['rouge-L']:.4f}")
    print(f"METEOR: {results['meteor_score']:.4f}")
    print(f"BERTScore F1: {results['bert_scores']['bert-f1']:.4f}")
    print(f"Semantic Similarity: {results['semantic_similarity']:.4f}")

if __name__ == "__main__":
    main()
