"""
Evaluator module for clickbait spoiling task
"""

import json
import numpy as np
from typing import List, Dict, Union, Tuple
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

class SpoilerEvaluator:
    """Evaluator for spoiler generation task"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', 
                                     num_layers=40, 
                                     lang='en', 
                                     device=device)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.smoothing_function = SmoothingFunction().method1
        
    def calculate_bleu_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores"""
        assert len(references) == len(candidates), "References and candidates must have same length"
        
        bleu_scores = {'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}
        
        for ref, cand in zip(references, candidates):
            ref_tokens = [ref.split()]
            cand_tokens = cand.split()
            
            # Calculate BLEU scores for different n-grams
            try:
                bleu_1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), 
                                     smoothing_function=self.smoothing_function)
                bleu_2 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), 
                                     smoothing_function=self.smoothing_function)
                bleu_3 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.33, 0.33, 0.33, 0), 
                                     smoothing_function=self.smoothing_function)
                bleu_4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), 
                                     smoothing_function=self.smoothing_function)
                
                bleu_scores['bleu-1'].append(bleu_1)
                bleu_scores['bleu-2'].append(bleu_2)
                bleu_scores['bleu-3'].append(bleu_3)
                bleu_scores['bleu-4'].append(bleu_4)
            except:
                bleu_scores['bleu-1'].append(0.0)
                bleu_scores['bleu-2'].append(0.0)
                bleu_scores['bleu-3'].append(0.0)
                bleu_scores['bleu-4'].append(0.0)
        
        # Calculate average scores
        avg_bleu_scores = {k: np.mean(v) for k, v in bleu_scores.items()}
        return avg_bleu_scores
    
    def calculate_rouge_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        assert len(references) == len(candidates), "References and candidates must have same length"
        
        rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-L': []}
        
        for ref, cand in zip(references, candidates):
            scores = self.rouge_scorer.score(ref, cand)
            rouge_scores['rouge-1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge-2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rouge-L'].append(scores['rougeL'].fmeasure)
        
        # Calculate average scores
        avg_rouge_scores = {k: np.mean(v) for k, v in rouge_scores.items()}
        return avg_rouge_scores
    
    def calculate_meteor_score(self, references: List[str], candidates: List[str]) -> float:
        """Calculate METEOR scores"""
        assert len(references) == len(candidates), "References and candidates must have same length"
        
        meteor_scores = []
        for ref, cand in zip(references, candidates):
            try:
                score = meteor_score([ref.split()], cand.split())
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return np.mean(meteor_scores)
    
    def calculate_bert_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate BERTScore"""
        assert len(references) == len(candidates), "References and candidates must have same length"
        
        # Calculate BERTScore
        P, R, F1 = self.bert_scorer.score(candidates, references)
        
        bert_scores = {
            'bert-precision': P.mean().item(),
            'bert-recall': R.mean().item(),
            'bert-f1': F1.mean().item()
        }
        
        return bert_scores
    
    def calculate_semantic_similarity(self, references: List[str], candidates: List[str]) -> float:
        """Calculate semantic similarity using sentence transformers"""
        assert len(references) == len(candidates), "References and candidates must have same length"
        
        # Get embeddings
        ref_embeddings = self.sentence_model.encode(references)
        cand_embeddings = self.sentence_model.encode(candidates)
        
        # Calculate cosine similarity for each pair
        similarities = []
        for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
            sim = cosine_similarity([ref_emb], [cand_emb])[0][0]
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def calculate_length_stats(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate length statistics"""
        ref_lengths = [len(ref.split()) for ref in references]
        cand_lengths = [len(cand.split()) for cand in candidates]
        
        length_stats = {
            'avg_ref_length': np.mean(ref_lengths),
            'avg_cand_length': np.mean(cand_lengths),
            'length_ratio': np.mean(cand_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
            'length_diff': np.mean([abs(r - c) for r, c in zip(ref_lengths, cand_lengths)])
        }
        
        return length_stats
    
    def evaluate_comprehensive(self, references: List[str], candidates: List[str]) -> Dict[str, Union[float, Dict]]:
        """Comprehensive evaluation with all metrics"""
        print("Calculating BLEU scores...")
        bleu_scores = self.calculate_bleu_score(references, candidates)
        
        print("Calculating ROUGE scores...")
        rouge_scores = self.calculate_rouge_score(references, candidates)
        
        print("Calculating METEOR score...")
        meteor_score = self.calculate_meteor_score(references, candidates)
        
        print("Calculating BERTScore...")
        bert_scores = self.calculate_bert_score(references, candidates)
        
        print("Calculating semantic similarity...")
        semantic_sim = self.calculate_semantic_similarity(references, candidates)
        
        print("Calculating length statistics...")
        length_stats = self.calculate_length_stats(references, candidates)
        
        # Combine all metrics
        results = {
            'bleu_scores': bleu_scores,
            'rouge_scores': rouge_scores,
            'meteor_score': meteor_score,
            'bert_scores': bert_scores,
            'semantic_similarity': semantic_sim,
            'length_stats': length_stats
        }
        
        return results
    
    def format_results(self, results: Dict) -> str:
        """Format evaluation results for display"""
        formatted = []
        formatted.append("=" * 60)
        formatted.append("SPOILER GENERATION EVALUATION RESULTS")
        formatted.append("=" * 60)
        
        # BLEU scores
        formatted.append("\nBLEU Scores:")
        for metric, score in results['bleu_scores'].items():
            formatted.append(f"  {metric.upper()}: {score:.4f}")
        
        # ROUGE scores
        formatted.append("\nROUGE Scores:")
        for metric, score in results['rouge_scores'].items():
            formatted.append(f"  {metric.upper()}: {score:.4f}")
        
        # METEOR score
        formatted.append(f"\nMETEOR Score: {results['meteor_score']:.4f}")
        
        # BERTScore
        formatted.append("\nBERTScore:")
        for metric, score in results['bert_scores'].items():
            formatted.append(f"  {metric.upper()}: {score:.4f}")
        
        # Semantic similarity
        formatted.append(f"\nSemantic Similarity: {results['semantic_similarity']:.4f}")
        
        # Length statistics
        formatted.append("\nLength Statistics:")
        for metric, value in results['length_stats'].items():
            formatted.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
        
        formatted.append("=" * 60)
        return "\n".join(formatted)
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save formatted results
        formatted_path = output_path.replace('.json', '_formatted.txt')
        with open(formatted_path, 'w', encoding='utf-8') as f:
            f.write(self.format_results(results))
    
    def compare_models(self, model_results: Dict[str, Dict]) -> str:
        """Compare results from multiple models"""
        formatted = []
        formatted.append("=" * 80)
        formatted.append("MODEL COMPARISON RESULTS")
        formatted.append("=" * 80)
        
        # Extract key metrics for comparison
        key_metrics = ['bleu_scores.bleu-4', 'rouge_scores.rouge-L', 'meteor_score', 
                      'bert_scores.bert-f1', 'semantic_similarity']
        
        formatted.append(f"\n{'Model':<20} {'BLEU-4':<10} {'ROUGE-L':<10} {'METEOR':<10} {'BERT-F1':<10} {'Sem-Sim':<10}")
        formatted.append("-" * 80)
        
        for model_name, results in model_results.items():
            bleu4 = results['bleu_scores']['bleu-4']
            rougeL = results['rouge_scores']['rouge-L']
            meteor = results['meteor_score']
            bert_f1 = results['bert_scores']['bert-f1']
            sem_sim = results['semantic_similarity']
            
            formatted.append(f"{model_name:<20} {bleu4:<10.4f} {rougeL:<10.4f} {meteor:<10.4f} {bert_f1:<10.4f} {sem_sim:<10.4f}")
        
        formatted.append("=" * 80)
        return "\n".join(formatted)


def evaluate_model_output(predictions_file: str, references_file: str, output_dir: str = "evaluation_results"):
    """Evaluate model predictions against references"""
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions and references
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f.readlines()]
    
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f.readlines()]
    
    # Initialize evaluator
    evaluator = SpoilerEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_comprehensive(references, predictions)
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    evaluator.save_results(results, results_path)
    
    # Print results
    print(evaluator.format_results(results))
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Evaluator module loaded successfully!")
    
    # Test with dummy data
    references = ["This is a reference text.", "Another reference sentence."]
    candidates = ["This is a candidate text.", "Another candidate sentence."]
    
    evaluator = SpoilerEvaluator()
    results = evaluator.evaluate_comprehensive(references, candidates)
    print(evaluator.format_results(results))
