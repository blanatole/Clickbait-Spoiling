#!/usr/bin/env python3
"""
Data preprocessing for clickbait spoiling tasks
"""

import json
import re
import html
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

# NLP libraries
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import emoji

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocessor for clickbait spoiling data"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        logger.info("Initializing data preprocessor...")
        
        # Initialize tokenizer for spoiler generation
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        # Initialize SBERT model for spoiler type classification
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Preprocessor initialized successfully")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML, emojis, and special characters
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or text.strip() == "":
            return ""
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emojis
        text = emoji.demojize(text, delimiters=("", ""))
        text = re.sub(r':[a-zA-Z_]+:', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-"]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def combine_post_and_target(self, post_text: List[str], target_paragraphs: List[str]) -> str:
        """
        Combine post text and target paragraphs into a single input
        
        Args:
            post_text: List of post text strings
            target_paragraphs: List of target paragraph strings
            
        Returns:
            Combined and cleaned text
        """
        # Combine post text
        post_combined = " ".join(post_text) if post_text else ""
        
        # Combine target paragraphs
        target_combined = " ".join(target_paragraphs) if target_paragraphs else ""
        
        # Create combined input
        combined_text = f"{post_combined} [SEP] {target_combined}"
        
        # Clean the combined text
        cleaned_text = self.clean_text(combined_text)
        
        return cleaned_text
    
    def tokenize_with_gpt2(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Tokenize text using GPT-2 Byte-Pair Tokenizer
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Tokenized output with input_ids, attention_mask, etc.
        """
        encoded = self.gpt2_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'text': text,
            'token_count': len(encoded['input_ids'].squeeze())
        }
    
    def create_sbert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create SBERT embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings (n_texts, 384)
        """
        # Clean texts
        cleaned_texts = [self.clean_text(text) if text else "" for text in texts]
        
        # Create embeddings
        embeddings = self.sbert_model.encode(cleaned_texts, show_progress_bar=True)
        
        return embeddings
    
    def prepare_spoiler_generation_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare data for spoiler generation task
        
        Args:
            data: Raw data list
            
        Returns:
            Processed data for spoiler generation
        """
        logger.info("Preparing data for spoiler generation...")
        processed_data = []
        
        for item in tqdm(data, desc="Processing spoiler generation data"):
            # Extract fields
            post_text = item.get('postText', [])
            target_paragraphs = item.get('targetParagraphs', [])
            spoiler = item.get('spoiler', [])
            
            # Combine input
            input_text = self.combine_post_and_target(post_text, target_paragraphs)
            
            # Clean spoiler
            spoiler_text = " ".join(spoiler) if spoiler else ""
            cleaned_spoiler = self.clean_text(spoiler_text)
            
            # Tokenize input with max length limit
            tokenized_input = self.tokenize_with_gpt2(input_text, max_length=768)
            
            # Tokenize target
            tokenized_target = self.tokenize_with_gpt2(cleaned_spoiler, max_length=256)
            
            processed_item = {
                'uuid': item.get('uuid', ''),
                'input_text': input_text,
                'target_text': cleaned_spoiler,
                'input_tokens': tokenized_input,
                'target_tokens': tokenized_target,
                'spoiler_type': item.get('tags', [''])[0] if item.get('tags') else '',
                'platform': item.get('postPlatform', ''),
                'original_post': post_text,
                'original_spoiler': spoiler
            }
            
            processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} samples for spoiler generation")
        return processed_data
    
    def prepare_spoiler_classification_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare data for spoiler type classification task
        
        Args:
            data: Raw data list
            
        Returns:
            Processed data for spoiler type classification
        """
        logger.info("Preparing data for spoiler type classification...")
        processed_data = []
        
        # Extract all text fields
        all_post_texts = []
        all_target_titles = []
        all_target_paragraphs = []
        all_target_descriptions = []
        all_target_keywords = []
        
        for item in data:
            # Extract and clean fields
            post_text = " ".join(item.get('postText', []))
            target_title = item.get('targetTitle', '')
            target_paragraphs = " ".join(item.get('targetParagraphs', []))
            target_description = item.get('targetDescription', '')
            target_keywords = item.get('targetKeywords', '')
            
            all_post_texts.append(post_text)
            all_target_titles.append(target_title)
            all_target_paragraphs.append(target_paragraphs)
            all_target_descriptions.append(target_description)
            all_target_keywords.append(target_keywords)
        
        # Create embeddings for each field
        logger.info("Creating SBERT embeddings for text fields...")
        
        post_embeddings = self.create_sbert_embeddings(all_post_texts)
        title_embeddings = self.create_sbert_embeddings(all_target_titles)
        paragraph_embeddings = self.create_sbert_embeddings(all_target_paragraphs)
        description_embeddings = self.create_sbert_embeddings(all_target_descriptions)
        keyword_embeddings = self.create_sbert_embeddings(all_target_keywords)
        
        # Combine embeddings and create processed data
        for i, item in enumerate(tqdm(data, desc="Processing classification data")):
            # Concatenate embeddings (384 * 5 = 1920 dimensions)
            combined_embedding = np.concatenate([
                post_embeddings[i],
                title_embeddings[i],
                paragraph_embeddings[i],
                description_embeddings[i],
                keyword_embeddings[i]
            ])
            
            # Get spoiler type label
            spoiler_type = item.get('tags', [''])[0] if item.get('tags') else ''
            
            processed_item = {
                'uuid': item.get('uuid', ''),
                'features': combined_embedding,
                'spoiler_type': spoiler_type,
                'platform': item.get('postPlatform', ''),
                'post_text': all_post_texts[i],
                'target_title': all_target_titles[i],
                'target_paragraphs': all_target_paragraphs[i],
                'target_description': all_target_descriptions[i],
                'target_keywords': all_target_keywords[i],
                'individual_embeddings': {
                    'post': post_embeddings[i],
                    'title': title_embeddings[i],
                    'paragraph': paragraph_embeddings[i],
                    'description': description_embeddings[i],
                    'keywords': keyword_embeddings[i]
                }
            }
            
            processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} samples for spoiler type classification")
        return processed_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str, task_type: str):
        """
        Save processed data to file
        
        Args:
            data: Processed data
            output_path: Output file path
            task_type: Type of task ('generation' or 'classification')
        """
        logger.info(f"Saving processed data to {output_path}...")
        
        if task_type == 'generation':
            # Save generation data (can be saved as JSON)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    # Convert tensors to lists for JSON serialization
                    item_copy = item.copy()
                    item_copy['input_tokens']['input_ids'] = item_copy['input_tokens']['input_ids'].tolist()
                    item_copy['input_tokens']['attention_mask'] = item_copy['input_tokens']['attention_mask'].tolist()
                    item_copy['target_tokens']['input_ids'] = item_copy['target_tokens']['input_ids'].tolist()
                    item_copy['target_tokens']['attention_mask'] = item_copy['target_tokens']['attention_mask'].tolist()
                    
                    json.dump(item_copy, f, ensure_ascii=False)
                    f.write('\n')
        
        elif task_type == 'classification':
            # Save classification data (use numpy for embeddings)
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        logger.info(f"Data saved successfully to {output_path}")
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        stats = {
            'total_samples': len(data),
            'spoiler_types': {},
            'platforms': {},
            'text_lengths': []
        }
        
        for item in data:
            # Count spoiler types
            spoiler_type = item.get('spoiler_type', 'unknown')
            stats['spoiler_types'][spoiler_type] = stats['spoiler_types'].get(spoiler_type, 0) + 1
            
            # Count platforms
            platform = item.get('platform', 'unknown')
            stats['platforms'][platform] = stats['platforms'].get(platform, 0) + 1
            
            # Text lengths
            if 'input_text' in item:
                stats['text_lengths'].append(len(item['input_text']))
        
        if stats['text_lengths']:
            stats['avg_text_length'] = np.mean(stats['text_lengths'])
            stats['median_text_length'] = np.median(stats['text_lengths'])
            stats['max_text_length'] = max(stats['text_lengths'])
            stats['min_text_length'] = min(stats['text_lengths'])
        
        return stats
