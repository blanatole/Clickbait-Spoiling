#!/usr/bin/env python3
"""
GPT-2 Spoiler Generation Model Training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime
import wandb
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpoilerDataset(Dataset):
    """Dataset for spoiler generation"""
    
    def __init__(self, data_file: str, tokenizer: GPT2Tokenizer, max_length: int = 512):
        """
        Args:
            data_file: Path to processed JSONL file
            tokenizer: GPT2 tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get input and target text
        input_text = item['input_text']
        target_text = item['target_text']
        
        # Truncate input if too long (reserve space for target and separator)
        max_input_length = self.max_length - 200  # Reserve space for target and separator
        if len(input_text) > max_input_length * 4:  # Rough estimate: 4 chars per token
            input_text = input_text[:max_input_length * 4]
        
        # Truncate target if too long  
        max_target_length = 150  # Reasonable target length
        if len(target_text) > max_target_length * 4:
            target_text = target_text[:max_target_length * 4]
        
        # Create the full sequence: input + " -> " + target + EOS
        full_text = f"{input_text} -> {target_text}<|endoftext|>"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # For GPT-2, labels are the same as input_ids (shifted internally)
        labels = input_ids.clone()
        
        # Find the separator " -> " position to mask input part
        separator_token = " -> "
        try:
            separator_pos = full_text.find(separator_token)
            if separator_pos != -1:
                # Encode just the input part to find where to start learning
                input_part = full_text[:separator_pos + len(separator_token)]
                input_encoded = self.tokenizer(input_part, add_special_tokens=False)
                input_length = len(input_encoded['input_ids'])
                
                # Mask the input part in labels (set to -100 to ignore in loss)
                labels[:input_length] = -100
        except:
            pass
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_text': input_text,
            'target_text': target_text
        }

class GPT2SpoilerModel:
    """GPT-2 model for spoiler generation"""
    
    def __init__(self, model_name: str = "gpt2-medium", device: str = None):
        """
        Args:
            model_name: GPT-2 model variant
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        logger.info(f"Loading {model_name} model...")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_model(self, 
                   train_dataset: SpoilerDataset,
                   val_dataset: SpoilerDataset,
                   batch_size: int = 8,
                   learning_rate: float = 5e-5,
                   num_epochs: int = 10,
                   warmup_steps: int = 500,
                   output_dir: str = "models/gpt2-spoiler",
                   use_wandb: bool = False):
        """
        Train the GPT-2 model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for scheduler
            output_dir: Directory to save model
            use_wandb: Whether to use wandb for logging
        """
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="clickbait-spoiling",
                name=f"gpt2-medium-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": self.model_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "warmup_steps": warmup_steps
                }
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Set up optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(output_path / "best_model")
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_model(output_path / f"checkpoint_epoch_{epoch + 1}")
        
        # Save final model
        self.save_model(output_path / "final_model")
        logger.info("Training completed!")
        
        if use_wandb:
            wandb.finish()
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected: {loss.item()}")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def generate_spoiler(self, input_text: str, max_length: int = 100, 
                        temperature: float = 0.7, top_k: int = 50, 
                        top_p: float = 0.95) -> str:
        """
        Generate spoiler for given input
        
        Args:
            input_text: Input text (post + target content)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Generated spoiler text
        """
        self.model.eval()
        
        # Prepare input
        prompt = f"{input_text} -> "
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract spoiler part
        try:
            spoiler = generated_text.split(" -> ")[1].strip()
            return spoiler
        except:
            return generated_text.replace(prompt, "").strip()
    
    def save_model(self, save_path: Path):
        """Save model and tokenizer"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: Path):
        """Load model and tokenizer"""
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")

def main():
    """Main training function"""
    
    # Configuration
    config = {
        "model_name": "gpt2-medium",
        "batch_size": 4,  # Reduced batch size for stability
        "learning_rate": 2e-5,  # Reduced learning rate for stability  
        "num_epochs": 10,
        "warmup_steps": 500,
        "output_dir": "models/gpt2-spoiler",
        "use_wandb": False,  # Set to True if you want to use wandb
        "gradient_clip_val": 1.0,  # Add gradient clipping
        "weight_decay": 0.01  # Add weight decay
    }
    
    # Create model
    model = GPT2SpoilerModel(config["model_name"])
    
    # Create datasets
    train_dataset = SpoilerDataset(
        "processed_data/train_generation.jsonl",
        model.tokenizer
    )
    
    val_dataset = SpoilerDataset(
        "processed_data/validation_generation.jsonl",
        model.tokenizer
    )
    
    # Train model
    model.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        warmup_steps=config["warmup_steps"],
        output_dir=config["output_dir"],
        use_wandb=config["use_wandb"]
    )

if __name__ == "__main__":
    main()
