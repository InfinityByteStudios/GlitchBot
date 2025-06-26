"""
Basic AI Assistant Training Script

This script trains a small AI model specifically for basic conversations and math.
It's designed to be lightweight and fast for quick proof-of-concept training.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import setup_logging, set_seed, get_device
from phase4_model.transformer_model import create_gpt_model
from basic_data_generator import BasicTrainingDataGenerator

logger = setup_logging("INFO")


class BasicDataset(Dataset):
    """Simple dataset for basic training data."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item['text'])
        
        logger.info(f"Loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Use the tokenizer to encode the text properly
        tokens = self.tokenizer.encode(text[:self.max_length])
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens, dtype=torch.long)


class SimpleTokenizer:
    """Very basic character-level tokenizer for proof of concept."""
    
    def __init__(self):
        # Character vocabulary
        self.chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t")
        self.vocab_size = len(self.chars) + 1  # +1 for padding
        self.char_to_id = {ch: i+1 for i, ch in enumerate(self.chars)}  # Start from 1, 0 is padding
        self.id_to_char = {i+1: ch for i, ch in enumerate(self.chars)}
        self.id_to_char[0] = '<PAD>'
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.id_to_char.get(id, '') for id in token_ids])


class BasicTrainer:
    """Simple trainer for basic AI assistant capabilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device()
        set_seed(config.get('seed', 42))
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # Update config with actual vocab size
        self.config['vocab_size'] = self.tokenizer.vocab_size
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
    
    def _create_model(self):
        """Create a small transformer model."""
        model_config = {
            'vocab_size': self.config['vocab_size'],
            'd_model': self.config['d_model'],
            'num_heads': self.config['num_heads'],
            'num_layers': self.config['num_layers'],
            'd_ff': self.config['d_ff'],
            'max_seq_length': self.config['max_seq_length'],
            'dropout': self.config['dropout'],
            'pad_token_id': 0
        }
        return create_gpt_model(model_config)
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Input is the sequence, target is the same sequence shifted by 1
        input_ids = batch[:, :-1].to(self.device)
        target_ids = batch[:, 1:].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Calculate loss
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[:, :-1].to(self.device)
                target_ids = batch[:, 1:].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8):
        """Generate text given a prompt."""
        self.model.eval()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                if len(generated) >= self.config['max_seq_length']:
                    break
                
                # Get model output
                outputs = self.model(input_tensor)
                
                # Get next token probabilities
                next_token_logits = outputs[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                
                # Check for natural stopping
                if next_token == 0 or self.tokenizer.id_to_char.get(next_token, '') == '\n':
                    # Stop if we hit padding or newline after assistant response
                    if 'Assistant:' in self.tokenizer.decode(generated[-20:]):
                        break
                
                generated.append(next_token)
                
                # Update input tensor
                input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
                
                # Truncate if too long
                if input_tensor.size(1) > self.config['max_seq_length']:
                    input_tensor = input_tensor[:, -self.config['max_seq_length']:]
        
        return self.tokenizer.decode(generated)
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = 5):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Training loop
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
            
            # Generate sample text
            if (epoch + 1) % 2 == 0:  # Every 2 epochs
                sample_prompt = "Human: What is 5 + 3?\nAssistant:"
                generated = self.generate_text(sample_prompt, max_length=50)
                logger.info(f"Sample generation: {generated}")
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer': self.tokenizer
        }, path)
        logger.info(f"Model saved to {path}")


def main():
    """Main training function."""
    # Configuration
    config = {
        'vocab_size': 100,  # Will be updated by tokenizer
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 4,
        'd_ff': 512,
        'max_seq_length': 512,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'batch_size': 8,
        'max_grad_norm': 1.0,
        'seed': 42
    }
    
    # Generate training data if needed
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    train_file = data_dir / "basic_training.jsonl"
    val_file = data_dir / "basic_validation.jsonl"
    
    if not train_file.exists():
        logger.info("Generating basic training data...")
        generator = BasicTrainingDataGenerator()
        generator.save_training_data(train_file, num_math_problems=2000)
        
        # Generate validation data
        validation_data = generator.generate_training_data(num_math_problems=300)
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in validation_data:
                training_item = {
                    "text": f"Human: {item['input']}\nAssistant: {item['output']}\n"
                }
                f.write(json.dumps(training_item) + '\n')
    
    # Initialize trainer
    trainer = BasicTrainer(config)
    
    # Create datasets
    train_dataset = BasicDataset(train_file, trainer.tokenizer, config['max_seq_length'])
    val_dataset = BasicDataset(val_file, trainer.tokenizer, config['max_seq_length'])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Train model
    trainer.train(train_dataloader, val_dataloader, num_epochs=10)
    
    # Save model
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    trainer.save_model(output_dir / "basic_ai_model.pth")
    
    # Test some examples
    logger.info("\nTesting the trained model:")
    test_prompts = [
        "Human: Hello\nAssistant:",
        "Human: What is 2 + 3?\nAssistant:",
        "Human: How are you?\nAssistant:",
        "Human: What is 10 * 5?\nAssistant:"
    ]
    
    for prompt in test_prompts:
        response = trainer.generate_text(prompt, max_length=50, temperature=0.7)
        logger.info(f"Prompt: {prompt.strip()}")
        logger.info(f"Response: {response}\n")


if __name__ == "__main__":
    main()
