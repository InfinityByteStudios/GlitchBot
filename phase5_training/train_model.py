"""
Training Pipeline for Small Language Model

This module implements the complete training pipeline for the AI assistant,
including data loading, model training, evaluation, and checkpointing.
This is Phase 5: Train a Small Language Model (Proof of Concept).
"""

import json
import math
import time
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    setup_logging, set_seed, get_device, count_parameters, 
    CosineWarmupScheduler, TrainingMetrics, ModelCheckpoint
)
from phase4_model.transformer_model import create_gpt_model
from phase3_data.data_pipeline import create_data_pipeline, TextDataset

# Configure logging
logger = setup_logging("INFO")


@dataclass
class TrainingConfig:
    """Configuration for training the language model."""
    
    # Model configuration
    vocab_size: int = 50000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 1024
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduling
    warmup_steps: int = 1000
    max_steps: int = 50000  # Small model training
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    
    # Hardware
    device: str = "auto"
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Paths
    output_dir: str = "outputs/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    # Data
    train_data_path: str = "data/processed/train.jsonl"
    val_data_path: str = "data/processed/validation.jsonl"
    tokenizer_path: str = "tokenizer/"
    
    # Evaluation
    eval_batch_size: int = 8
    num_eval_samples: int = 1000
    generate_samples: bool = True
    
    # Experiment
    run_name: str = "gpt_small_v1"
    seed: int = 42


class LanguageModelTrainer:
    """Main trainer class for the language model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device
        if config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(config.device)
        
        logger.info("Using device: %s", self.device)
        
        # Create output directories
        self.setup_directories()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataset = None
        self.val_dataset = None
        self.val_dataloader = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics and checkpointing
        self.metrics = TrainingMetrics()
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            save_total_limit=5
        )
        
    def setup_directories(self):
        """Create necessary directories for training."""
        directories = [
            self.config.output_dir,
            self.config.checkpoint_dir,
            self.config.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self):
        """Initialize the model."""
        logger.info("Setting up model...")
        
        model_config = {
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'd_ff': self.config.d_ff,
            'max_seq_length': self.config.max_seq_length,
            'dropout': self.config.dropout
        }
        
        self.model = create_gpt_model(model_config)
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            # Note: Our custom model doesn't have gradient checkpointing built-in
            # This would need to be implemented in the transformer model
            logger.info("Gradient checkpointing requested but not implemented in custom model")
        
        # Count parameters
        num_params = count_parameters(self.model)
        logger.info("Model created with %s parameters", f"{num_params:,}")
        
        # Save model config
        config_path = Path(self.config.output_dir) / "model_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2)
    
    def setup_data(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        try:
            # Create data pipeline if not exists
            if not Path(self.config.train_data_path).exists():
                logger.info("Creating data pipeline...")
                from phase3_data.data_pipeline import DataConfig
                data_config = DataConfig()
                data_pipeline = create_data_pipeline(data_config)
                
                # Use the created datasets
                self.train_dataset = data_pipeline['train_dataset']  # pylint: disable=attribute-defined-outside-init
                self.val_dataset = data_pipeline['val_dataset']  # pylint: disable=attribute-defined-outside-init
            else:
                # Load existing data
                from phase3_data.data_pipeline import SimpleTokenizer
                tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)
                tokenizer.load(self.config.tokenizer_path)
                
                self.train_dataset = TextDataset(  # pylint: disable=attribute-defined-outside-init
                    self.config.train_data_path, 
                    tokenizer, 
                    self.config.max_seq_length
                )
                self.val_dataset = TextDataset(  # pylint: disable=attribute-defined-outside-init
                    self.config.val_data_path, 
                    tokenizer, 
                    self.config.max_seq_length
                )
            
            # Create data loaders
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            logger.info("Train dataset size: %s", len(self.train_dataset))
            logger.info("Validation dataset size: %s", len(self.val_dataset))
            
        except (FileNotFoundError, json.JSONDecodeError, ImportError) as e:
            logger.error("Failed to setup data: %s", e)
            # Create dummy data for testing
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy data for testing when real data is not available."""
        logger.info("Creating dummy data for testing...")
        
        # Create simple dummy datasets
        dummy_size = 1000
        dummy_data = []
        
        for _ in range(dummy_size):
            # Create random sequences
            sequence_length = np.random.randint(50, self.config.max_seq_length)
            input_ids = torch.randint(0, self.config.vocab_size, (sequence_length - 1,))
            labels = torch.randint(0, self.config.vocab_size, (sequence_length - 1,))
            
            dummy_data.append({
                'input_ids': input_ids,
                'labels': labels
            })
        
        # Split into train/val
        split_idx = int(0.9 * len(dummy_data))
        train_data = dummy_data[:split_idx]
        val_data = dummy_data[split_idx:]
        
        # Create simple dataset class
        class DummyDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.train_dataset = DummyDataset(train_data)  # pylint: disable=attribute-defined-outside-init
        self.val_dataset = DummyDataset(val_data)  # pylint: disable=attribute-defined-outside-init
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        logger.info("Created dummy train dataset: %s samples", len(self.train_dataset))
        logger.info("Created dummy val dataset: %s samples", len(self.val_dataset))
    
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        # Pad sequences to the same length
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Find max length in batch
        max_len = max(len(seq) for seq in input_ids)
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        
        for inp, lab in zip(input_ids, labels):
            pad_len = max_len - len(inp)
            padded_input_ids.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
            padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'labels': torch.stack(padded_labels)
        }
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            min_lr_ratio=self.config.min_learning_rate / self.config.learning_rate
        )
        
        # Setup mixed precision training
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        if self.scaler is not None:
            with autocast():
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_steps = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            # Training step
            step_loss = self.train_step(batch)
            total_loss += step_loss
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                num_steps += 1
                
                # Update progress bar
                current_loss = total_loss / num_steps
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.metrics.update({
                        'train_loss': current_loss,
                        'learning_rate': current_lr,
                        'epoch': self.epoch
                    }, self.global_step)
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.metrics.update(eval_metrics, self.global_step)
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if we've reached max steps
                if self.global_step >= self.config.max_steps:
                    break
        
        return {
            'train_loss': total_loss / max(num_steps, 1),
            'num_steps': num_steps
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit evaluation for speed
                if num_batches >= self.config.num_eval_samples // self.config.eval_batch_size:
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss)
        
        logger.info("Evaluation - Loss: %.4f, Perplexity: %.2f", avg_loss, perplexity)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(is_best=True)
        
        self.model.train()  # Return to training mode
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        logger.info("Saving checkpoint at step %s", self.global_step)
        
        metrics = {
            'train_loss': self.metrics.get_latest('train_loss'),
            'eval_loss': self.metrics.get_latest('eval_loss'),
            'eval_perplexity': self.metrics.get_latest('eval_perplexity'),
            'best_val_loss': self.best_val_loss
        }
        
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            metrics=metrics,
            is_best=is_best
        )
        
        # Save metrics
        metrics_path = Path(self.config.output_dir) / "training_metrics.json"
        self.metrics.save(str(metrics_path))
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup all components
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # Training loop
        start_time = time.time()
        
        try:
            for epoch in range(1000):  # Large number, will break based on steps
                self.epoch = epoch
                
                epoch_metrics = self.train_epoch()
                
                logger.info(
                    "Epoch %d completed - Train Loss: %.4f, Steps: %d, Global Step: %d",
                    epoch,
                    epoch_metrics['train_loss'],
                    epoch_metrics['num_steps'],
                    self.global_step
                )
                
                # Check if we've reached max steps
                if self.global_step >= self.config.max_steps:
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("Training failed: %s", e)
            raise
        
        finally:
            # Final save
            self.save_checkpoint()
            
            # Training summary
            end_time = time.time()
            training_time = end_time - start_time
            
            logger.info("Training completed!")
            logger.info("Total training time: %.2f seconds", training_time)
            logger.info("Total steps: %s", self.global_step)
            logger.info("Best validation loss: %.4f", self.best_val_loss)


def main():
    """Main training function."""
    # Create training configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = LanguageModelTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
