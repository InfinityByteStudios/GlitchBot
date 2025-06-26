#!/usr/bin/env python3
"""
GlitchBot Training Script

Train GlitchBot language model on custom dataset using local compute.
No pretrained weights or third-party APIs.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
from pathlib import Path
from tqdm import tqdm
import json
import time
import argparse
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.glitchbot_transformer import GlitchBotTransformer, create_glitchbot_model
from tokenizer.glitchbot_tokenizer import GlitchBotTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for text training."""
    
    def __init__(self, texts: List[str], tokenizer: GlitchBotTokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            # Pad with pad tokens
            token_ids.extend([self.tokenizer.pad_token_id] * (self.max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)


class GlitchBotTrainer:
    """GlitchBot training class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Setup paths
        self.setup_paths()
        
        # Load or create tokenizer
        self.tokenizer = self.setup_tokenizer()
        
        # Update vocab size in config
        self.config['model']['vocab_size'] = len(self.tokenizer.vocab)
        
        # Create model
        self.model = self.create_model()
        
        # Setup training
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if self.config.get('device', {}).get('mixed_precision', False) else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def setup_paths(self):
        """Create necessary directories."""
        paths = self.config['paths']
        for path_key, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def setup_tokenizer(self) -> GlitchBotTokenizer:
        """Load or create tokenizer."""
        tokenizer_path = Path(self.config['data']['tokenizer_path'])
        
        if (tokenizer_path / 'tokenizer.pkl').exists():
            logger.info(f"📖 Loading tokenizer from {tokenizer_path}")
            return GlitchBotTokenizer.load(str(tokenizer_path))
        else:
            logger.info("Creating new tokenizer...")
            
            # Load corpus
            corpus_path = Path(self.config['data']['corpus_path'])
            if not corpus_path.exists():
                logger.info("Creating corpus from existing data...")
                from create_corpus import create_cleaned_corpus
                create_cleaned_corpus()
            
            # Read corpus
            with open(corpus_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Training tokenizer on {len(texts):,} texts...")
            
            # Create and train tokenizer
            tokenizer = GlitchBotTokenizer(vocab_size=self.config['model']['vocab_size'])
            tokenizer.train(texts[:10000], save_path=str(tokenizer_path))  # Train on subset for speed
            
            return tokenizer
    
    def create_model(self) -> GlitchBotTransformer:
        """Create GlitchBot model."""
        logger.info("Creating GlitchBot model...")
        
        model = create_glitchbot_model(self.config['model'])
        model.to(self.device)
        
        # Model compilation (PyTorch 2.0+)
        if self.config.get('device', {}).get('compile_model', False):
            try:
                model = torch.compile(model)
                logger.info("⚡ Model compiled with PyTorch 2.0")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Log model info
        total_params = model.get_num_params()
        trainable_params = model.get_num_trainable_params()
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    
    def setup_optimizer(self) -> AdamW:
        """Setup optimizer."""
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(training_config['learning_rate']),
            betas=optimizer_config['betas'],
            eps=float(optimizer_config['eps']),
            weight_decay=float(training_config['weight_decay'])
        )
        
        logger.info(f"⚙️  Optimizer: AdamW (lr={training_config['learning_rate']})")
        return optimizer
    
    def setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        
        if scheduler_config.get('name') == 'cosine':
            warmup_steps = scheduler_config.get('warmup_steps', 1000)
            total_steps = self.config['training']['epochs'] * 1000  # Estimate
            
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps - warmup_steps,
                T_mult=1,
                eta_min=0
            )
            logger.info(f"📈 Scheduler: Cosine with {warmup_steps} warmup steps")
            return scheduler
        
        return None
    
    def load_dataset(self) -> tuple:
        """Load training dataset."""
        corpus_path = Path(self.config['data']['corpus_path'])
        
        logger.info(f"📖 Loading dataset from {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"📊 Loaded {len(texts):,} texts")
        
        # Split train/val
        split_idx = int(len(texts) * 0.9)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Create datasets
        train_dataset = TextDataset(train_texts, self.tokenizer, self.config['data']['max_length'])
        val_dataset = TextDataset(val_texts, self.tokenizer, self.config['data']['max_length'])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"📊 Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")
        
        return train_loader, val_loader
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        
        input_ids = batch.to(self.device)
        labels = input_ids.clone()
        
        # Mixed precision training
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                input_ids = batch.to(self.device)
                labels = input_ids.clone()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                
                # Calculate tokens (excluding padding)
                valid_tokens = (labels != self.tokenizer.pad_token_id).sum().item()
                total_tokens += valid_tokens
                total_loss += loss.item() * valid_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }
    
    def generate_samples(self, epoch: int):
        """Generate text samples."""
        self.model.eval()
        
        samples_dir = Path(self.config['paths']['samples'])
        sample_file = samples_dir / f"sample_epoch_{epoch}.txt"
        
        prompts = self.config['evaluation']['sample_prompts']
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(f"GlitchBot Samples - Epoch {epoch}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, prompt in enumerate(prompts):
                f.write(f"Prompt {i+1}: {prompt}\n")
                f.write("-" * 30 + "\n")
                
                # Encode prompt
                input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
                
                # Generate
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=50,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Decode
                generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
                f.write(generated_text + "\n\n")
        
        logger.info(f"📝 Generated samples saved to {sample_file}")
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['paths']['checkpoints'])
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 New best model saved (loss: {loss:.4f})")
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting GlitchBot training...")
        
        # Load dataset
        train_loader, val_loader = self.load_dataset()
        
        # Training metrics
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\n🔄 Epoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Training
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(pbar):
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Save checkpoint every N steps
                if self.global_step % self.config['training']['save_every'] == 0:
                    self.save_checkpoint(epoch, loss)
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Evaluation
            if (epoch + 1) % self.config['training']['eval_every'] == 0:
                eval_metrics = self.evaluate(val_loader)
                val_losses.append(eval_metrics['loss'])
                
                # Check if best model
                is_best = eval_metrics['loss'] < self.best_loss
                if is_best:
                    self.best_loss = eval_metrics['loss']
                
                # Save checkpoint
                self.save_checkpoint(epoch, eval_metrics['loss'], is_best)
                
                # Generate samples
                self.generate_samples(epoch)
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                logger.info(f"📊 Epoch {epoch + 1} Summary:")
                logger.info(f"   Train Loss: {avg_train_loss:.4f}")
                logger.info(f"   Val Loss: {eval_metrics['loss']:.4f}")
                logger.info(f"   Perplexity: {eval_metrics['perplexity']:.2f}")
                logger.info(f"   Time: {epoch_time:.1f}s")
                logger.info(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save final model
        final_model_path = Path(self.config['paths']['models']) / "glitchbot-v1-final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_path': self.config['data']['tokenizer_path'],
            'config': self.config
        }, final_model_path)
        
        logger.info(f"🎉 Training complete! Final model saved to {final_model_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train GlitchBot language model")
    parser.add_argument("--config", type=str, default="configs/glitchbot_base.yaml", 
                       help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / 'train.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create trainer
    trainer = GlitchBotTrainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if trainer.scaler and 'scaler_state_dict' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.epoch = checkpoint['epoch']
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
