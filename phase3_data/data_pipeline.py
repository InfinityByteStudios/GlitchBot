"""
Data Collection and Preprocessing Pipeline

This module handles the collection, cleaning, and preprocessing of training data
for the AI assistant. It includes tokenization, data validation, and efficient
data loading for large-scale training.
"""

import os
import json
import re
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing pipeline."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    tokenizer_dir: str = "tokenizer"
    min_text_length: int = 50
    max_text_length: int = 2048
    vocab_size: int = 50000
    chunk_size: int = 1000000  # Process data in chunks
    num_workers: int = 4
    validation_split: float = 0.05
    test_split: float = 0.05


class DataCollector:
    """
    Collect text data from various sources for training.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_sources = {
            'openwebtext': self._download_openwebtext,
            'wikipedia': self._download_wikipedia,
            'books': self._download_books,
            'news': self._download_news
        }
    
    def collect_all_data(self, sources: List[str] = None) -> Dict[str, str]:
        """
        Collect data from all specified sources.
        
        Args:
            sources: List of data sources to collect from
            
        Returns:
            Dictionary mapping source names to file paths
        """
        if sources is None:
            sources = list(self.data_sources.keys())
        
        collected_files = {}
        
        for source in sources:
            if source in self.data_sources:
                logger.info(f"Collecting data from {source}...")
                try:
                    file_path = self.data_sources[source]()
                    collected_files[source] = file_path
                    logger.info(f"Successfully collected {source} data")
                except Exception as e:
                    logger.error(f"Failed to collect {source} data: {e}")
            else:
                logger.warning(f"Unknown data source: {source}")
        
        return collected_files
    
    def _download_openwebtext(self) -> str:
        """Download OpenWebText dataset."""
        try:
            # Use Hugging Face datasets library
            dataset = load_dataset("openwebtext", split="train", streaming=True)
            
            output_file = os.path.join(self.config.raw_data_dir, "openwebtext.jsonl")
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, example in enumerate(dataset):
                    if i >= 100000:  # Limit for initial testing
                        break
                    json.dump({'text': example['text']}, f)
                    f.write('\n')
            
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to download OpenWebText: {e}")
            # Fallback: create sample data
            return self._create_sample_data("openwebtext_sample.jsonl")
    
    def _download_wikipedia(self) -> str:
        """Download Wikipedia dataset."""
        try:
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            
            output_file = os.path.join(self.config.raw_data_dir, "wikipedia.jsonl")
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, example in enumerate(dataset):
                    if i >= 50000:  # Limit for initial testing
                        break
                    json.dump({'text': example['text']}, f)
                    f.write('\n')
            
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to download Wikipedia: {e}")
            return self._create_sample_data("wikipedia_sample.jsonl")
    
    def _download_books(self) -> str:
        """Download books dataset (Project Gutenberg)."""
        # Placeholder implementation - would need to implement actual book downloading
        return self._create_sample_data("books_sample.jsonl")
    
    def _download_news(self) -> str:
        """Download news dataset."""
        # Placeholder implementation - would integrate with news APIs
        return self._create_sample_data("news_sample.jsonl")
    
    def _create_sample_data(self, filename: str) -> str:
        """Create sample data for testing when real datasets aren't available."""
        output_file = os.path.join(self.config.raw_data_dir, filename)
        os.makedirs(self.config.raw_data_dir, exist_ok=True)
        
        sample_texts = [
            "The field of artificial intelligence has seen remarkable progress in recent years.",
            "Machine learning models are becoming increasingly sophisticated and capable.",
            "Natural language processing enables computers to understand and generate human language.",
            "Deep learning networks can learn complex patterns from large amounts of data.",
            "Transformers have revolutionized the way we approach language modeling tasks.",
            "Building AI systems from scratch provides deep insights into their inner workings.",
            "The attention mechanism allows models to focus on relevant parts of the input.",
            "Large language models demonstrate emergent capabilities as they scale up.",
            "Training neural networks requires careful consideration of optimization techniques.",
            "Data quality and preprocessing are crucial for successful model training."
        ] * 1000  # Repeat to create more training data
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                json.dump({'text': text}, f)
                f.write('\n')
        
        logger.info(f"Created sample data file: {output_file}")
        return output_file


class TextCleaner:
    """
    Clean and preprocess raw text data.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean a single text string.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string or None if text should be filtered out
        """
        if not text or len(text.strip()) < self.config.min_text_length:
            return None
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove very long lines (likely code or data dumps)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line) < 500]
        text = '\n'.join(cleaned_lines)
        
        # Filter out texts that are mostly numbers or special characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.7:
            return None
        
        # Truncate if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        return text
    
    def clean_file(self, input_file: str, output_file: str) -> int:
        """
        Clean an entire file of text data.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            
        Returns:
            Number of cleaned texts
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cleaned_count = 0
        total_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc=f"Cleaning {os.path.basename(input_file)}"):
                total_count += 1
                try:
                    data = json.loads(line.strip())
                    cleaned_text = self.clean_text(data['text'])
                    
                    if cleaned_text:
                        json.dump({'text': cleaned_text}, outfile)
                        outfile.write('\n')
                        cleaned_count += 1
                        
                except (json.JSONDecodeError, KeyError):
                    continue
        
        logger.info(f"Cleaned {cleaned_count}/{total_count} texts from {input_file}")
        return cleaned_count


class SimpleTokenizer:
    """
    Simple BPE-style tokenizer implementation from scratch.
    Note: For production use, consider using SentencePiece or other optimized tokenizers.
    """
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = {}
        self.vocab = {}
        
    def _get_word_freqs(self, texts: List[str]):
        """Count word frequencies in the corpus."""
        self.word_freqs = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_freqs[word] = self.word_freqs.get(word, 0) + 1
    
    def _get_splits(self):
        """Split words into characters."""
        self.splits = {}
        for word in self.word_freqs:
            self.splits[word] = list(word)
    
    def _compute_pair_freqs(self):
        """Compute frequencies of character pairs."""
        pair_freqs = {}
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        return pair_freqs
    
    def _merge_vocab(self, pair):
        """Merge the most frequent pair in vocabulary."""
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [pair[0] + pair[1]] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
    
    def train(self, texts: List[str]):
        """Train the BPE tokenizer on given texts."""
        logger.info("Training BPE tokenizer...")
        
        # Step 1: Get word frequencies
        self._get_word_freqs(texts)
        
        # Step 2: Split words into characters
        self._get_splits()
        
        # Step 3: Build initial vocabulary
        alphabet = set()
        for word in self.word_freqs:
            alphabet.update(word)
        
        self.vocab = {char: i for i, char in enumerate(sorted(alphabet))}
        
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Step 4: Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in tqdm(range(num_merges), desc="BPE merges"):
            pair_freqs = self._compute_pair_freqs()
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self._merge_vocab(best_pair)
            self.merges[best_pair] = i
            self.vocab[best_pair[0] + best_pair[1]] = len(self.vocab)
        
        logger.info(f"Tokenizer training complete. Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.splits:
                word_tokens = self.splits[word]
            else:
                # Handle unknown words
                word_tokens = list(word)
            
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['<unk>'])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id_, '<unk>') for id_ in token_ids]
        return ' '.join(tokens)
    
    def save(self, save_dir: str):
        """Save tokenizer to directory."""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'vocab.json'), 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        with open(os.path.join(save_dir, 'merges.json'), 'w') as f:
            json.dump({f"{k[0]} {k[1]}": v for k, v in self.merges.items()}, f, indent=2)
    
    def load(self, save_dir: str):
        """Load tokenizer from directory."""
        with open(os.path.join(save_dir, 'vocab.json'), 'r') as f:
            self.vocab = json.load(f)
        
        with open(os.path.join(save_dir, 'merges.json'), 'r') as f:
            merges_data = json.load(f)
            self.merges = {tuple(k.split()): v for k, v in merges_data.items()}


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data with tokenization.
    """
    
    def __init__(self, data_file: str, tokenizer: SimpleTokenizer, max_length: int = 1024):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
    
    def _load_data(self) -> List[str]:
        """Load text data from file."""
        texts = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    texts.append(data['text'])
                except (json.JSONDecodeError, KeyError):
                    continue
        return texts
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            # Pad with pad token
            pad_token_id = self.tokenizer.vocab.get('<pad>', 0)
            token_ids.extend([pad_token_id] * (self.max_length - len(token_ids)))
        
        # For causal language modeling, input and target are the same but shifted
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': target_ids
        }


def create_data_pipeline(config: DataConfig = None) -> Dict[str, any]:
    """
    Create complete data processing pipeline.
    
    Args:
        config: Data configuration
        
    Returns:
        Dictionary containing processed datasets and tokenizer
    """
    if config is None:
        config = DataConfig()
    
    logger.info("Starting data processing pipeline...")
    
    # Step 1: Collect raw data
    collector = DataCollector(config)
    raw_files = collector.collect_all_data(['openwebtext'])  # Start with one source
    
    # Step 2: Clean data
    cleaner = TextCleaner(config)
    cleaned_files = {}
    
    for source, raw_file in raw_files.items():
        cleaned_file = os.path.join(config.processed_data_dir, f"{source}_cleaned.jsonl")
        cleaner.clean_file(raw_file, cleaned_file)
        cleaned_files[source] = cleaned_file
    
    # Step 3: Combine all cleaned data
    combined_file = os.path.join(config.processed_data_dir, "combined_data.jsonl")
    os.makedirs(config.processed_data_dir, exist_ok=True)
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for source, cleaned_file in cleaned_files.items():
            with open(cleaned_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    
    # Step 4: Train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Load texts for tokenizer training
    texts = []
    with open(combined_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:  # Use subset for tokenizer training
                break
            try:
                data = json.loads(line.strip())
                texts.append(data['text'])
            except (json.JSONDecodeError, KeyError):
                continue
    
    tokenizer.train(texts)
    tokenizer.save(config.tokenizer_dir)
    
    # Step 5: Create train/validation splits
    train_file, val_file, test_file = _split_data(combined_file, config)
    
    # Step 6: Create datasets
    train_dataset = TextDataset(train_file, tokenizer)
    val_dataset = TextDataset(val_file, tokenizer)
    test_dataset = TextDataset(test_file, tokenizer)
    
    logger.info("Data processing pipeline complete!")
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'tokenizer': tokenizer,
        'config': config
    }


def _split_data(data_file: str, config: DataConfig) -> Tuple[str, str, str]:
    """Split data into train/validation/test sets."""
    logger.info("Splitting data into train/validation/test sets...")
    
    # Read all lines
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle lines
    np.random.shuffle(lines)
    
    # Calculate split indices
    total_lines = len(lines)
    val_size = int(total_lines * config.validation_split)
    test_size = int(total_lines * config.test_split)
    train_size = total_lines - val_size - test_size
    
    # Split data
    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]
    
    # Write split files
    train_file = os.path.join(config.processed_data_dir, "train.jsonl")
    val_file = os.path.join(config.processed_data_dir, "validation.jsonl")
    test_file = os.path.join(config.processed_data_dir, "test.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    logger.info(f"Data split complete: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")
    
    return train_file, val_file, test_file


if __name__ == "__main__":
    # Create data pipeline
    config = DataConfig()
    data_pipeline = create_data_pipeline(config)
    
    # Test the pipeline
    print("\n=== Data Pipeline Test ===")
    print(f"Train dataset size: {len(data_pipeline['train_dataset'])}")
    print(f"Validation dataset size: {len(data_pipeline['val_dataset'])}")
    print(f"Test dataset size: {len(data_pipeline['test_dataset'])}")
    print(f"Tokenizer vocab size: {len(data_pipeline['tokenizer'].vocab)}")
    
    # Test tokenization
    sample_text = "This is a sample text for testing the tokenizer."
    token_ids = data_pipeline['tokenizer'].encode(sample_text)
    decoded_text = data_pipeline['tokenizer'].decode(token_ids)
    
    print(f"\nTokenization test:")
    print(f"Original: {sample_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded_text}")
    
    # Test data loader
    train_loader = DataLoader(
        data_pipeline['train_dataset'],
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    batch = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("\n✅ Data pipeline setup complete!")
    print("💡 Next: Proceed to Phase 4 - Implement Transformer Model")
