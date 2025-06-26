"""
PyTorch Fundamentals for AI Assistant Development

This module covers essential PyTorch concepts needed for building language models from scratch.
Focus on tensor operations, automatic differentiation, and basic neural network components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from typing import Optional, Tuple, List


class BasicNeuralNetwork(nn.Module):
    """
    A simple feedforward neural network to demonstrate PyTorch basics.
    This serves as a foundation before moving to more complex architectures.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super(BasicNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class SimpleAttention(nn.Module):
    """
    Simple attention mechanism implementation from scratch.
    This is a scaled-down version to understand the core concepts.
    """
    
    def __init__(self, hidden_size: int):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights and apply to values.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_size]
            key: Key tensor [batch_size, seq_len, hidden_size]
            value: Value tensor [batch_size, seq_len, hidden_size]
            mask: Optional mask tensor
            
        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        # Transform inputs
        Q = self.W_q(query)  # [batch_size, seq_len, hidden_size]
        K = self.W_k(key)    # [batch_size, seq_len, hidden_size]
        V = self.W_v(value)  # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds position information to token embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(0), :]


class SimpleTokenEmbedding(nn.Module):
    """
    Simple token embedding layer with positional encoding.
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000):
        super(SimpleTokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings with positional encoding.
        
        Args:
            x: Token indices [batch_size, seq_len]
            
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as in original transformer
        embeddings = self.embedding(x) * math.sqrt(self.d_model)
        return self.pos_encoding(embeddings)


def demonstrate_pytorch_basics():
    """
    Demonstrate basic PyTorch operations and concepts.
    Run this function to see examples of tensor operations, gradients, etc.
    """
    print("=== PyTorch Basics Demonstration ===\n")
    
    # Tensor creation and operations
    print("1. Tensor Operations:")
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    z = torch.matmul(x, y)
    print(f"Matrix multiplication: {x.shape} @ {y.shape} = {z.shape}")
    
    # Automatic differentiation
    print("\n2. Automatic Differentiation:")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print(f"dy/dx at x=2: {x.grad}")
    
    # Simple neural network training step
    print("\n3. Simple Neural Network:")
    model = BasicNeuralNetwork(10, 20, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy data
    inputs = torch.randn(32, 10)  # Batch of 32 samples
    targets = torch.randint(0, 5, (32,))  # Random class labels
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item():.4f}")
    
    # Attention demonstration
    print("\n4. Simple Attention:")
    attention = SimpleAttention(hidden_size=64)
    seq_len, batch_size, hidden_size = 10, 4, 64
    
    # Create dummy sequence data
    sequences = torch.randn(batch_size, seq_len, hidden_size)
    
    # Self-attention (query, key, value are the same)
    attention_output = attention(sequences, sequences, sequences)
    print(f"Attention input shape: {sequences.shape}")
    print(f"Attention output shape: {attention_output.shape}")
    
    # Positional encoding demonstration
    print("\n5. Positional Encoding:")
    pos_enc = PositionalEncoding(d_model=128, max_len=100)
    embeddings = torch.randn(20, 8, 128)  # [seq_len, batch_size, d_model]
    encoded = pos_enc(embeddings)
    print(f"Embeddings with positional encoding: {encoded.shape}")


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for autoregressive modeling.
    This prevents the model from seeing future tokens during training.
    
    Args:
        size: Sequence length
        
    Returns:
        Causal mask tensor
    """
    mask = torch.tril(torch.ones(size, size))
    return mask


def compute_model_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_pytorch_basics()
    
    # Additional examples
    print("\n=== Additional Examples ===")
    
    # Model parameter counting
    model = BasicNeuralNetwork(784, 256, 10)  # MNIST-like model
    num_params = compute_model_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Causal mask example
    causal_mask = create_causal_mask(5)
    print(f"\nCausal mask for sequence length 5:")
    print(causal_mask)
