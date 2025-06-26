"""
Attention Mechanism Implementation from Scratch

This module implements various attention mechanisms that are fundamental
to transformer architectures. Everything is built from scratch to ensure
deep understanding before moving to more complex implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in "Attention is All You Need".
    This is the core attention mechanism used in transformers.
    """
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_k]
            key: Key tensor [batch_size, seq_len, d_k]
            value: Value tensor [batch_size, seq_len, d_v]
            mask: Optional attention mask
            
        Returns:
            (attention_output, attention_weights)
        """
        batch_size, seq_len, d_k = query.size()
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-head attention forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Multi-head attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        # 1. Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention to each head
        # Reshape mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
        
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.W_o(attention_output)
        
        return output


class SelfAttention(nn.Module):
    """
    Self-attention mechanism where query, key, and value come from the same input.
    This is the standard attention used in transformer blocks.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Self-attention output [batch_size, seq_len, d_model]
        """
        return self.multihead_attention(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism where query comes from one sequence
    and key, value come from another sequence.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, query_seq: torch.Tensor, key_value_seq: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query_seq: Query sequence [batch_size, seq_len_q, d_model]
            key_value_seq: Key-value sequence [batch_size, seq_len_kv, d_model]
            mask: Optional attention mask
            
        Returns:
            Cross-attention output [batch_size, seq_len_q, d_model]
        """
        return self.multihead_attention(query_seq, key_value_seq, key_value_seq, mask)


def create_padding_mask(sequences: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create a padding mask to ignore padded tokens in attention.
    
    Args:
        sequences: Input sequences [batch_size, seq_len]
        pad_token: Padding token ID
        
    Returns:
        Padding mask [batch_size, 1, 1, seq_len]
    """
    return (sequences != pad_token).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention.
    
    Args:
        size: Sequence length
        device: Device to create tensor on
        
    Returns:
        Causal mask [1, 1, size, size]
    """
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def visualize_attention_weights(attention_weights: torch.Tensor, 
                              tokens: list = None,
                              head_idx: int = 0,
                              save_path: str = None):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        tokens: List of token strings for labeling
        head_idx: Which attention head to visualize
        save_path: Optional path to save the plot
    """
    # Extract weights for specific head and first batch item
    weights = attention_weights[0, head_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
    
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(f'Attention Weights - Head {head_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def demonstrate_attention_mechanisms():
    """
    Demonstrate different attention mechanisms with examples.
    """
    print("=== Attention Mechanisms Demonstration ===\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    batch_size, seq_len, d_model = 2, 8, 512
    num_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("1. Scaled Dot-Product Attention:")
    attention = ScaledDotProductAttention(d_k=d_model)
    output, weights = attention(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\n2. Multi-Head Attention:")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output = mha(x, x, x)
    print(f"Multi-head output shape: {mha_output.shape}")
    
    print("\n3. Self-Attention:")
    self_attn = SelfAttention(d_model, num_heads)
    self_attn_output = self_attn(x)
    print(f"Self-attention output shape: {self_attn_output.shape}")
    
    print("\n4. Cross-Attention:")
    cross_attn = CrossAttention(d_model, num_heads)
    encoder_output = torch.randn(batch_size, 10, d_model)  # Different length
    cross_attn_output = cross_attn(x, encoder_output)
    print(f"Cross-attention output shape: {cross_attn_output.shape}")
    
    print("\n5. Attention Masks:")
    # Causal mask
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    
    # Padding mask
    padded_sequences = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0],
                                   [1, 2, 3, 4, 5, 6, 0, 0]])
    padding_mask = create_padding_mask(padded_sequences)
    print(f"Padding mask shape: {padding_mask.shape}")
    
    # Apply masked self-attention
    masked_output = self_attn(x, causal_mask)
    print(f"Masked self-attention output shape: {masked_output.shape}")


class AttentionVisualization:
    """
    Helper class for visualizing attention patterns and understanding
    what the model learns to attend to.
    """
    
    @staticmethod
    def compute_attention_stats(attention_weights: torch.Tensor) -> dict:
        """
        Compute statistics about attention patterns.
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Dictionary with attention statistics
        """
        # Average attention entropy (measure of attention distribution)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
        avg_entropy = entropy.mean().item()
        
        # Maximum attention weight per position
        max_attention = attention_weights.max(dim=-1)[0].mean().item()
        
        # Attention to diagonal (self-attention to same position)
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        diagonal_idx = torch.arange(seq_len)
        diagonal_attention = attention_weights[:, :, diagonal_idx, diagonal_idx].mean().item()
        
        return {
            'average_entropy': avg_entropy,
            'max_attention_weight': max_attention,
            'diagonal_attention': diagonal_attention,
            'shape': attention_weights.shape
        }
    
    @staticmethod
    def analyze_attention_heads(model: nn.Module, input_text: torch.Tensor) -> dict:
        """
        Analyze what different attention heads learn to focus on.
        
        Args:
            model: Model with attention mechanisms
            input_text: Input token sequence
            
        Returns:
            Analysis results for each attention head
        """
        # This would be implemented based on specific model architecture
        # For now, return a placeholder structure
        return {
            'head_specializations': [],
            'attention_patterns': [],
            'layer_analysis': []
        }


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_attention_mechanisms()
    
    # Additional example: Attention with real-world-like scenario
    print("\n=== Real-world Example ===")
    
    # Simulate a simple sentence with attention
    # "The cat sat on the mat"
    vocab_size = 1000
    d_model = 256
    num_heads = 4
    
    # Create token embeddings (simplified)
    embedding = nn.Embedding(vocab_size, d_model)
    tokens = torch.tensor([[1, 10, 20, 30, 40, 50]])  # Simplified token IDs
    embeddings = embedding(tokens)
    
    # Apply self-attention
    self_attention = SelfAttention(d_model, num_heads)
    output = self_attention(embeddings)
    
    print(f"Sentence tokens shape: {tokens.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Self-attention output shape: {output.shape}")
    print("\nBasic attention mechanism demonstration completed!")
