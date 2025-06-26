"""
Transformer Model Implementation From Scratch

This module implements a complete transformer architecture from scratch,
including all the essential components: multi-head attention, feed-forward
networks, layer normalization, positional encoding, and the full model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    Adds position information to token embeddings using sine and cosine functions.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_seq_length, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        # x shape: [seq_length, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism from 'Attention is All You Need'.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False) 
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional attention mask
            
        Returns:
            (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
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
        
        # Linear transformations and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.W_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Two linear transformations with ReLU activation in between.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward network forward pass."""
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block with self-attention and feed-forward layers.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer block forward pass with residual connections.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GPTModel(nn.Module):
    """
    GPT-style autoregressive transformer model.
    Decoder-only architecture for causal language modeling.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1,
                 pad_token_id: int = 0):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal (lower triangular) mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask to ignore padded tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Padding mask [batch_size, 1, 1, seq_len]
        """
        return (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GPT model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing logits and optionally hidden states
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Scale embeddings by sqrt(d_model) as in original transformer
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Add positional encoding
        # Transpose for positional encoding (expects [seq_len, batch_size, d_model])
        token_embeds = token_embeds.transpose(0, 1)
        hidden_states = self.positional_encoding(token_embeds)
        # Transpose back to [batch_size, seq_len, d_model]
        hidden_states = hidden_states.transpose(0, 1)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = causal_mask * attention_mask
        else:
            # Create padding mask from input_ids
            padding_mask = self.create_padding_mask(input_ids)
            mask = causal_mask * padding_mask
        
        # Pass through transformer blocks
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, mask)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final layer normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Output projection to vocabulary
        logits = self.output_projection(hidden_states)
        
        # Prepare output dictionary
        output = {'logits': logits}
        if return_hidden_states:
            output['hidden_states'] = all_hidden_states
        
        return output
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 do_sample: bool = True,
                 pad_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using the trained model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs['logits']
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('inf'))
                
                # Sample or use greedy decoding
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit the maximum sequence length
                if generated.size(1) >= self.max_seq_length:
                    break
        
        return generated
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gpt_model(config: Dict[str, Any]) -> GPTModel:
    """
    Create a GPT model with the given configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized GPT model
    """
    model = GPTModel(
        vocab_size=config.get('vocab_size', 50000),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        d_ff=config.get('d_ff', 2048),
        max_seq_length=config.get('max_seq_length', 1024),
        dropout=config.get('dropout', 0.1),
        pad_token_id=config.get('pad_token_id', 0)
    )
    
    return model


if __name__ == "__main__":
    # Test the model implementation
    print("Testing GPT Model Implementation...")
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 1024,
        'dropout': 0.1
    }
    
    # Create model
    model = create_gpt_model(config)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {config['vocab_size']}]")
    
    # Test generation
    print("\nTesting text generation...")
    prompt = torch.randint(0, config['vocab_size'], (1, 10))
    generated = model.generate(prompt, max_length=20, temperature=0.8, do_sample=True)
    
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\n✅ Model implementation test completed successfully!")
    print("💡 Next: Proceed to Phase 5 - Train Small Language Model")
