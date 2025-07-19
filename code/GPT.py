"""
GPT (Generative Pre-trained Transformer) Implementation

This module implements a complete GPT model from scratch, including:
- Self-attention mechanism with causal masking
- Multi-head attention
- Feed-forward networks
- Transformer blocks with residual connections
- Token and position embeddings
- Complete GPT model with generation capabilities

Author: Nazar Korniichuk
Date: 02.07.2025
"""

import torch
import torch.nn.functional as F
from module import Linear, ReLU, Sequential, LayerNorm, Embedding, Dropout

#SelfAttention layer
class SelfAttention:
    """
    Self-attention mechanism with causal masking for autoregressive language modeling.
    
    This implements scaled dot-product attention with optional dropout and causal
    masking to ensure each position can only attend to previous positions.
    
    Args:
        n_embd (int): Input embedding dimension
        head_size (int): Size of each attention head
        dropout (float): Dropout probability (default: 0.1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_embd, head_size, dropout=0.1, device='cpu'):
        self.device = device
        self.n_embd = n_embd
        self.head_size = head_size
        
        # Linear projections for query, key, and value
        self.query = Linear(n_embd, head_size, bias=False, device=device)
        self.key = Linear(n_embd, head_size, bias=False, device=device)
        self.value = Linear(n_embd, head_size, bias=False, device=device)
        
        # Cache the causal mask to avoid recreating it every forward pass
        self.causal_mask = None
        self.dropout = Dropout(p=dropout, training=True, device=device)

    def parameters(self):
        """Return all trainable parameters."""
        return self.query.parameters() + self.key.parameters() + self.value.parameters()
    
    def eval(self):
        """Set the module to evaluation mode."""
        self.dropout.training = False
        return self
    
    def train(self):
        """Set the module to training mode."""
        self.dropout.training = True
        return self
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.query.to(device)
        self.key.to(device)
        self.value.to(device)
        self.dropout.to(device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(device)
        return self
    
    def __call__(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, head_size)
        """
        # Input shape: (N, L, H_in) -> Output shape: (N, L, H_out)
        N, L, _ = x.size()
        H_out = self.head_size
        
        # Compute query, key, and value projections
        q = self.query(x)  # (N, L, H_out)
        k = self.key(x)    # (N, L, H_out)
        v = self.value(x)  # (N, L, H_out)
        
        # Compute attention scores with scaling
        wei = q @ k.transpose(-2, -1) * H_out**-0.5  # (N, L, L)
        
        # Create causal mask only once and cache it
        if self.causal_mask is None or self.causal_mask.size(0) != L:
            self.causal_mask = torch.tril(torch.ones(L, L, device=self.device))  # (L, L)
        
        # Apply causal masking (set future positions to -inf)
        wei = wei.masked_fill(self.causal_mask == 0, float('-inf'))  # (N, L, L)
        
        # Apply softmax and dropout
        wei = F.softmax(wei, dim=-1)  # (N, L, L)
        wei = self.dropout(wei)
        
        # Apply attention weights to values
        out = wei @ v  # (N, L, H_out)
        
        return out

#MultiHeadAttention layer
class MultiHeadAttention:
    """
    Multi-head attention mechanism.
    
    Combines multiple self-attention heads and projects the concatenated output
    back to the original embedding dimension.
    
    Args:
        n_embd (int): Input embedding dimension
        head_size (int): Size of each attention head
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_embd, head_size, n_heads, dropout=0.1, device='cpu'):
        self.device = device
        self.n_embd = n_embd
        self.head_size = head_size
        self.n_heads = n_heads
        
        # Create multiple attention heads
        self.heads = [SelfAttention(n_embd, head_size, dropout=dropout, device=device) 
                     for _ in range(n_heads)]
        
        # Output projection layer
        self.project = Linear(head_size * n_heads, n_embd, device=device)
        self.dropout = Dropout(p=dropout, training=True, device=device)
    
    def eval(self):
        """Set the module to evaluation mode."""
        for head in self.heads:
            head.eval()
        self.dropout.training = False
        return self
    
    def train(self):
        """Set the module to training mode."""
        for head in self.heads:
            head.train()
        self.dropout.training = True
        return self
    
    def parameters(self):
        """Return all trainable parameters."""
        return [param for head in self.heads for param in head.parameters()] + self.project.parameters()
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.dropout.to(device)
        for head in self.heads:
            head.to(device)
        self.project.to(device)
        return self
    
    def __call__(self, x):
        """
        Forward pass of multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Process through each attention head
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        
        # Concatenate outputs from all heads
        out = torch.cat(outputs, dim=-1)
        
        # Project back to original embedding dimension
        out = self.project(out)
        out = self.dropout(out)
        
        return out

#FeedForward layer
class FeedForward:
    """
    Feed-forward network with residual connection.
    
    Implements a two-layer feed-forward network with ReLU activation and dropout.
    The hidden dimension is typically 4x the input dimension.
    
    Args:
        n_embd (int): Input and output embedding dimension
        dropout (float): Dropout probability (default: 0.1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_embd, dropout=0.1, device='cpu'):
        self.device = device
        self.n_embd = n_embd
        
        # Feed-forward network: Linear -> ReLU -> Linear -> Dropout
        self.net = Sequential([
            Linear(n_embd, 4 * n_embd, bias=True, device=device),
            ReLU(device=device),
            Linear(4 * n_embd, n_embd, bias=True, device=device),
            Dropout(p=dropout, training=True, device=device)
        ], device=device)
    
    def eval(self):
        """Set the module to evaluation mode."""
        self.net.layers[-1].eval()
        return self
    
    def train(self):
        """Set the module to training mode."""
        self.net.layers[-1].train()
        return self
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.net.to(device)
        return self
    
    def parameters(self):
        """Return all trainable parameters."""
        return self.net.parameters()
    
    def __call__(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        return self.net(x)

#ResidualTransformerBlock
class ResidualTransformerBlock:
    """
    Transformer block with residual connections.
    
    Implements a complete transformer block with:
    - Multi-head self-attention with residual connection
    - Feed-forward network with residual connection
    - Layer normalization before each sub-layer
    
    Args:
        n_embd (int): Embedding dimension
        n_head (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_embd, n_head, dropout=0.1, device='cpu'):
        self.device = device
        self.n_embd = n_embd
        head_size = n_embd // n_head
        self.n_head = n_head
        
        # Sub-layers
        self.attention = MultiHeadAttention(n_embd, head_size, n_head, dropout=dropout, device=device)
        self.norm1 = LayerNorm(n_embd, device=device)
        self.feedforward = FeedForward(n_embd, dropout=dropout, device=device)
        self.norm2 = LayerNorm(n_embd, device=device)
    
    def eval(self):
        """Set the module to evaluation mode."""
        self.attention.eval()
        self.feedforward.eval()
        return self
    
    def train(self):
        """Set the module to training mode."""
        self.attention.train()
        self.feedforward.train()
        return self
    
    def parameters(self):
        """Return all trainable parameters."""
        return (self.attention.parameters() + self.norm1.parameters() + 
                self.norm2.parameters() + self.feedforward.parameters())
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.attention.to(device)
        self.norm1.to(device)
        self.norm2.to(device)
        self.feedforward.to(device)
        return self
    
    def __call__(self, x):
        """
        Forward pass of transformer block with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Attention sub-layer with residual connection
        x = x + self.attention(self.norm1(x))
        
        # Feed-forward sub-layer with residual connection
        x = x + self.feedforward(self.norm2(x))
        
        return x
    
#Token + position embedding
class TokenEmbedding:
    """
    Token and position embedding layer.
    
    Combines token embeddings with learned position embeddings to provide
    the model with both semantic and positional information.
    
    Args:
        vocab_size (int): Size of the vocabulary
        n_embd (int): Embedding dimension
        block_size (int): Maximum sequence length
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, vocab_size, n_embd, block_size, device='cpu'):
        self.device = device
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Embedding tables
        self.token_embedding_table = Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = Embedding(block_size, n_embd, device=device)
        
        # Pre-compute position indices to avoid repeated arange calls
        self.pos_indices = torch.arange(block_size, device=device)

    def parameters(self):
        """Return all trainable parameters."""
        return self.token_embedding_table.parameters() + self.position_embedding_table.parameters()
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.token_embedding_table.to(device)
        self.position_embedding_table.to(device)
        if self.pos_indices is not None:
            self.pos_indices = self.pos_indices.to(device)
        return self
    
    def __call__(self, idx):
        """
        Forward pass of token and position embedding.
        
        Args:
            idx (torch.Tensor): Token indices of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Combined embeddings of shape (batch_size, seq_len, n_embd)
        """
        B, T = idx.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        
        # Get position embeddings
        pos_emb = self.position_embedding_table(self.pos_indices[:T])  # (T, n_embd)
        
        # Add token and position embeddings
        return tok_emb + pos_emb
    
#GPT model
class GPT:
    """
    Generative Pre-trained Transformer (GPT) model.
    
    A complete implementation of the GPT architecture for autoregressive
    language modeling with generation capabilities.
    
    Args:
        vocab_size (int): Size of the vocabulary
        n_embd (int): Embedding dimension
        n_head (int): Number of attention heads
        n_layer (int): Number of transformer layers
        block_size (int): Maximum sequence length
        dropout (float): Dropout probability (default: 0.1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1, device='cpu'):
        # Model components
        self.token_embedding = TokenEmbedding(vocab_size, n_embd, block_size, device=device)
        self.lm_head = Linear(n_embd, vocab_size, device=device)
        self.layernorm = LayerNorm(n_embd, device=device)
        
        # Stack of transformer blocks
        self.transformer_blocks = Sequential([
            ResidualTransformerBlock(n_embd, n_head, dropout=dropout, device=device) 
            for _ in range(n_layer)
        ], device=device)
        
        # Model configuration
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.device = device
    
    def eval(self):
        """Set the model to evaluation mode."""
        for block in self.transformer_blocks.layers:
            block.eval()
        return self
    
    def train(self):
        """Set the model to training mode."""
        for block in self.transformer_blocks.layers:
            block.train()
        return self
    
    def parameters(self):
        """Return all trainable parameters."""
        return (self.token_embedding.parameters() + self.lm_head.parameters() + 
                self.transformer_blocks.parameters())
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.token_embedding.to(device)
        self.lm_head.to(device)
        self.layernorm.to(device)
        self.transformer_blocks.to(device)
        return self
    
    def __call__(self, idx, targets=None):
        """
        Forward pass of the GPT model.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
            targets (torch.Tensor, optional): Target token indices for loss computation
            
        Returns:
            tuple: (logits, loss) where loss is None if targets is None
        """
        # Get token and position embeddings
        x = self.token_embedding(idx)
        
        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        
        # Final layer normalization
        x = self.layernorm(x)
        
        # Language model head
        x = self.lm_head(x)
        
        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(x, targets)
        
        return (x, loss)

    def generate(self, context, max_new_tokens):
        """
        Generate new tokens autoregressively.
        
        Args:
            context (torch.Tensor): Initial context tokens of shape (batch_size, seq_len)
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            torch.Tensor: Generated sequence including context
        """
        # Pre-allocate output tensor for better memory efficiency
        B, T = context.shape
        new_tokens = torch.zeros(B, max_new_tokens, dtype=context.dtype, device=self.device)
        
        for i in range(max_new_tokens):
            # Use the last block_size tokens for efficiency
            context_window = context[:, -self.block_size:]
            
            # Get logits for next token
            logits, loss = self(context_window)
            logits = logits[:, -1, :]  # Get last token logits
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            new_tokens[:, i] = idx_next.squeeze(-1)
            
            # Append to context
            context = torch.cat((context, idx_next), dim=1)
        
        return context

