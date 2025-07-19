"""
GPT Training Script

This script trains a GPT model from scratch using tiktoken tokenization with GPT-4 tokenizer.
It includes the complete training loop with evaluation, loss tracking, and text generation.

Author: Nazar Korniichuk
Date: 02.07.2025
"""

import torch
from GPT import GPT
from optim import Adam
import time
import os
import tiktoken

# Model hyperparameters
block_size = 512      # Context length
batch_size = 64       # Batch size for training
max_iters = 5000      # Total training iterations
eval_interval = 500   # Evaluate every N iterations
learning_rate = 3e-4  # Learning rate
n_embd = 384         # Embedding dimension
n_heads = 6         # Number of attention heads
n_layers = 6         # Number of transformer layers
dropout = 0.2        # Dropout probability
eval_iters = 200     # Number of iterations for loss estimation

# Device configuration
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA")
    # Set memory fraction to avoid OOM
    torch.cuda.empty_cache()
else:
    device = 'cpu'
    print(f"Using CPU")

torch.set_float32_matmul_precision("high")

# Initialize tiktoken tokenizer
print("Initializing tiktoken GPT-4 tokenizer...")
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    print("Successfully loaded GPT-4 tokenizer")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Falling back to GPT-2 tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

# Data loading
def load_training_data():
    """
    Load training data from file.
    
    Returns:
        str: Raw text data for training
    """
    # Try different possible paths for the input file
    possible_paths = [
        '../data/input.txt',
        './data/input.txt',
        'data/input.txt',
        'input.txt'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    
    # If no file found, create a simple example
    print("No input.txt found, using example text")
    return """
    Hello world! This is a simple example text for training a GPT model.
    The model will learn to predict the next token in a sequence.
    This is useful for language modeling tasks.
    """

def tokenize_text(text):
    """
    Tokenize text using tiktoken.
    
    Args:
        text (str): Raw text to tokenize
        
    Returns:
        list: List of token IDs
    """
    return tokenizer.encode(text)

def detokenize_tokens(token_ids):
    """
    Convert token IDs back to text using tiktoken.
    
    Args:
        token_ids (list): List of token IDs
        
    Returns:
        str: Decoded text
    """
    return tokenizer.decode(token_ids)

def build_dataset_from_tokens(token_ids, train_ratio=0.9):
    """
    Build training and validation datasets from tokenized data.
    
    Args:
        token_ids (list): List of token IDs
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        tuple: (train_data, val_data) as torch tensors
    """
    # Convert to tensor
    data = torch.tensor(token_ids, dtype=torch.long)
    
    # Split into train and validation
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data

# Load and prepare data
text = load_training_data()
token_ids = tokenize_text(text)
train_data, val_data = build_dataset_from_tokens(token_ids)

# Get vocabulary size from tokenizer
vocab_size = tokenizer.n_vocab

print(f"Vocabulary size: {vocab_size}")
print(f"Training data size: {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")
print(f"Sample tokens: {token_ids[:20]}")
print(f"Sample decoded: {detokenize_tokens(token_ids[:20])}")


def get_batch(split):
    """
    Generate a batch of data for training or validation.
    
    Args:
        split (str): Either 'train' or 'val' to specify dataset
        
    Returns:
        tuple: (x, y) where x is input tokens and y is target tokens
    """
    data = train_data if split == 'train' else val_data
    
    # Sample random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input and target sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to device with non_blocking for faster transfer
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimate loss on training and validation sets.
    
    Returns:
        dict: Dictionary with 'train' and 'val' loss values
    """
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
    return out


# Initialize model and optimizer
print("Initializing model...")
model = GPT(vocab_size, n_embd, n_heads, n_layers, block_size, dropout=dropout, device=device)
optimizer = Adam(model.parameters(), lr=learning_rate, device=device)

# Enable gradients for all parameters
for p in model.parameters():
    p.requires_grad = True

# Print model statistics
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

# Training loop
print("\nStarting training...")
start_time = time.time()
start_time_iter = time.time()
last_iter = 0

model.train()

for iter in range(max_iters):
    # Evaluate loss periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}/{max_iters}: "
              f"train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")
        
        # Print timing information
        end_time_iter = time.time()
        print(f"Time for {iter - last_iter} iterations: "
              f"{end_time_iter - start_time_iter:.2f} seconds")
        last_iter = iter
        start_time_iter = time.time()

    # Get training batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training completed
end_time = time.time()
print(f"\nTraining completed!")
print(f"Total training time: {end_time - start_time:.2f} seconds")

# Generate text from the trained model
print("\nGenerating sample text...")
model.eval()

# Start with a simple context (start token)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate text
generated_tokens = model.generate(context, max_new_tokens=500)
decoded_text = detokenize_tokens(generated_tokens[0].tolist())

print("Generated text:")
print("-" * 50)
print(decoded_text)
print("-" * 50)

print("\nTraining script completed successfully!")