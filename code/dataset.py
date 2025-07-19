"""
Dataset Utilities for Language Modeling

This module provides utilities for building vocabularies and datasets
for character-level language modeling tasks.

Author: Nazar Korniichuk
Date: 02.07.2025
"""

import torch


def build_vocab(words):
    """
    Build vocabulary from a list of words.
    
    Creates character-level vocabulary with special token for end-of-sequence.
    Returns encoding/decoding functions and vocabulary statistics.
    
    Args:
        words (list): List of words to build vocabulary from
        
    Returns:
        tuple: (vocab_size, stoi, itos, encode, decode) where:
            - vocab_size (int): Size of the vocabulary
            - stoi (dict): String to integer mapping
            - itos (dict): Integer to string mapping
            - encode (function): Function to encode strings to token lists
            - decode (function): Function to decode token lists to strings
    """
    # Get unique characters from all words
    chars = sorted(list(set(''.join(words))))
    
    # Create string to integer mapping (starting from 1)
    stoi = {ch: i+1 for i, ch in enumerate(chars)}
    itos = {i+1: ch for i, ch in enumerate(chars)}
    
    # Add special token for end-of-sequence (index 0)
    stoi['.'] = 0
    itos[0] = '.'
    
    vocab_size = len(itos)
    
    # Define encoding and decoding functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return vocab_size, stoi, itos, encode, decode


def build_dataset(text, encode):
    """
    Build training and validation datasets from text.
    
    Splits the encoded text into training (90%) and validation (10%) sets.
    Uses torch.tensor with dtype specification for better memory efficiency.
    
    Args:
        text (str): Raw text to encode and split
        encode (function): Function to encode text to token indices
        
    Returns:
        tuple: (train_data, val_data) where both are torch.Tensor objects
    """
    # Encode text to token indices
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split into train (90%) and validation (10%) sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data