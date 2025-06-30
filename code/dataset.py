import torch

def build_vocab(words):
    chars = sorted(list(set(''.join(words))))
    # create a mapping from characters to integers
    stoi = {ch:i+1 for i, ch in enumerate(chars)}
    itos = {i+1:ch for i, ch in enumerate(chars)}
    stoi['.'] = 0
    itos[0] = '.'
    vocab_size = len(itos)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return vocab_size, stoi, itos, encode, decode

def build_dataset(text, encode):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data