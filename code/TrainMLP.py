"""
MLP Training Script

Trains an MLP model for character-level language modeling using modules from module.py.
Follows the structure of TrainGPT.py.
"""

import torch
import torch.nn.functional as F
from module import Embedding, Linear, ReLU, Dropout, Sequential, Flatten, LayerNorm
from optim import Adam
from dataset import build_vocab, build_dataset
import time
import os

# Model hyperparameters
block_size = 128
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_layers = 4  # Number of hidden layers
hidden_dim = 256  # Hidden layer size
dropout = 0.2
eval_iters = 200

if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA")
    torch.cuda.empty_cache()
else:
    device = 'cpu'
    print(f"Using CPU")

def load_training_data():
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
    print("No input.txt found, using example text")
    return """
    Hello world! This is a simple example text for training an MLP model.
    The model will learn to predict the next character in a sequence.
    """

text = load_training_data()
vocab_size, stoi, itos, encode, decode = build_vocab(text)
train_data, val_data = build_dataset(text, encode)

print(f"Vocabulary size: {vocab_size}")
print(f"Training data size: {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

class MLPModel:
    def __init__(self, vocab_size, n_embd, hidden_dim, n_layers, block_size, dropout, device='cpu'):
        self.device = device
        self.block_size = block_size
        self.embedding = Embedding(vocab_size, n_embd, device=device)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.norm_layers = []
        layers = [Flatten(start_dim=1, device=device)]
        in_dim = n_embd * block_size
        for i in range(n_layers):
            layers.append(Linear(in_dim, hidden_dim, device=device))
            layers.append(ReLU(device=device))
            self.norm_layers.append(LayerNorm(hidden_dim, device=device))
            layers.append(Dropout(p=dropout, training=True, device=device))
            in_dim = hidden_dim
        layers.append(Linear(hidden_dim, vocab_size, device=device))  # Output only vocab_size logits
        self.mlp = Sequential(layers, device=device)
    def parameters(self):
        params = self.embedding.parameters()
        for n in self.norm_layers:
            params += n.parameters()
        params += self.mlp.parameters()
        return params
    def to(self, device):
        self.device = device
        self.embedding.to(device)
        for n in self.norm_layers:
            n.to(device)
        self.mlp.to(device)
        return self
    def train(self):
        for layer in self.mlp.layers:
            if hasattr(layer, 'train'):
                layer.train()
        return self
    def eval(self):
        for layer in self.mlp.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        return self
    def __call__(self, idx, targets=None):
        x = self.embedding(idx)  # (B, T, n_embd)
        x = x.view(x.shape[0], -1)  # Flatten all except batch
        norm_idx = 0
        for i, layer in enumerate(self.mlp.layers):
            prev_x = x
            x = layer(x)
            # After each Linear+ReLU+Dropout, apply residual+LayerNorm if dims match
            if isinstance(layer, Dropout) and norm_idx < len(self.norm_layers):
                if prev_x.shape == x.shape:
                    x = x + prev_x
                x = self.norm_layers[norm_idx](x)
                norm_idx += 1
        logits = x  # (B, vocab_size)
        if targets is None:
            loss = None
        else:
            last_targets = targets[:, -1]  # (B,)
            loss = F.cross_entropy(logits, last_targets)
        return logits, loss
    def generate(self, context, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = context[:, -self.block_size:]
            # Pad if needed
            if idx_cond.shape[1] < self.block_size:
                pad_len = self.block_size - idx_cond.shape[1]
                pad = torch.zeros((idx_cond.shape[0], pad_len), dtype=idx_cond.dtype, device=idx_cond.device)
                idx_cond = torch.cat([pad, idx_cond], dim=1)
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
        return context

print("Initializing model...")
model = MLPModel(vocab_size, n_embd, hidden_dim, n_layers, block_size, dropout, device=device)
optimizer = Adam(model.parameters(), lr=learning_rate, device=device)
for p in model.parameters():
    p.requires_grad = True
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

@torch.no_grad()
def estimate_loss():
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

print("\nStarting training...")
start_time = time.time()
start_time_iter = time.time()
last_iter = 0
model.train()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}/{max_iters}: "
              f"train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")
        end_time_iter = time.time()
        print(f"Time for {iter - last_iter} iterations: "
              f"{end_time_iter - start_time_iter:.2f} seconds")
        last_iter = iter
        start_time_iter = time.time()
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end_time = time.time()
print(f"\nTraining completed!")
print(f"Total training time: {end_time - start_time:.2f} seconds")

print("\nGenerating sample text...")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=500)
decoded_text = decode(generated_text[0].tolist())
print("Generated text:")
print("-" * 50)
print(decoded_text)
print("-" * 50)
print("\nTraining script completed successfully!") 