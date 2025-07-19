"""
LSTM Training Script

Trains an LSTM model for character-level language modeling using modules from module.py.
Follows the structure of TrainGPT.py.
"""

import torch
import torch.nn.functional as F
from module import Embedding, LSTM, Dropout, Linear, Sequential, LayerNorm
from optim import Adam
from dataset import build_vocab, build_dataset
import time
import os

# Model hyperparameters
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_hidden = 512
num_layers = 6
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
    Hello world! This is a simple example text for training an LSTM model.
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

class LSTMModel:
    def __init__(self, vocab_size, n_embd, n_hidden, block_size, dropout, num_layers=2, device='cpu'):
        self.device = device
        self.block_size = block_size
        self.embedding = Embedding(vocab_size, n_embd, device=device)
        self.num_layers = num_layers
        self.lstm_layers = []
        self.norm_layers = []
        for i in range(num_layers):
            input_size = n_embd if i == 0 else n_hidden
            self.lstm_layers.append(LSTM(input_size, n_hidden, device=device))
            self.norm_layers.append(LayerNorm(n_hidden, device=device))
        self.dropout = Dropout(p=dropout, training=True, device=device)
        self.fc = Linear(n_hidden, vocab_size, device=device)
    def parameters(self):
        params = self.embedding.parameters()
        for l, n in zip(self.lstm_layers, self.norm_layers):
            params += l.parameters() + n.parameters()
        params += self.dropout.parameters() + self.fc.parameters()
        return params
    def to(self, device):
        self.device = device
        self.embedding.to(device)
        for l, n in zip(self.lstm_layers, self.norm_layers):
            l.to(device)
            n.to(device)
        self.dropout.to(device)
        self.fc.to(device)
        return self
    def train(self):
        if hasattr(self.dropout, 'train'):
            self.dropout.train()
        return self
    def eval(self):
        if hasattr(self.dropout, 'eval'):
            self.dropout.eval()
        return self
    def __call__(self, idx, targets=None):
        x = self.embedding(idx)  # (B, T, n_embd)
        for lstm, norm in zip(self.lstm_layers, self.norm_layers):
            out = lstm(x)  # (B, T, n_hidden)
            if x.shape == out.shape:
                out = out + x  # Residual connection
            out = norm(out)
            x = out
        x = self.dropout(x)
        x = self.fc(x)  # (B, T, vocab_size)
        B, T, C = x.shape
        if targets is None:
            loss = None
        else:
            logits = x.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return x, loss
    def generate(self, context, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = context[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
        return context

print("Initializing model...")
model = LSTMModel(vocab_size, n_embd, n_hidden, block_size, dropout, num_layers=num_layers, device=device)
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