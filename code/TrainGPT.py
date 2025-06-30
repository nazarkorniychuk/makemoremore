import torch
from GPT import GPT
from optim import Adam
from dataset import build_vocab, build_dataset
import time

block_size = 128
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 40
n_heads = 4
n_layers = 4
eval_iters = 200
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA")
else:
    device = 'cpu'
    print(f"Using CPU")



with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size, stoi, itos, encode, decode = build_vocab(text)

train_data, val_data = build_dataset(text, encode)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


model = GPT(vocab_size, n_embd, n_heads, n_layers, block_size)
optimizer = Adam(model.parameters())

for p in model.parameters():
    p.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
print(f"The total number of parameters is: {total_params}")


start_time = time.time()
start_time_iter = time.time()
last_iter = 0
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        end_time_iter = time.time()
        print(f"Time taken for doing {iter - last_iter} iterations: {end_time_iter - start_time_iter:.2f} seconds")
        last_iter = iter
        start_time_iter = time.time()

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


end_time = time.time()
print(f"Overall time taken: {end_time - start_time:.2f} seconds")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))