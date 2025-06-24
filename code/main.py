import torch
import matplotlib.pyplot as plt # for making figures
import torch.nn.functional as F
import time
from model import *

block_size = 3

#get all words from names.txt
words = open('../data/names.txt', 'r').read().splitlines()


#get all the unique characters in the words
chars = sorted(list(set(''.join(words))))

# create a mapping from characters to integers
stoi = {ch:i+1 for i, ch in enumerate(chars)}
itos = {i+1:ch for i, ch in enumerate(chars)}
stoi['.'] = 0
itos[0] = '.'
vocab_size = len(itos)


#build the dataset
def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return (X, Y)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%



batch_size = 2000
max_iters = 3000
learning_rate = 1e-2
n_embd = 10
n_hidden = 10
n_input = block_size * n_embd
n_output = vocab_size


#define the model
model = Sequential(
    [Embedding(vocab_size, n_embd),
    Linear(n_embd, n_hidden), Tanh(),
    Linear(n_hidden, n_output)]
)

#define the model
C = torch.randn(vocab_size, n_embd)
w1 = torch.randn(n_input, n_hidden)
w2 = torch.randn(n_hidden, n_output)
b1 = torch.randn(n_hidden)
b2 = torch.randn(n_output)

parameters = [C, w1, w2, b1, b2]

for p in parameters:
    p.requires_grad = True
    
start_time = time.time()
for iter in range(max_iters):
    idx = torch.randint(0, Xtr.shape[0], (batch_size,))

    x_batch = Xtr[idx]
    y_batch = Ytr[idx]

    #flatten the batch
    x_batch = C[x_batch].view(x_batch.shape[0], -1)


    #forward pass
    h1 = x_batch @ w1 + b1
    h1 = torch.tanh(h1)
    h2 = h1 @ w2 + b2

    #calculate the loss
    loss = F.cross_entropy(h2, y_batch)

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update the parameters
    for p in parameters:
        p.data -= learning_rate * p.grad
        
    
    if iter % 100 == 0:
        print(f"iter {iter}, loss {loss.item()}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"The for loop took {elapsed_time:.4f} seconds to run.")
