import torch
import torch.nn.functional as F
from module import Linear, ReLU, Sequential, LayerNorm, Embedding

#SelfAttention layer
class SelfAttention:
    def __init__(self, n_embd, head_size):
        self.n_embd = n_embd
        self.head_size = head_size
        self.query = Linear(n_embd, head_size, bias=False)
        self.key = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size, bias=False)

    def parameters(self):
        return self.query.parameters() + self.key.parameters() + self.value.parameters()
    
    def __call__(self, x):
        #input shape: (N, L, H_in)
        #output shape: (N, L, H_out)
        N, L, _ = x.size()
        H_out = self.head_size
        q = self.query(x) #(N, L, H_out)
        k = self.key(x) #(N, L, H_out)
        v = self.value(x) #(N, L, H_out)
        
        wei = q @ k.transpose(-2, -1) * H_out**-0.5 #(N, L, L)
        tril = torch.tril(torch.ones(L, L)) #(L, L)
        wei = wei.masked_fill(tril == 0, float('-inf')) #(N, L, L)
        wei = F.softmax(wei, dim=-1) #(N, L, L)
        
        out = wei @ v #(N, L, H_out)
        return out
    
    def parameters(self):
        return self.query.parameters() + self.key.parameters() + self.value.parameters()

#MultiHeadAttention layer
class MultiHeadAttention:
    def __init__(self, n_embd, head_size, n_heads):
        self.n_embd = n_embd
        self.head_size = head_size
        self.n_heads = n_heads
        self.heads = [SelfAttention(n_embd, head_size) for _ in range(n_heads)]
        self.project = Linear(head_size * n_heads, n_embd)
    
    def parameters(self):
        return [param for head in self.heads for param in head.parameters()] + self.project.parameters()
    
    def __call__(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.project(out)
        return out

#FeedForward layer
class FeedForward:
    def __init__(self, n_embd):
        self.n_embd = n_embd
        self.net = Sequential(
            [
            Linear(n_embd, 4*n_embd, bias=True),
            ReLU(),
            Linear(4*n_embd, n_embd, bias=True)
            ]
        )
    
    def parameters(self):
        return self.net.parameters()
    
    def __call__(self, x):
        return self.net(x)

#ResidualTransformerBlock
class ResidualTransformerBlock:
    def __init__(self, n_embd, n_head):
        self.n_embd = n_embd
        head_size = n_embd // n_head
        self.n_head = n_head
        self.attention = MultiHeadAttention(n_embd, head_size, n_head)
        self.norm1 = LayerNorm(n_embd)
        self.feedforward = FeedForward(n_embd)
        self.norm2 = LayerNorm(n_embd)
    
    def parameters(self):
        return self.attention.parameters() + self.norm1.parameters() + self.norm2.parameters() + self.feedforward.parameters()
    
    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x
    
#Token + position embedding
class TokenEmbedding:
    def __init__(self, vocab_size, n_embd, block_size):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.token_embedding_table = Embedding(vocab_size, n_embd)
        self.position_embedding_table = Embedding(block_size, n_embd)

    def parameters(self):
        return self.token_embedding_table.parameters() + self.position_embedding_table .parameters()
    
    def __call__(self, idx):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)

        return tok_emb + pos_emb
    
#GPT model
class GPT:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        self.token_embedding = TokenEmbedding(vocab_size, n_embd, block_size)
        self.lm_head = Linear(n_embd, vocab_size)
        self.layernorm = LayerNorm(n_embd)
        self.transformer_blocks = Sequential([ResidualTransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
    
    def parameters(self):
        return self.token_embedding.parameters() + self.lm_head.parameters() + self.transformer_blocks.parameters()
    
    def __call__(self, idx, targets=None):
        x = self.token_embedding(idx)
        x = self.transformer_blocks(x)
        x = self.layernorm(x)
        x = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x, targets)
        return (x, loss)

    def generate(self, context, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(context[:, -self.block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
        return context

