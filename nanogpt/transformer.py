import os
import torch
import torch.nn as nn
from torch.nn import functional as F

BATCH_SIZE = 64  # 32
BLOCK_SIZE = 256  # 8
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4  # 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device used is: {device}')
EVAL_ITERS = 200
N_EMBED = 384  # 32
N_HEADS = 6  # 4
N_LAYERS = 6  # 3
DROPOUT = 0.2  # 0.0

torch.manual_seed(1337)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FILENAME = 'input.txt'
DATA_PATH = os.path.join(DATA_DIR, FILENAME)
# DATA_PATH = 'C:\\src\\forecasting-electricity\\data\\input.txt'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda arr: "".join([itos[i] for i in arr])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
data_train, data_val = data[:n], data[n:]


def get_batch(split):
    data = data_train if split == "train" else data_val
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size)
        self.query = nn.Linear(N_EMBED, head_size)
        self.value = nn.Linear(N_EMBED, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, HEAD_SIZE)
        q = self.query(x)  # (B, T, HEAD_SIZE)
        v = self.value(x)  # (B, T, HEAD_SIZE)
        weights = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T) -> w_{i, j} is the response of k_j for query q_i -> only lower triangular: q_{1, 3} doesn't make sense because it would be the impact of the 3rd position on the 1st position.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHead(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))  # B, T, C
        x = x + self.ffwd(self.ln2(x))  # B, T, C
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.block = nn.Sequential(*[Block(N_EMBED, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBED)  # Final layer norm
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # (B, T, N_EMBED)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, N_EMBED)
        x = token_emb + pos_emb  # (B, T, N_EMBED)
        x = self.block(x)  # B, T, C
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]  # B, T, C (max C) Otherwise self.position_embedding will fail
            logits, _ = self(idx_cond) # (B, T, C) 
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
    
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

input = torch.zeros((1, 1), dtype=torch.long)
print("\n\n\nEvaluation:\n")
print(decode(model.generate(input, max_new_tokens=100)[0].tolist()))
