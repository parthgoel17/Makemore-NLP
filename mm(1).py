import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 32  
BLOCK_SIZE = 2   
MAX_ITERS = 20000
EVAL_INTERVAL = 2000
LEARNING_RATE = 1e-2
DEVICE = 'cpu'
EVAL_ITERS = 2000


torch.manual_seed(42)

with open('names.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takemlist of integers, outputstring

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]


def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) # (B,T,C)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, _ = self(idx)     # get predictions
            logits = logits[:, -1, :] # take last token of size (B, C)
            probs = F.softmax(logits, dim=-1) # create probability of size (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution of size (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


for iter in range(MAX_ITERS):

    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')


    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, DEVICE=DEVICE)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))