import torch
import torch.nn as nn
from torch.nn import functional as F
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from utility.tiktoken_tokenizer import TiktokenTokenizer

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_new_tokens = 50  # For BLEU score evaluation
bleu_subset_size = 10 # Batch numers to parallel evaluate BLEU
# ------------

torch.manual_seed(42)

with open('input1.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
tokenizer = TiktokenTokenizer()
vocab_size = tokenizer.vocab_size()

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss_and_bleu():
    out = {}
    model.eval()
    smooth_func = SmoothingFunction().method1
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        bleu_scores = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

            predictions = model.generate(X[:, :1], max_new_tokens)  # Generate based on the first token in X

            # Subset of BLEU calculation (reduce batch size to speed up BLEU calculation)
            bleu_subset = min(bleu_subset_size, X.size(0))  # Limit BLEU calculation to a subset
            references = [[Y[i].cpu().tolist()] for i in range(bleu_subset)]
            hypotheses = [predictions[i].cpu().tolist() for i in range(bleu_subset)]
            
            # Calculate BLEU for the subset of sequences in the batch
            if hypotheses and references:
                bleu_scores[k] = corpus_bleu(references, hypotheses, smoothing_function=smooth_func)
                print(f"evaluation {k} times")

        print('output......')
        out[split] = {
            'loss': losses.mean(),
            'bleu': bleu_scores.mean(),
        }
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=256):
        super().__init__()
        # Input -> Projection -> output
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B, T) tensor of token indices
        embeddings = self.token_embedding_table(idx)  # (B, T, embedding_dim)
        logits = self.output_layer(embeddings)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        results = estimate_loss_and_bleu()
        train_loss, train_bleu = results['train']['loss'], results['train']['bleu']
        val_loss, val_bleu = results['val']['loss'], results['val']['bleu']
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train BLEU {train_bleu:.4f}, val BLEU {val_bleu:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
