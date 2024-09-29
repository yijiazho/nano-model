import torch
import torch.nn as nn
from torch.nn import functional as F

from utility.tiktoken_tokenizer import TiktokenTokenizer

from nltk.translate.bleu_score import sentence_bleu

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel
block_size = 16 # what is the maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 300
embedding_dim = 288
n_head = 12
n_layer = 12
dropout = 0.2
weight_decay = 1e-4

# Hyperparameters for early stopping and learning rate scheduler
scheduler_patience = 5
early_stopping_patience = 10
best_val_loss = float('inf')  # Initialize best loss as infinity
num_bad_epochs = 0  # Tracks the number of epochs without improvement

# ------------

torch.manual_seed(42)
path = 'model/model_schedule.pth'
input_paths = [
    'input/tale_of_two_cities.txt',
    'input/david_copperfield.txt',
    'input/great_expectations.txt',
    'input/war_and_peace.txt',
    'input/les_miserables.txt',
    'input/the_three_musketeers.txt',
    'input/the_count_of_monte_cristo.txt'
]

combined_text = ""

# Loop through each file and read its content, concatenating it to the combined_text
for file_path in input_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        combined_text += f.read() + "\n"

tokenizer = TiktokenTokenizer()
vocab_size = tokenizer.vocab_size()

# Train and test splits
data = torch.tensor(tokenizer.encode(combined_text), dtype=torch.long)
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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention weight
        w = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = w @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
# not that simple model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Input -> Projection -> Output
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_dim) # final layer norm
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B, T) tensor of token indices
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
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
# ----------------------------------
# Train

# model.load_state_dict(torch.load(path))
m = model.to(device)

# create a PyTorch optimizer and shceduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=scheduler_patience)


# Training loop
for iter in range(max_iters):

    # Evaluate loss on train and validation set every eval_interval steps
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        val_loss = losses['val']
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}")

        # Scheduler step: reduce LR if validation loss hasn't improved for 'scheduler patience'
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_bad_epochs = 0  # Reset counter if validation loss improves
            torch.save(model.state_dict(), 'model/best_model.pth')  # Save the best model
        else:
            num_bad_epochs += 1

        # Check if we should stop early
        if num_bad_epochs >= early_stopping_patience:
            print(f"Early stopping at step {iter}, no improvement in validation loss for {early_stopping_patience} intervals.")
            break  # Stop training

    # Sample a batch of data and perform training
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# If early stopping was triggered, load the best model before continuing to further evaluation or inference
model.load_state_dict(torch.load('model/best_model.pth'))

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))

torch.save(model.state_dict(), path)

# -----------------------------------
# Evaluation
# Load
model.load_state_dict(torch.load(path))
model.to(device)
model.eval()

# Get a batch of validation data (X, Y), where X is the context and Y is the target
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (1,))
    x = torch.stack([val_data[i:i+block_size] for i in ix])
    y = torch.stack([val_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

num_iterations = 100  # You can adjust this number based on how many samples you want to evaluate
total_bleu_score = 0

for _ in range(num_iterations):
    # Fetch a single example from the validation set
    context, reference_sequence = get_val_batch()

    # Limit the number of tokens generated to match reference size (i.e., block_size)
    generated_sequence = model.generate(context, max_new_tokens=block_size - 1)[0].tolist()

    # Decode the generated sequence and the reference sequence
    generated_text = tokenizer.decode(generated_sequence)
    reference_text = tokenizer.decode(reference_sequence[0].tolist())  # Decode first reference in batch

    # Tokenize both the generated text and the reference text for BLEU score calculation
    generated_tokenized = tokenizer.encode(generated_text)
    reference_tokenized = tokenizer.encode(reference_text)

    # BLEU score requires a list of reference tokenized sequences
    reference = [reference_tokenized]  # Reference should be a list of lists
    candidate = generated_tokenized  # Candidate is the generated tokenized sequence

    # Calculate BLEU score for this sample
    bleu_score = sentence_bleu(reference, candidate)
    
    # Accumulate BLEU score
    total_bleu_score += bleu_score

# Calculate the average BLEU score over all iterations
average_bleu_score = total_bleu_score / num_iterations
print(f"Average BLEU score over {num_iterations} samples: {average_bleu_score}")