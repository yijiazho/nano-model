import torch
import torch.nn as nn
from torch.nn import functional as F

from utility.tiktoken_tokenizer import TiktokenTokenizer

from nltk.translate.bleu_score import sentence_bleu

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(42)
path = 'modedl/model_tokenizer.pth'

with open('input/tale_of_twin_cities.txt', 'r', encoding='utf-8') as f:
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

# ----------------------------------
# Train

model.load_state_dict(torch.load(path))
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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