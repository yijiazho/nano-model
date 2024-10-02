import csv
import random
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from nltk.translate.bleu_score import sentence_bleu

from utility.tiktoken_tokenizer import TiktokenTokenizer

batch_size = 64
epochs = 3
sample_size = 500000
eval_interval = 1000
block_size = 8
num_iterations = 100
total_bleu_score = 0
total_loss = 0

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = TiktokenTokenizer(model="gpt-2")

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.train()


torch.manual_seed(42)
loss_file_path = 'results/combined_losses.csv'
path = 'model/gpt2.pth'
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
combined_loss = []
for file_path in input_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        combined_text += f.read() + "\n"
        

# Train and test splits
data = torch.tensor(tokenizer.encode(combined_text), dtype=torch.long)
random_indices = random.sample(range(len(data)), sample_size)
sampled_data = data[random_indices]
n = int(0.9 * len(sampled_data))
train_data = sampled_data[:n]
val_data = sampled_data[n:]

def write_losses_to_file(losses, file_path="combined_loss.csv"):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(losses)
        
def get_data_chunks(data, chunk_size=100000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
        
# Fine Tuning GPT2
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(epochs):
    for step in range(0, len(train_data), batch_size):
        inputs = train_data[step: step + batch_size]
        labels = inputs.clone()

        # Forward pass
        outputs = model(inputs.to(device), labels=labels.to(device))
        loss = outputs.loss

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# Function to generate text using GPT-2
def generate_text(prompt, max_new_tokens=100):
    # Encode the prompt (tokenize)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
    
    # Decode generated output to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Get a batch of validation data (X, Y), where X is the context and Y is the target
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (1,))
    x = torch.stack([val_data[i:i+block_size] for i in ix])
    y = torch.stack([val_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

loss_fn = torch.nn.CrossEntropyLoss()

model.eval()
for _ in range(num_iterations):
    context, reference_sequence = get_val_batch()

    with torch.no_grad():
        outputs = model(context, labels=reference_sequence)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = reference_sequence[..., 1:].contiguous()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    total_loss += loss.item()

    # Generate text for BLEU evaluation
    generated_sequence = model.generate(context, max_new_tokens=block_size - 1)[0].tolist()

    generated_text = tokenizer.decode(generated_sequence)
    reference_text = tokenizer.decode(reference_sequence[0].tolist())

    generated_tokenized = tokenizer.encode(generated_text)
    reference_tokenized = tokenizer.encode(reference_text)
    reference = [reference_tokenized]
    candidate = generated_tokenized

    bleu_score = sentence_bleu(reference, candidate)
    total_bleu_score += bleu_score

average_bleu_score = total_bleu_score / num_iterations
average_loss = total_loss / num_iterations

print(f"Average BLEU score over {num_iterations} samples: {average_bleu_score}")
print(f"Average loss over {num_iterations} samples: {average_loss}")

torch.save(model.state_dict(), path)
# GPT2 level is about 0.413