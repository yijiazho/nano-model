import csv
import random
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from nltk.translate.bleu_score import sentence_bleu

batch_size = 64
epochs = 10
block_size = 1024
sample_size = 10240
eval_interval = 1000
num_iterations = 100
total_bleu_score = 0
total_loss = 0

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

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
tokenized_data = tokenizer.encode(combined_text, return_tensors='pt')[0]

# Select a subset of the tokenized data
sample_size = min(sample_size, len(tokenized_data)) 
sampled_data = tokenized_data[:sample_size]

n = int(0.9 * len(sampled_data))
train_data = sampled_data[:n]
val_data = sampled_data[n:]

def write_losses_to_file(losses, file_path=loss_file_path):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(losses)

def chunk_data(data, block_size=1024):
    for i in range(0, len(data) - block_size + 1, block_size):
        yield data[i: i + block_size]
        
train_data_chunks = list(chunk_data(train_data, block_size))

# Fine Tuning GPT2
optimizer = AdamW(model.parameters(), lr=5e-5)

print(len(sampled_data))

for epoch in range(epochs):
    epoch_loss = []
    random.shuffle(train_data_chunks) 
    for step in range(0, len(train_data_chunks), batch_size):
        batch_chunks = train_data_chunks[step: step + batch_size]
        inputs = torch.stack(batch_chunks).to(device)
        labels = inputs.clone()

        # Forward pass
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        epoch_loss.append(loss.item())
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print loss every few steps
        if step % eval_interval == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    
    write_losses_to_file(epoch_loss)
    print(f"Epoch {epoch} finished with average loss: {sum(epoch_loss) / len(epoch_loss)}")


# Function to generate text using GPT-2
def generate_text(prompt, max_new_tokens=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Get a batch of validation data (X, Y), where X is the context and Y is the target
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (1,))
    x = torch.stack([val_data[i:i+block_size] for i in ix])
    y = torch.stack([val_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

loss_fn = torch.nn.CrossEntropyLoss()

# generate from gpt2
print(generate_text(""))

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