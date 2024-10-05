import csv
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from nltk.translate.bleu_score import sentence_bleu

batch_size = 16
epochs = 10
block_size = 16
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
input_path = 'input/tale_of_two_cities.txt'
    
with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()
        

tokenizer.model_max_length = len(text)
model.config.pad_token_id = tokenizer.eos_token_id

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def write_losses_to_file(losses, file_path=loss_file_path):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(losses)

        
# Fine Tuning GPT2
optimizer = AdamW(model.parameters(), lr=5e-5)


for epoch in range(epochs):
    epoch_loss = []
    total_loss = 0

    # Move training data to device
    inputs = train_data.to(device)
    
    for i in range(0, len(train_data), batch_size):
        batch = inputs[i:i + batch_size]
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        
        # Optimize
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_epoch_loss = total_loss / (len(train_data) // batch_size)
    print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
    write_losses_to_file([avg_epoch_loss])

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
print(generate_text("The strong tide, so swift, so deep, and certain,"))

model.eval()
total_loss = 0
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