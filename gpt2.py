import random
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu


# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

# Train and test splits
with open('input/tale_of_twin_cities.txt', 'r', encoding='utf-8') as f:
    text = f.read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Function to generate text using GPT-2
def generate_text(prompt, max_new_tokens=50):
    # Encode the prompt (tokenize)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,  # Pad with EOS token if needed
            do_sample=True,  # Sampling to generate diverse text
            top_k=50,        # Limit sampling to top-k tokens
            top_p=0.95,      # Use nucleus sampling
            temperature=1.0  # Adjust sampling temperature (higher for more randomness)
        )
    
    # Decode generated output to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

block_size = 8
# Get a batch of validation data (X, Y), where X is the context and Y is the target
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (1,))
    x = torch.stack([val_data[i:i+block_size] for i in ix])
    y = torch.stack([val_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

loss_fn = torch.nn.CrossEntropyLoss()

num_iterations = 100
total_bleu_score = 0
total_loss = 0

for _ in range(num_iterations):
    # Get validation batch
    context, reference_sequence = get_val_batch()
    
    # Get logits from the model instead of just generating text
    with torch.no_grad():
        outputs = model(context, labels=reference_sequence)
        logits = outputs.logits

    # Calculate loss (cross-entropy)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = reference_sequence[..., 1:].contiguous()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    total_loss += loss.item()

    # Limit the number of tokens generated to match reference size for BLEU
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

    bleu_score = sentence_bleu(reference, candidate)
    total_bleu_score += bleu_score

# Calculate the average BLEU score and the average loss
average_bleu_score = total_bleu_score / num_iterations
average_loss = total_loss / num_iterations

print(f"Average BLEU score over {num_iterations} samples: {average_bleu_score}")
print(f"Average loss over {num_iterations} samples: {average_loss}")

# GPT2 level is about 0.413