from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import torch

# Set device: use MPS (Apple Silicon) if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load raw text corpus from a .txt file and convert to HuggingFace Dataset
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [para.strip() for para in text.split("\n") if para.strip()]
    return Dataset.from_dict({"text": paragraphs})

# Load your law corpus
dataset = load_text_file("data.txt")

# Format each example (append newline to ensure better separation)
def format_text(example):
    return {"text": f"{example['text']}\n"}

dataset = dataset.map(format_text)

# Load pre-trained tokenizer and model (DistilGPT2)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 models do not have a pad token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id  # Important for padding
model.to(device)

# Tokenization function: tokenize and set labels for causal language modeling
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./law_GPT_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,              # Small batch size for MacBook memory
    gradient_accumulation_steps=8,              # Accumulate gradients to simulate larger batch
    num_train_epochs=5,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=False,                                 # MPS does not support FP16 yet
    bf16=False,
    torch_compile=False,                        # Can improve performance if True in some setups
    dataloader_pin_memory=False,
    report_to="none"                            # Disable WandB or other reporters
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save final model and tokenizer
trainer.save_model("./law_GPT_model")
tokenizer.save_pretrained("./law_GPT_model")

print("Training complete. Model saved.")
