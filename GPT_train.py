from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments 
from datasets import load_dataset, Dataset
import torch
import os

# Set device for M2 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load raw text corpus
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [para.strip() for para in text.split("\n") if para.strip()]
    return Dataset.from_dict({"text": paragraphs})

dataset = load_text_file("law-of-crimes-sem--2-22-23.txt")

# Format text (simple wrap)
def format_text(example):
    return {"text": f"{example['text']}\n"}

dataset = dataset.map(format_text)

#Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id
model.to(device)

# Step 4: Tokenize dataset
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

#Training arguments
training_args = TrainingArguments(
    output_dir="./law_GPT_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,        
    gradient_accumulation_steps=8,       
    num_train_epochs=5,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    torch_compile=False,                   
    dataloader_pin_memory=False,
    report_to="none"
)

#Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

#Train the model
trainer.train()

#Save the final model
trainer.save_model("./law_GPT_model")
tokenizer.save_pretrained("./law_GPT_model")

print("Training complete. Model saved.")
