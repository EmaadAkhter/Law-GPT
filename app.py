from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load the fine-tuned model
MODEL_PATH = "./law_GPT_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

# Set padding and eos token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please provide a valid question."})

    # Create a simple prompt with the user's question
    prompt = f"You asked: \"{question}\"\n\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,  # Limit the response length for concise answers
            min_length=50,   # Ensure a minimum length for meaningful responses
            do_sample=False, # Disable sampling for deterministic output
            top_p=1.0,       # Include all tokens (no nucleus sampling)
            top_k=1,         # Consider only the most probable token
            temperature=0.3, # Lower temperature for more deterministic behavior
            repetition_penalty=2.0,  # Penalize repetitive text
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Only return the part after the prompt
    answer = generated_text[len(prompt):].strip()

    # Optional: Strip out anything weird after extra EOS or line breaks
    answer = answer.split("\n\n")[0].strip()

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
