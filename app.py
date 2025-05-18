from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)

#Load the fine-tuned law GPT model
MODEL_PATH = "./law_GPT_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()  # Set model to inference mode

#Set required padding and EOS token for GPT-style models
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

#Route: Home page
@app.route("/")
def index():
    return render_template("index.html")  # Assumes you have a templates/index.html file

#Route: Handle user question POST requests
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please provide a valid question."})

    #Construct prompt from question
    prompt = f"You asked: \"{question}\"\n\n"

    #Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    #Generate response from model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,            # Limit the length of the response
            min_length=50,             # Force at least some elaboration
            do_sample=False,           # Deterministic generation
            top_k=1,                   # Only the most likely token
            top_p=1.0,                 # Disable nucleus sampling
            temperature=0.3,           # Make responses conservative/deterministic
            repetition_penalty=2.0,    # Discourage repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    #Decode output and post-process
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()             # Remove the prompt
    answer = answer.split("\n\n")[0].strip()                  # Clean up extra spacing

    return jsonify({"answer": answer})

#Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

