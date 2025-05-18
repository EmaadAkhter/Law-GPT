from datasets import load_dataset
import random

def main():
    print("Loading CodeSearchNet Python dataset...")
    
    # Load the Python subset of the CodeSearchNet dataset
    # Using the 'train' split only
    dataset = load_dataset("code_search_net", "python", split="train")

    print("🧹 Filtering for examples with both docstring and code...")

    # Keep examples that have both a docstring and function code
    def has_both_fields(example):
        return bool(example.get("func_documentation_string")) and bool(example.get("func_code_string"))

    dataset = dataset.filter(has_both_fields)

    print(f"Found {len(dataset)} valid examples.")

    # Shuffle the dataset and limit it to 10,000 examples for manageable size
    dataset = dataset.shuffle(seed=42).select(range(min(10000, len(dataset))))

    print("Formatting and saving to file...")

    # Format each example and write to a text file
    with open("code_search_net_corpus.txt", "w", encoding="utf-8") as f:
        for example in dataset:
            doc = example["func_documentation_string"].strip().replace("\n", " ")
            code = example["func_code_string"].strip()
            combined = f"<docstring> {doc} <code> {code}\n"
            f.write(combined)

    print("Saved corpus to 'code_search_net_corpus.txt'")

if __name__ == "__main__":
    main()
