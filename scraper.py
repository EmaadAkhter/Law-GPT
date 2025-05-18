from datasets import load_dataset
import random

def main():
    print("🔽 Loading CodeSearchNet Python dataset...")
    # Correct config-based loading
    dataset = load_dataset("code_search_net", "python", split="train")

    print("🧹 Filtering for examples with both docstring and code...")
    def has_both_fields(example):
        return bool(example.get("func_documentation_string")) and bool(example.get("func_code_string"))

    dataset = dataset.filter(has_both_fields)

    print(f"✅ Found {len(dataset)} valid examples.")

    # Shuffle and limit to 10,000 examples
    dataset = dataset.shuffle(seed=42).select(range(min(10000, len(dataset))))

    print("📝 Formatting and saving to file...")
    with open("code_search_net_corpus.txt", "w", encoding="utf-8") as f:
        for example in dataset:
            doc = example["func_documentation_string"].strip().replace("\n", " ")
            code = example["func_code_string"].strip()
            combined = f"<docstring> {doc} <code> {code}\n"
            f.write(combined)

    print("✅ Saved corpus to 'code_search_net_corpus.txt'")

if __name__ == "__main__":
    main()
