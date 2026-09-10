'''
Loads rag-mini-wikipedia, prints real column names
 
Usage:
    python scripts/download_dataset.py
    
'''
from datasets import load_dataset

def main() -> None:
    print("Loading rajpurkar/squad...")
    squad = load_dataset("rajpurkar/squad")
 
    for split_name, split in squad.items():
        print(f"\n=== split: {split_name} ===")
        print(f"  num_rows: {split.num_rows}")
        print(f"  column_names: {split.column_names}")
        row = split[0]
        print(f"  first row:")
        for k, v in row.items():
            preview = str(v)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"    {k}: {preview}")
 
    # Quick sanity check relevant to corpus-building: how many unique
    # contexts (= candidate passages) exist in the train split, vs how
    # many questions share each one.
    train = squad["train"]
    unique_contexts = len(set(train["context"]))
    print(f"\nTrain split: {train.num_rows} questions over {unique_contexts} unique contexts")
    print(f"Average questions per context: {train.num_rows / unique_contexts:.2f}")

if __name__ == "__main__":
    main()