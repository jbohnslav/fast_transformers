from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")
    
    sample = ds[0]
    print("Sample structure:")
    print(f"- images: {len(sample['images'])} items, first is {type(sample['images'][0])}")
    print(f"- texts: {len(sample['texts'])} items")
    
    print("\nTexts content:")
    for i, text in enumerate(sample['texts']):
        print(f"[{i}] {text[:200] if len(text) > 200 else text}")