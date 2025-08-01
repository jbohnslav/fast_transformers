import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 1. Load processor and model
print("Loading model and processor...")
# Note: Using the 7B model for compatibility with the benchmark script's likely intent,
# even though it specifies the 2B. The debugging principles are the same.
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True).eval()

# 2. Prepare inputs from the_cauldron dataset
print("Preparing inputs from the_cauldron dataset...")
ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")
sample = ds[0]

image = sample["images"][0]
question = sample["texts"][0]["user"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    },
]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# The original benchmark uses a utility function `process_vision_info`,
# but for a single image, passing it directly to the processor is sufficient.
inputs = processor(text=[prompt], images=[image], return_tensors="pt")


# 3. Run forward pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(**inputs)
print("Script finished.")
