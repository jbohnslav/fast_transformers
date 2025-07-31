import argparse
import sys

import torch
import transformers
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda:0"


def process_inputs(processor, sample):
    """Convert a dataset sample into model-friendly tensors."""
    image, question = sample["images"][0], sample["texts"][0]["user"]
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)


def run(out_path=None, ref_path=None, eps=1e-6):
    """Run forward pass with either baseline or fork transformers."""
    # Load dataset - using same sample as original scripts
    dataset = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")

    # Create processor and model using whichever transformers version is available
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = (
        transformers.Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    # Process inputs and run forward pass
    inputs = process_inputs(processor, dataset[0])
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]  # (vocab_size,) - last token logits

    # Show choice probabilities for both baseline and fork runs
    probs = torch.softmax(logits, dim=-1)
    choices = ["A", "B", "C", "D", "E"]
    choice_probs = []
    for choice in choices:
        token_id = processor.tokenizer.convert_tokens_to_ids(choice)
        if token_id is not None:
            prob = probs[token_id].item()
            choice_probs.append((choice, prob))
    print(f"Choice probabilities: {choice_probs}")

    # Save baseline results
    if out_path:
        torch.save(logits.cpu(), out_path)
        print(f"Dumped tensor → {out_path}")

    # Compare with reference (fork run)
    if ref_path:
        reference_logits = torch.load(ref_path, map_location="cpu")
        l2_diff = torch.norm(logits.cpu() - reference_logits).item()
        print(f"L2 diff: {l2_diff}")
        if l2_diff > eps:
            sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", help="Path to save baseline tensor")
    parser.add_argument("--ref", help="Path to reference tensor for comparison")
    args = parser.parse_args()

    run(out_path=args.dump, ref_path=args.ref)
