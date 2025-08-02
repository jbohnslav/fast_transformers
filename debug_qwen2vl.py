"""Minimal sanity-check script for Qwen2-VL.

Runs one sample through the *eager* model and an identical *torch.compile*
version, then prints the max / L2 differences of the logits so we can see
where numerical drift starts.

Example usage with a specific virtual environment:
    .venvs/venv-transformers-fork/bin/python debug_qwen2vl.py
"""

import random

import numpy as np
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Reproducibility -------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Uncomment to make eager / compiled use exactly the same SDPA backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


print("Loading processor and model …")
repo = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model_eager = (
    Qwen2VLForConditionalGeneration.from_pretrained(
        repo, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .eval()
    .cuda()
)
# model_comp = torch.compile(model_eager)
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
    model_comp = torch.compile(model_eager)

print("Preparing dataset sample …")
(ds,) = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")
image = ds["images"][0]
question = ds["texts"][0]["user"]

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = processor(text=[chat_prompt], images=[image], return_tensors="pt").to("cuda")
image_inputs, video_inputs = process_vision_info(
    [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
)

inputs = processor(
    text=[chat_prompt],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
).to("cuda")

print("Running eager / compiled forward pass …")
with torch.no_grad():
    # Casting the logits to float keeps the math the same for the forward pass, but
    # improves the quality of the distance calculation + the softmax.
    logits_eager = model_eager(**inputs).logits[0, -1].float()
    logits_comp = model_comp(**inputs).logits[0, -1].float()

diff = logits_eager - logits_comp
print("max abs diff:", diff.abs().max().item())
print("L2 norm diff:", torch.linalg.vector_norm(diff).item())

# Compare probabilities for choices A-E (same as benchmark.py)
choices = ["A", "B", "C", "D", "E"]
ids = [processor.tokenizer.convert_tokens_to_ids(c) for c in choices]
probs_eager = torch.softmax(logits_eager, dim=-1)
probs_comp = torch.softmax(logits_comp, dim=-1)
print("prob shift (A-E):")
for c, idx in zip(choices, ids, strict=False):
    print(f"  {c}: eager {probs_eager[idx]:.4f}  compiled {probs_comp[idx]:.4f}")
