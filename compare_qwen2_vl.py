import json
import sys

sys.path.insert(0, "..")

from load_two_transformers import load_fork

transformers, transformers_fast = load_fork("transformers-fork")

import torch
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda:0"


class ImageJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles PIL Image objects."""

    def default(self, obj):
        if isinstance(obj, Image.Image):
            return f"<Image: mode={obj.mode}, size={obj.size[0]}x{obj.size[1]}>"
        return super().default(obj)


def forward_pass(model, inputs):
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)

    logits_BSV = outputs.logits  # (batch_size, seq_len, vocab_size)
    probs_BSV = torch.softmax(logits_BSV[0, -1], dim=-1)

    # Get probabilities for each choice (A, B, C, D, E)
    choices = ["A", "B", "C", "D", "E"]
    choice_probs = []

    for choice in choices:
        token_id = processor.tokenizer.convert_tokens_to_ids(choice)
        if token_id is not None:
            prob = probs_BSV[token_id].item()
            choice_probs.append((choice, prob))

    return logits_BSV, probs_BSV, choice_probs


def process_inputs(processor, sample):
    # Get image and text from the AI2 sample
    image = sample["images"][0]
    question = sample["texts"][0]["user"]

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs.to(device)


if __name__ == "__main__":
    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")

    sample = ds[0]
    print(json.dumps(sample, indent=2, cls=ImageJSONEncoder))
    # {
    #   "images": [
    #     "<Image: mode=RGB, size=299x227>"
    #   ],
    #   "texts": [
    #     {
    #       "user": "Question: What do respiration and combustion give out\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Heat\nAnswer with the letter.",
    #       "assistant": "Answer: B",
    #       "source": "AI2D"
    #     }
    #   ]
    # }

    # Use the slow processor to ensure compatibility
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    inputs = process_inputs(processor, sample)

    # FIRST MODEL: from transformers main branch!
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    logits_BSV, probs_BSV, choice_probs = forward_pass(model, inputs)
    print(choice_probs)

    # Sanity check: run-to-run consistency
    logits_list = []
    for _ in range(10):
        logits_BSV, probs_BSV, choice_probs = forward_pass(model, inputs)
        logits_list.append(logits_BSV)

    # run, batch, seq_len, vocab_size
    logits_stack_RBSV = torch.stack(logits_list)
    last_logit_RV = logits_stack_RBSV[:, 0, -1, :]
    # run x run L2 distance
    logits_dist_RR = torch.norm(last_logit_RV[:, None] - last_logit_RV, dim=2, p=2)
    print(f"L2 norm between logits of different runs: {logits_dist_RR}")

    # SECOND MODEL: from transformers fork
    model_fast = transformers_fast.Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model_fast.eval()

    logits_BSV_fast, probs_BSV_fast, choice_probs_fast = forward_pass(model_fast, inputs)
    print(choice_probs_fast)

    # compare the two models
    last_logit_V = logits_BSV[0, -1, :]
    last_logit_V_fast = logits_BSV_fast[0, -1, :]
    print(
        f"L2 norm between next-token logits of different models: {torch.norm(last_logit_V - last_logit_V_fast, dim=0, p=2)}"
    )
