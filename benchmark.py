"""Benchmark script for Qwen2 VL model.

Best-practice steps implemented:
1. Load one representative sample (image + question) from AI2D dataset.
2. Prepare model inputs using the official processor helper.
3. Perform several warm-up iterations to remove first-time overheads (CUDA kernel load, graph construction, etc.).
4. Measure end-to-end latency over N timed iterations, synchronising the CUDA device before and after each forward pass to obtain accurate timings.
5. Report average, p50, p90 and p99 latencies.
6. Run a single forward pass under PyTorch Profiler and export an optional Chrome trace (trace.json) for further inspection.

NOTE: This script assumes a GPU is available and that `torch.cuda.is_available()` is True.
"""

import datetime
import gc
import sys
import time
from pathlib import Path
from statistics import mean

# has to be done before importing transformers or torch
sys.path.insert(0, "..")
sys.path.insert(0, "transformers-fork")

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile
from transformers_fast.models.qwen2_vl.modeling_qwen2_vl import (  # noqa: F401 # pyright: ignore[reportUnusedImport]
    Qwen2VisionTransformerPretrainedModel,  # pyright: ignore[reportUnusedImport]
    Qwen2VLAttention,  # pyright: ignore[reportUnusedImport]
    Qwen2VLDecoderLayer,  # pyright: ignore[reportUnusedImport]
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,  # pyright: ignore[reportUnusedImport]
    Qwen2VLRotaryEmbedding,  # pyright: ignore[reportUnusedImport]
    Qwen2VLTextModel,  # pyright: ignore[reportUnusedImport]
    Qwen2VLVisionBlock,  # pyright: ignore[reportUnusedImport]
    VisionAttention,  # pyright: ignore[reportUnusedImport]
    VisionRotaryEmbedding,  # pyright: ignore[reportUnusedImport]
)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = torch.device("cuda:0")
WARMUP_ITERS = 5  # not timed
MEASURE_ITERS = 10  # timed


def prepare_inputs(processor, sample):
    """Convert a TheCauldron sample into model-friendly tensors on DEVICE."""
    image = sample["images"][0]
    question = sample["texts"][0]["user"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(DEVICE)

def benchmark(model, inputs, note: str = ""):
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(**inputs)
    torch.cuda.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for _ in range(MEASURE_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(**inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

    latencies_ms.sort()
    avg_ms = mean(latencies_ms)
    p50_ms = latencies_ms[int(0.5 * len(latencies_ms))]
    p90_ms = latencies_ms[int(0.9 * len(latencies_ms))]
    p99_ms = latencies_ms[-1] if len(latencies_ms) > 0 else float("nan")

    print("\n=== Latency statistics ===")
    print(f"Average over {MEASURE_ITERS} runs : {avg_ms:.2f} ms")
    print(f"p50                                 : {p50_ms:.2f} ms")
    print(f"p90                                 : {p90_ms:.2f} ms")
    print(f"p99                                 : {p99_ms:.2f} ms")

    print("\nRunning PyTorch profiler for a single forward pass …")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof, torch.no_grad():
        _ = model(**inputs)
    trace_output = f"trace_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}_{note}.json"
    prof.export_chrome_trace(trace_output)

    print("Profiler trace saved to", Path(trace_output).resolve())
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

def main():
    # Use the slow processor for maximum compatibility
    processor = transformers_fast.AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    # Representative sample from dataset (first element from train split)
    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")
    sample = ds[0]
    inputs = prepare_inputs(processor, sample)

    # # Load model with flash attention 2
    # model = transformers_fast.Qwen2VLForConditionalGeneration.from_pretrained(
    #     MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    # ).to(DEVICE)
    # benchmark(model, inputs, "flash_attention_2")
    # del model
    # gc.collect()


    # Load model with sdpa
    model = transformers_fast.Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)

    benchmark(model, inputs, "sdpa")


    # compile the sdpa model
    model = torch.compile(model)
    benchmark(model, inputs, "sdpa_compiled")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
