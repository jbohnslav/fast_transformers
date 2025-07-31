"""Clean benchmark script for Qwen2 VL model using UV environments.

This replaces the old benchmark.py with its complex sys.path manipulation.
Now uses clean UV environment separation instead.

Usage:
    # Benchmark baseline (GitHub main):
    uv run --python .venv-baseline/bin/python benchmark_clean.py --version=baseline

    # Benchmark fork:
    uv run --python .venv-fork/bin/python benchmark_clean.py --version=fork

    # Use shared timestamp for matching outputs:
    uv run --python .venv-baseline/bin/python benchmark_clean.py --version=baseline --timestamp=20250131_120000
    uv run --python .venv-fork/bin/python benchmark_clean.py --version=fork --timestamp=20250131_120000
"""

import argparse
import datetime
import gc
import json
import time
from pathlib import Path
from statistics import mean

import torch
import transformers
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile

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


def benchmark(model, inputs, note: str = "", version: str = "unknown", timestamp: str = None):
    """Benchmark model performance with detailed timing."""
    model.eval()

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # Measure
    latencies_ms = []
    with torch.no_grad():
        for _ in range(MEASURE_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(**inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

    # Statistics
    latencies_ms.sort()
    avg_ms = mean(latencies_ms)
    p50_ms = latencies_ms[int(0.5 * len(latencies_ms))]
    p90_ms = latencies_ms[int(0.9 * len(latencies_ms))]
    p99_ms = latencies_ms[-1] if len(latencies_ms) > 0 else float("nan")

    print(f"\n=== Latency statistics ({note or 'default'}) ===")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Transformers path: {transformers.__file__}")
    print(f"Average over {MEASURE_ITERS} runs : {avg_ms:.2f} ms")
    print(f"p50                                 : {p50_ms:.2f} ms")
    print(f"p90                                 : {p90_ms:.2f} ms")
    print(f"p99                                 : {p99_ms:.2f} ms")

    # Save timing information as JSON
    timing_data = {
        "timestamp": timestamp,
        "version": version,
        "model_note": note or "default",
        "transformers_version": transformers.__version__,
        "transformers_path": str(transformers.__file__),
        "warmup_iterations": WARMUP_ITERS,
        "measure_iterations": MEASURE_ITERS,
        "latencies_ms": latencies_ms,
        "statistics": {"average_ms": avg_ms, "p50_ms": p50_ms, "p90_ms": p90_ms, "p99_ms": p99_ms},
    }

    timing_output = f"timing_{timestamp}_{version}_{note or 'default'}.json"
    with open(timing_output, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"Timing data saved to: {Path(timing_output).resolve()}")

    # Profiler trace
    print("\nRunning PyTorch profiler for a single forward pass …")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof, torch.no_grad():
        _ = model(**inputs)

    trace_output = f"trace_{timestamp}_{version}_{note or 'default'}.json"
    prof.export_chrome_trace(trace_output)

    print("Profiler trace saved to", Path(trace_output).resolve())
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen2-VL model with timing and trace outputs")
    parser.add_argument("--version", default="unknown", help="Version identifier (e.g., 'baseline', 'fork')")
    parser.add_argument("--timestamp", help="Shared timestamp for matching outputs (format: YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    print(f"Using transformers from: {transformers.__file__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Version identifier: {args.version}")

    # Use provided timestamp or generate new one
    if args.timestamp:
        timestamp = args.timestamp
        print(f"Using provided timestamp: {timestamp}")
    else:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        print(f"Generated timestamp: {timestamp}")

    # Use the slow processor for maximum compatibility
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    # Representative sample from dataset (first element from train split)
    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")
    sample = ds[0]
    inputs = prepare_inputs(processor, sample)

    # Load model with sdpa (most reliable)
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)

    benchmark(model, inputs, "sdpa", version=args.version, timestamp=timestamp)

    # Benchmark compiled version
    print("\n" + "=" * 50)
    print("Now benchmarking with torch.compile...")
    model_compiled = torch.compile(model)
    benchmark(model_compiled, inputs, "sdpa_compiled", version=args.version, timestamp=timestamp)

    del model, model_compiled
    gc.collect()


if __name__ == "__main__":
    main()
