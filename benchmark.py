import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch
import transformers
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile
from transformers.utils.logging import disable_progress_bar

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda:0"
WARMUP_ITERS = 5
MEASURE_ITERS = 10


def prepare_inputs(processor, sample):
    """Convert a TheCauldron sample into model-friendly tensors on DEVICE."""
    image = sample["images"][0]
    question = sample["texts"][0]["user"]
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(
        DEVICE
    )


def perform_benchmark(version: str, output_dir: str):
    """Run benchmarks for both eager and compiled modes, saving all artifacts."""
    output_dir = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train[:1]")
    inputs = prepare_inputs(processor, ds[0])

    def _benchmark_internal(model, note) -> None:
        print(f"\n--- Benchmarking: {note} ---")
        model.eval()

        # Run forward pass once to get logits
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]

        # Save logits and choice probabilities
        torch.save(logits.cpu(), output_dir / f"logits_{note}.pt")
        probs = torch.softmax(logits, dim=-1)
        choices = ["A", "B", "C", "D", "E"]
        choice_probs = {c: probs[processor.tokenizer.convert_tokens_to_ids(c)].item() for c in choices}
        with open(output_dir / f"choices_{note}.json", "w") as f:
            json.dump(choice_probs, f, indent=2)

        # Performance and memory measurement
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                model(**inputs)
            torch.cuda.synchronize()
            
            # Reset memory stats after warmup
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
            
            latencies_ms = []
            for _ in range(MEASURE_ITERS):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(**inputs)
                end.record()
                torch.cuda.synchronize()
                latencies_ms.append(start.elapsed_time(end))
            
            # Get peak memory usage
            peak_memory = torch.cuda.max_memory_allocated()
            peak_memory_gb = (peak_memory - baseline_memory) / 1024**3

        latencies_ms.sort()
        stats = {
            "average_ms": mean(latencies_ms),
            "p50_ms": latencies_ms[int(0.5 * len(latencies_ms))],
            "p90_ms": latencies_ms[int(0.9 * len(latencies_ms))],
            "p99_ms": latencies_ms[-1],
            "peak_memory_gb": peak_memory_gb,
            "peak_memory_total_gb": peak_memory / 1024**3,
        }
        print("Latency stats:", stats)

        # Save timing data
        timing_data = {
            "timestamp": timestamp,
            "version": version,
            "model_note": note,
            "transformers_version": transformers.__version__,
            "statistics": stats,
        }
        with open(output_dir / f"timing_{timestamp}_{version}_{note}.json", "w") as f:
            json.dump(timing_data, f, indent=2)

        # Save profiler trace
        with (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            ) as prof,
            torch.no_grad(),
        ):
            model(**inputs)
        prof.export_chrome_trace(str(output_dir / f"trace_{timestamp}_{version}_{note}.json"))

    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)

    _benchmark_internal(model, "eager_sdpa")

    print("\nCompiling model with torch.compile()...")
    try:
        model_compiled = torch.compile(model)
        _benchmark_internal(model_compiled, "compiled_sdpa")
    except (RuntimeError, torch.jit.JITException) as e:
        print(f"Could not compile model, skipping compiled benchmark. Error: {e}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a transformers version.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    disable_progress_bar()
    perform_benchmark(args.version, args.output_dir)
