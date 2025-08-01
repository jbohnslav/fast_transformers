# Transformers Benchmarking Framework

A Python-based framework for performance and numerical accuracy benchmarking of different `transformers` library versions. It uses `uv` for isolated environment management and compares both eager and `torch.compile()`d model performance.

## Quickstart

### 1. Configure

Edit the `CONFIG` dictionary in `main.py` to define the `transformers` versions to test.

```python
# main.py
CONFIG = {
    "versions": [
        {"name": "4.54.0", "source": "transformers==4.54.0"},
        {"name": "transformers-main", "source": "git+https://github.com/huggingface/transformers.git"},
        {"name": "transformers-fork", "source": "-e transformers-fork/"},
    ],
    # ...
}
```

### 2. Run

Execute the benchmark from your terminal. The `uv run` command ensures all dependencies, including `torch` for the final report generation, are available.

```bash
# Run the full benchmark workflow
uv run main.py

# To ensure a clean state, deleting all previous results and environments:
uv run main.py --clean
```

### 3. Analyze

Review the generated `summary.md` in the `results/run_<timestamp>/` directory.

---

## Example `summary.md` Output

Each run generates a self-contained summary report with performance metrics and a complete N×N matrix of logit differences between every variant.

````markdown
# Benchmark Summary

- **Run directory**: `/path/to/results/run_20250801_150000`

## Performance Results

| Variant | Note | p50 (ms) | p90 (ms) | p99 (ms) | Average (ms) |
|---|---|---|---|---|---|
| 4.54.0_compiled_sdpa | compiled_sdpa | 45.87 | 48.91 | 49.02 | 46.35 |
| 4.54.0_eager_sdpa | eager_sdpa | 68.11 | 72.50 | 72.89 | 68.55 |
| transformers-fork_compiled_sdpa | compiled_sdpa | 45.92 | 48.88 | 48.99 | 46.41 |
| transformers-fork_eager_sdpa | eager_sdpa | 69.50 | 70.11 | 70.34 | 68.99 |
| transformers-main_compiled_sdpa | compiled_sdpa | 58.03 | 61.21 | 61.55 | 57.49 |
| transformers-main_eager_sdpa | eager_sdpa | 89.11 | 94.23 | 94.50 | 89.72 |

## Logit L2 Differences (N×N)

| | 4.54.0_eager_sdpa | 4.54.0_compiled_sdpa | transformers-fork_eager_sdpa | ... |
|---|---|---|---|---|
| **4.54.0_eager_sdpa** | 0.000000 | 0.000101 | 14.937500 | ... |
| **4.54.0_compiled_sdpa** | 0.000101 | 0.000000 | 14.937598 | ... |
| **transformers-fork_eager_sdpa** | 14.937500 | 14.937598 | 0.000000 | ... |
| ... | ... | ... | ... | ... |
````

---

## Workflow Details

-   **`main.py`**: The orchestrator script. It reads the configuration, manages `uv` environments, and calls the benchmark worker for each version. After all runs are complete, it aggregates the results from all artifacts and generates the final `summary.md` report.

-   **`benchmark.py`**: The worker script. For a single `transformers` version, it performs the following:
    1.  Runs a forward pass and performance benchmark for the model in **eager mode**.
    2.  Saves the eager mode logits (`logits_eager_sdpa.pt`) and performance metrics.
    3.  Compiles the model using `torch.compile()`.
    4.  Runs a forward pass and performance benchmark for the model in **compiled mode**.
    5.  Saves the compiled mode logits (`logits_compiled_sdpa.pt`) and performance metrics.

## Output Artifacts

For each version tested, the framework generates a subdirectory containing:

-   `timing_..._eager_sdpa.json`: Performance data for the standard model.
-   `trace_..._eager_sdpa.json`: PyTorch profiler trace for the standard model.
-   `logits_eager_sdpa.pt`: Raw output logits from the standard model.
-   `timing_..._compiled_sdpa.json`: Performance data for the compiled model.
-   `trace_..._compiled_sdpa.json`: PyTorch profiler trace for the compiled model.
-   `logits_compiled_sdpa.pt`: Raw output logits from the compiled model.
