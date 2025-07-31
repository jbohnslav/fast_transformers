# Clean UV-based Transformers Benchmarking

This project uses **UV environments** to cleanly separate baseline and fork versions of transformers, eliminating all the complex `sys.path` manipulation and import gymnastics.

## 🧘‍♂️ The Zen Approach

- **One environment = One transformers version**
- **No import tricks, no sys.path hacks**  
- **Clean, repeatable, maintainable**

## Quick Start

### 1. Setup (run once)

```bash
# Create and install both environments
make setup

# Or manually:
make setup-baseline  # GitHub main transformers
make setup-fork      # Local transformers-fork/
```

### 2. Run Comparison

```bash
# Easy way - run both phases:
make compare

# Or step by step:
make baseline  # Save baseline tensor
make fork      # Compare fork vs baseline

# Or use the script:
./run_comparison.sh
```

### 3. Benchmark Performance

```bash
# Benchmark baseline (GitHub main)
make benchmark-baseline

# Benchmark fork
make benchmark-fork
```

## File Structure

### New Clean Files
- `forward_qwen.py` - Main comparison script (replaces `compare_qwen2_vl.py`)
- `benchmark_clean.py` - Clean benchmark script (replaces `benchmark.py`)
- `Makefile` - All commands in one place
- `run_comparison.sh` - Simple bash script for comparisons

### Environments
- `.venv-baseline/` - GitHub transformers main
- `.venv-fork/` - Your local `transformers-fork/`

### Archived
- `archive/` - Old scripts with complex import tricks

## How It Works

### Phase 1: Baseline
```bash
.venv-baseline/bin/python forward_qwen.py --dump baseline.pt
```
Runs GitHub main transformers, saves the last-token logits to `baseline.pt`.

### Phase 2: Fork Comparison  
```bash
.venv-fork/bin/python forward_qwen.py --ref baseline.pt
```
Runs your fork, loads the baseline tensor, computes L2 distance.

## Key Benefits

✅ **No more wrestling with two libraries in one process**  
✅ **Each environment has exactly one transformers version**  
✅ **No sys.path manipulation or import tricks**  
✅ **Easily repeatable - just rerun make compare**  
✅ **Clean separation enables proper benchmarking**  
✅ **UV handles all the environment complexity**

## Dependencies

All dependencies are managed by UV in separate environments according to your `pyproject.toml`:

- `torch>=2.7.1`
- `datasets>=4.0.0` 
- `pillow>=11.3.0`
- `qwen-vl-utils>=0.0.11`
- Transformers: GitHub main vs your fork

## UV Commands Reference

```bash
# Setup environments
uv venv .venv-baseline
uv pip install --python .venv-baseline/bin/python [packages...]

# Run in specific environment (direct python to avoid UV project mode)
.venv-baseline/bin/python script.py

# Check what's installed
uv pip list --python .venv-baseline/bin/python
```

**Note:** We use direct python execution instead of `uv run --python` because UV's project mode (triggered by `pyproject.toml`) automatically creates/uses `.venv` even when specifying a different environment.

---

**No more import gymnastics! 🎉**
