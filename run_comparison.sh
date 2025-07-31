#!/bin/bash
# Run baseline vs fork comparison

set -e

TIMESTAMP=$(date -u '+%Y%m%d_%H%M%S')

# Baseline
.venv-baseline/bin/python forward_qwen.py --dump baseline.pt

# verify forward pass equality
echo "Checking forward pass equality..."
.venv-fork/bin/python forward_qwen.py --ref baseline.pt
EQUALITY_CHECK=$?

if [ $EQUALITY_CHECK -eq 0 ]; then
    echo "✅ Forward passes match - proceeding with benchmarks"
    
    # benchmark
    .venv-baseline/bin/python benchmark_clean.py --version=baseline --timestamp=$TIMESTAMP
    .venv-fork/bin/python benchmark_clean.py --version=fork --timestamp=$TIMESTAMP
    
    echo "Results: timing_${TIMESTAMP}_*.json, trace_${TIMESTAMP}_*.json"
else
    echo "❌ Forward passes don't match - skipping benchmarks"
    exit 1
fi