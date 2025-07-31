# Clean UV-based benchmarking workflow
# No more import gymnastics!

.PHONY: setup-baseline setup-fork baseline fork compare benchmark-baseline benchmark-fork clean

# Setup environments (run once)
setup-baseline:
	uv venv .venv-baseline --clear
	uv pip install --python .venv-baseline/bin/python \
		"git+https://github.com/huggingface/transformers.git" \
		"torch>=2.7.1" "torchvision>=0.22.1" "datasets>=4.0.0" \
		"pillow>=11.3.0" "qwen-vl-utils>=0.0.11"

setup-fork:
	uv venv .venv-fork --clear
	uv pip install --python .venv-fork/bin/python \
		-e transformers-fork/ \
		"torch>=2.7.1" "torchvision>=0.22.1" "datasets>=4.0.0" \
		"pillow>=11.3.0" "qwen-vl-utils>=0.0.11"

setup: setup-baseline setup-fork

# Run baseline and save tensor
baseline:
	.venv-baseline/bin/python forward_qwen.py --dump baseline.pt

# Run fork and compare with baseline
fork:
	.venv-fork/bin/python forward_qwen.py --ref baseline.pt

# Complete comparison workflow
compare: baseline fork

# Benchmark baseline environment
benchmark-baseline:
	.venv-baseline/bin/python benchmark_clean.py

# Benchmark fork environment  
benchmark-fork:
	.venv-fork/bin/python benchmark_clean.py

# Clean up generated files
clean:
	rm -f baseline.pt *.json

# Clean everything including environments
clean-all: clean
	rm -rf .venv .venv-baseline .venv-fork