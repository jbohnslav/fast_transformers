# /// script
# dependencies = [
#   "torch",
#   "numpy",
# ]
# ///

import argparse
import json
import logging
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

# --- Configuration ---
CONFIG = {
    "versions": [
        {
            "name": "4.53.3",
            "source": "transformers==4.53.3",
        },
        {
            "name": "4.54.0",
            "source": "transformers==4.54.0",
        },
        {
            "name": "transformers-main",
            "source": "git+https://github.com/huggingface/transformers.git",
        },
        {
            "name": "transformers-fork",
            "source": "-e transformers-fork/",
        },
    ],
    "common_dependencies": [
        "torch>=2.7.1",
        "torchvision>=0.22.1",
        "datasets>=4.0.0",
        "pillow>=11.3.0",
        "qwen-vl-utils>=0.0.11",
        "accelerate",
    ],
    "results_dir": Path("results"),
    "venvs_dir": Path(".venvs"),
}


# --- Internal Functions ---
def _run_command(command: list[str], log_file: Path) -> None:
    logging.info(f"Running command: {' '.join(command)}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Running: {' '.join(command)} ---\n")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        )
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            f.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def _setup_logging(run_dir: Path | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if run_dir:
        handlers.append(logging.FileHandler(run_dir / "run.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=handlers)


# --- Core Logic ---
def setup_environments(log_file: Path):
    logging.info("Setting up virtual environments...")
    venvs_dir = CONFIG["venvs_dir"]
    venvs_dir.mkdir(exist_ok=True)
    for cfg in CONFIG["versions"]:
        venv_path = venvs_dir / f"venv-{cfg['name']}"
        if not venv_path.exists():
            _run_command(["uv", "venv", str(venv_path)], log_file)
        python_executable = venv_path / "bin" / "python"
        install_command = [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_executable),
            *shlex.split(cfg["source"]),
            *CONFIG["common_dependencies"],
        ]
        _run_command(install_command, log_file)
    logging.info("All environments are ready.")


def run_experiment(run_dir: Path, log_file: Path):
    logging.info(f"Starting experiment in: {run_dir}")
    for cfg in CONFIG["versions"]:
        name = cfg["name"]
        logging.info(f"--- Processing version: {name} ---")
        python_executable = CONFIG["venvs_dir"] / f"venv-{name}" / "bin" / "python"
        version_output_dir = run_dir / name
        version_output_dir.mkdir(exist_ok=True)
        _run_command(
            [str(python_executable), "benchmark.py", f"--version={name}", f"--output-dir={version_output_dir}"],
            log_file,
        )


def generate_summary_report(run_dir: Path):
    logging.info("Generating summary report...")
    all_variants = []
    for cfg in CONFIG["versions"]:
        version_dir = run_dir / cfg["name"]
        for timing_file in sorted(version_dir.glob("timing_*.json")):
            with open(timing_file) as f:
                data = json.load(f)
            stats = data.get("statistics", {})
            variant_name = f"{data['version']}_{data['model_note']}"
            all_variants.append(
                {
                    "name": variant_name,
                    "version": data["version"],
                    "note": data.get("model_note", "N/A"),
                    "logits_path": version_dir / f"logits_{data['model_note']}.pt",
                    **stats,
                }
            )

    # N×N Matrix Calculation
    matrix = {v["name"]: {} for v in all_variants}
    for v_a in all_variants:
        if not v_a["logits_path"].exists():
            continue
        logits_a = torch.load(v_a["logits_path"], map_location="cpu")
        for v_b in all_variants:
            if not v_b["logits_path"].exists():
                continue
            logits_b = torch.load(v_b["logits_path"], map_location="cpu")
            l2_diff = torch.norm(logits_a - logits_b).item()
            matrix[v_a["name"]][v_b["name"]] = f"{l2_diff:.6f}"

    # Write Report
    report_path = run_dir / "summary.md"
    with open(report_path, "w") as f:
        f.write(f"# Benchmark Summary\n\n- **Run directory**: `{run_dir.resolve()}`\n\n")
        f.write("## Performance Results\n\n")
        f.write("| Variant | Note | p50 (ms) | p90 (ms) | p99 (ms) | Average (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.writelines(
            f"| {item['name']} | {item['note']} | {item['p50_ms']:.2f} | {item['p90_ms']:.2f} | {item['p99_ms']:.2f} | {item['average_ms']:.2f} |\n"
            for item in sorted(all_variants, key=lambda x: x["name"])
        )

        f.write("\n## Logit L2 Differences (N×N)\n\n")
        header = [""] + [v["name"] for v in all_variants]
        f.write("|" + " | ".join(header) + "|\n")
        f.write("|" + "---|" * len(header) + "\n")
        for v_a in all_variants:
            row = [f"**{v_a['name']}**"] + [matrix[v_a["name"]].get(v_b["name"], "N/A") for v_b in all_variants]
            f.write("| " + " | ".join(row) + " |\n")

    logging.info(f"Summary report saved to: {report_path}")


def clean():
    logging.info("Cleaning up generated files...")
    for dir_path in [CONFIG["results_dir"], CONFIG["venvs_dir"]]:
        if dir_path.exists():
            shutil.rmtree(dir_path)


def main():
    parser = argparse.ArgumentParser(description="Unified Python script for transformer benchmarking.")
    parser.add_argument("--clean", action="store_true", help="Clean up all generated artifacts before running.")
    args = parser.parse_args()

    if args.clean:
        clean()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CONFIG["results_dir"] / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    log_file = run_dir / "run.log"

    try:
        setup_environments(log_file)
        run_experiment(run_dir, log_file)
        generate_summary_report(run_dir)
        logging.info(f"Benchmark run completed successfully. Results are in: {run_dir.resolve()}")
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logging.exception(f"An error occurred: {e}", exc_info=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
