#!/usr/bin/env python3
"""
Contract checks for run_quantization runner refactor.

These tests validate:
1) Matrix-driven command expansion is stable and complete.
2) Shell runner supports a cheap DRY_RUN smoke path for quick verification.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts.quantization_matrix import iter_jobs, load_matrix, validate_matrix


ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "configs" / "quantization_matrix.yaml"


def test_matrix_contract() -> None:
    matrix = load_matrix(MATRIX_PATH)
    validate_matrix(matrix)

    jobs = list(iter_jobs(matrix, model_ids=[], dataset_ids=[], num_samples=512))
    assert len(matrix["models"]) == 6
    assert len(matrix["datasets"]) == 7
    assert len(jobs) == 42

    first = jobs[0]
    assert first["target_override"] == "target.model=Qwen/Qwen3-8B-Base"
    assert first["task_override"] == "data.task=wikitext"
    assert first["samples_override"] == "data.num_samples=512"

    last = jobs[-1]
    assert last["target_override"] == "target.model=mistralai/Mistral-7B-Instruct-v0.3"
    assert last["task_override"] == "data.task=squad"


def test_run_quantization_dry_run_smoke() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MODELS": "qwen25_7b_instruct,llama31_8b_base",
            "DATASETS": "wikitext,gsm8k",
            "NUM_SAMPLES": "4",
        }
    )
    completed = subprocess.run(
        ["bash", "run_quantization.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "Resolved 2 model(s) × 2 dataset(s)" in completed.stdout
    assert "DRY_RUN=1 (commands are printed only)" in completed.stdout
    assert completed.stdout.count(">>> python run.py --config configs/quantization.yaml") == 4
    assert "FAILED:" not in completed.stdout
