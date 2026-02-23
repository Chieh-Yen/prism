#!/usr/bin/env python3
"""
PRISM — Unified experiment runner.

Usage:
    python run.py --config configs/quantization.yaml
    python run.py --config configs/ood.yaml data.id_task=cifar100
    python run.py --config configs/merging.yaml merging.n_alphas=50 device=cpu

Positional arguments after --config are treated as dotted-path config
overrides  (key=value).  Nested keys use dots, e.g. ``target.model=X``.
Lists can be specified as ``proxy.quantization_bits='[Q8_0,Q4_K_M]'``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from prism.experiments import EXPERIMENT_REGISTRY


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` from ``'a.b.c'``."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    """Best-effort cast: int > float > bool > list/dict > str."""
    for caster in (int, float):
        try:
            return caster(raw)
        except (ValueError, TypeError):
            pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
    return raw


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_yaml(args.config)
    for override in args.overrides:
        if "=" not in override:
            print(f"Warning: ignoring malformed override '{override}' (expected key=value)")
            continue
        key, raw_value = override.split("=", 1)
        _set_nested(config, key, _parse_value(raw_value))
    return config


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PRISM — Proxy Risk Estimation via Isomorphic Spectral Matching",
        usage="python run.py --config CONFIG [key=value ...]",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/quantization.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted-path overrides, e.g. target.model=X data.num_samples=512",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = build_config(args)

    experiment_name = config.get("experiment")
    if experiment_name not in EXPERIMENT_REGISTRY:
        print(
            f"Error: unknown experiment '{experiment_name}'. "
            f"Available: {sorted(EXPERIMENT_REGISTRY.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    ExperimentClass = EXPERIMENT_REGISTRY[experiment_name]
    experiment = ExperimentClass(config)
    experiment.run()


if __name__ == "__main__":
    main()
