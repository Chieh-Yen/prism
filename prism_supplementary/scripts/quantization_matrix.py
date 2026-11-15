#!/usr/bin/env python3
"""
Prints the model rows and datasets of configs/quantization_matrix.yaml for run_quantization.sh.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    values: list[str] = []
    for token in raw.replace(",", " ").split():
        token = token.strip()
        if token:
            values.append(token)
    return values


def load_matrix(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    if not isinstance(matrix, dict):
        raise ValueError("matrix root must be a mapping")
    matrix.setdefault("datasets", [])
    matrix.setdefault("models", [])
    return matrix


def validate_matrix(matrix: Dict) -> None:
    datasets = matrix.get("datasets", [])
    models = matrix.get("models", [])

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")

    unknown_types = [d for d in datasets if not isinstance(d, str) or not d.strip()]
    if unknown_types:
        raise ValueError("datasets contains invalid entries")

    seen_ids: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("each model entry must be a mapping")
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            raise ValueError("model.id is required")
        if model_id in seen_ids:
            raise ValueError(f"duplicate model.id: {model_id}")
        seen_ids.add(model_id)

        for key in ("target_model", "proxy_model"):
            value = str(model.get(key, "")).strip()
            if not value:
                raise ValueError(f"{model_id}: {key} is required")

        bits = model.get("quantization_bits", [])
        if not isinstance(bits, list) or not bits:
            raise ValueError(f"{model_id}: quantization_bits must be a non-empty list")
        if any((not isinstance(b, str) or not b.strip()) for b in bits):
            raise ValueError(f"{model_id}: quantization_bits contains invalid entries")


def _index_models(matrix: Dict) -> Dict[str, Dict]:
    return {m["id"]: m for m in matrix["models"]}


def filter_datasets(matrix: Dict, requested: Sequence[str]) -> List[str]:
    datasets = list(matrix["datasets"])
    if not requested:
        return datasets
    known = set(datasets)
    unknown = [d for d in requested if d not in known]
    if unknown:
        raise ValueError(f"unknown dataset id(s): {', '.join(unknown)}")
    return [d for d in datasets if d in set(requested)]


def filter_models(matrix: Dict, requested: Sequence[str]) -> List[Dict]:
    models = list(matrix["models"])
    if not requested:
        return models
    by_id = _index_models(matrix)
    unknown = [m for m in requested if m not in by_id]
    if unknown:
        raise ValueError(f"unknown model id(s): {', '.join(unknown)}")
    order = {model_id: i for i, model_id in enumerate(requested)}
    return sorted((by_id[m] for m in requested), key=lambda m: order[m["id"]])


def to_bits_override(bits: Sequence[str]) -> str:
    return f"proxy.quantization_bits=[{','.join(bits)}]"


def command_models(args: argparse.Namespace) -> int:
    matrix = load_matrix(Path(args.matrix))
    validate_matrix(matrix)
    models = filter_models(matrix, _parse_csv_list(args.models))
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    for model in models:
        writer.writerow([
            model["id"],
            f"target.model={model['target_model']}",
            f"proxy.model={model['proxy_model']}",
            f"proxy.gguf_template={model.get('gguf_template', '').strip()}",
            to_bits_override(model["quantization_bits"]),
        ])
    return 0


def command_datasets(args: argparse.Namespace) -> int:
    matrix = load_matrix(Path(args.matrix))
    validate_matrix(matrix)
    datasets = filter_datasets(matrix, _parse_csv_list(args.datasets))
    for dataset in datasets:
        print(dataset)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantization matrix utility")
    parser.add_argument(
        "--matrix",
        default="configs/quantization_matrix.yaml",
        help="Path to quantization matrix YAML",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="Emit model rows as TSV")
    p_models.add_argument("--models", default="", help="Comma or space separated model ids")
    p_models.set_defaults(func=command_models)

    p_datasets = sub.add_parser("datasets", help="Emit datasets one per line")
    p_datasets.add_argument("--datasets", default="", help="Comma or space separated dataset ids")
    p_datasets.set_defaults(func=command_datasets)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - script-level guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
