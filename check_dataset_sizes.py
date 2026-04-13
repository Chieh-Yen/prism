#!/usr/bin/env python3
"""
Check the actual sizes of the six datasets used in quantization experiments.
"""

from datasets import load_dataset

# The six datasets from TASK_REGISTRY
DATASETS = {
    "wikitext": {
        "hf_id": "Salesforce/wikitext",
        "hf_subset": "wikitext-2-raw-v1",
        "split": "test"
    },
    "c4": {
        "hf_id": "allenai/c4",
        "hf_subset": "en",
        "split": "validation",  # test split maps to validation
        "streaming": True
    },
    "lambada": {
        "hf_id": "EleutherAI/lambada_openai",
        "split": "test"
    },
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "hf_subset": "ARC-Challenge",
        "split": "test"
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "hf_subset": "main",
        "split": "test"
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "hf_subset": "all",
        "split": "test"
    }
}


def check_dataset_size(name, config):
    """Check the size of a dataset."""
    print(f"\n{name}:")
    print(f"  HF ID: {config['hf_id']}")
    if "hf_subset" in config:
        print(f"  Subset: {config['hf_subset']}")
    print(f"  Split: {config['split']}")

    try:
        load_kwargs = {"split": config["split"]}
        if "hf_subset" in config:
            load_kwargs["name"] = config["hf_subset"]

        if config.get("streaming"):
            print(f"  ⚠️  Streaming dataset (no fixed size)")
            print(f"  Note: Currently sampled at 128 or 256 samples")
        else:
            dataset = load_dataset(config["hf_id"], **load_kwargs)
            size = len(dataset)
            print(f"  ✅ Size: {size:,} samples")

            # Check if 128 samples is sufficient
            if size < 128:
                print(f"  ⚠️  WARNING: Dataset has fewer than 128 samples!")
            elif size >= 128:
                coverage = (128 / size) * 100
                print(f"  📊 Using 128 samples covers {coverage:.2f}% of the dataset")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    print("=" * 70)
    print("  Dataset Size Check")
    print("=" * 70)
    print(f"\n  Current configuration: num_samples = 128")

    for name in ["wikitext", "c4", "lambada", "arc", "gsm8k", "mmlu"]:
        if name in DATASETS:
            check_dataset_size(name, DATASETS[name])

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print("""
  The six datasets have varying sizes:
  - Some are small (hundreds to thousands of samples)
  - C4 is a streaming dataset with effectively unlimited samples
  - Using 128 samples provides a reasonable balance between:
    • Computation time
    • Memory usage
    • Statistical reliability
    """)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
