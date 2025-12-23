"""
CLI entrypoint for training the surrogate ensemble from a YAML config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .surrogate import train_from_config


def main():
    parser = argparse.ArgumentParser(description="Train surrogate ensemble from config.")
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write metrics JSON (defaults next to model).",
    )
    args = parser.parse_args()

    metrics = train_from_config(args.config)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
