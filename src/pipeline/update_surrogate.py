"""
Retrain surrogate from a YAML config and emit metrics JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..models.surrogate import train_from_config


def main():
    parser = argparse.ArgumentParser(description="Retrain surrogate ensemble from config.")
    parser.add_argument("config", type=Path, help="Path to surrogate config YAML.")
    parser.add_argument("--out", type=Path, default=None, help="Optional metrics output JSON.")
    args = parser.parse_args()

    metrics = train_from_config(args.config)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
