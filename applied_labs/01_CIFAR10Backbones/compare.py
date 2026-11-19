"""Compare Applied-Lab-01 training reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = [load(path) for path in args.metrics]
    rows.sort(key=lambda row: float(row["test_accuracy"]), reverse=True)

    print(
        f"{'model':<12} {'params':>12} "
        f"{'best val':>10} {'test acc':>10} {'device':>10}"
    )
    print("-" * 60)
    for row in rows:
        print(
            f"{str(row['model']):<12} "
            f"{int(row['parameters']):>12,d} "
            f"{float(row['best_val_accuracy']):>10.4f} "
            f"{float(row['test_accuracy']):>10.4f} "
            f"{str(row['device']):>10}"
        )


if __name__ == "__main__":
    main()
