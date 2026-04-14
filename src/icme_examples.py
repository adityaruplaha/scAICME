"""CLI helper for running project examples."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def _examples_root() -> Path:
    return Path(__file__).resolve().parents[1] / "examples"


def _available_examples(examples_dir: Path) -> list[str]:
    if not examples_dir.exists():
        return []

    examples = []
    for entry in sorted(examples_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "run.py").exists():
            examples.append(entry.name)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an scAICME example by name.")
    parser.add_argument("example", nargs="?", help="Example name (e.g., pbmc3k)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples and exit.",
    )
    args = parser.parse_args()

    examples_dir = _examples_root()
    examples = _available_examples(examples_dir)

    if args.list:
        for name in examples:
            print(name)
        return

    if not args.example:
        parser.error("example name is required (use --list to see options)")

    if args.example not in examples:
        available = ", ".join(examples) if examples else "<none>"
        parser.error(f"unknown example '{args.example}'. Available: {available}")

    runpy.run_path(str(examples_dir / args.example / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
