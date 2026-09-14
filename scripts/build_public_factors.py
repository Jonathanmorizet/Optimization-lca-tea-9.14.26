"""Create a reviewable public factor table from a private internal export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRIVATE_COLUMNS = {
    "Provider", "Provider UUID", "Dataset UUID", "Activity UUID",
    "Exchange UUID", "License Notes",
}
REQUIRED = {"Material", "Unit", "Stage", "Basis"}


def build(source: Path, destination: Path, confirm_license: bool = False) -> Path:
    if not confirm_license:
        raise RuntimeError(
            "Refusing to publish factors without --confirm-license. "
            "Obtain permission for the intended public use first."
        )
    frame = pd.read_csv(source)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    public = frame.drop(columns=[c for c in PRIVATE_COLUMNS if c in frame.columns])
    destination.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/factors_internal.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/factors_public.csv"))
    parser.add_argument("--confirm-license", action="store_true")
    args = parser.parse_args()
    build(args.input, args.output, args.confirm_license)
    print(args.output)


if __name__ == "__main__":
    main()
