#!/usr/bin/env python3
"""Reject formal SuRe K2 training until its required GPU and data gates pass."""

from __future__ import annotations

import argparse

from opd.readiness import validate_sure_k2_readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    validate_sure_k2_readiness(arguments.config)
    print(
        "SuRe K2 readiness passed: data manifests, contamination audit, Base evaluations, "
        "target-GPU smoke, resumability pilot, pilot budget, and peak disk gate are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
