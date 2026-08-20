#!/usr/bin/env python3
"""One-command entry point for the teammate Jackal test mode."""

from __future__ import annotations

import sys

from run_jackal_robohub import main


if __name__ == "__main__":
    forwarded_args = [
        arg
        for arg in sys.argv[1:]
        if arg not in ("--test-mode", "--no-test-mode")
    ]
    raise SystemExit(main(["--test-mode", *forwarded_args]))
