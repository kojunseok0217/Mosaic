#!/usr/bin/env python3
"""Gemma ESR with the same A/B/C reference and D-after protocol as Qwen3."""
import sys

if __name__ == "__main__":
    if sys.argv[1:] == ["--check_environment"]:
        from esr_backends import validate_environment
        validate_environment("gemma")
    else:
        from evaluation_esr import main
        main(backend="gemma")
