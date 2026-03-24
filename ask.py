#!/usr/bin/env python3
"""ask.py — plain text wrapper around the flash-moe inference engine.

Usage:
    python ask.py "Explain quantum computing"
    python ask.py --tokens 500 "Write a haiku about GPUs"
    echo "What is 2+2?" | python ask.py
"""

import subprocess
import sys
import re
import argparse
import os

INFER = os.path.join(os.path.dirname(__file__), "metal_infer", "infer")


def main():
    parser = argparse.ArgumentParser(description="Query the local 397B inference engine")
    parser.add_argument("prompt", nargs="?", help="Prompt text (or pipe via stdin)")
    parser.add_argument("--tokens", type=int, default=500, help="Max tokens to generate (default: 500)")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    result = subprocess.run(
        [INFER, "--prompt", prompt, "--tokens", str(args.tokens)],
        capture_output=True,
        text=True,
    )

    stdout = result.stdout

    # Extract text between "--- Output ---" and "--- Statistics ---"
    match = re.search(r"--- Output ---\n(.*?)\n\n--- Statistics ---", stdout, re.DOTALL)
    if not match:
        sys.stderr.write("error: could not find output section\n")
        sys.stderr.write(stdout)
        sys.exit(1)

    text = match.group(1)

    # Strip <think>...</think> blocks (thinking model reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    print(text.strip())


if __name__ == "__main__":
    main()
