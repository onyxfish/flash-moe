#!/usr/bin/env python3
"""Export HuggingFace tokenizer.json vocab to vocab.bin for infer.m token decoding.

This is distinct from tokenizer.bin (export_tokenizer.py), which contains BPE
merge rules for encoding. vocab.bin is used only for decoding token IDs to strings.

Binary format (matches load_vocab() in infer.m):
  uint32  num_entries
  uint32  max_id
  For each token (in token_id order, 0..max_id inclusive):
    uint16  byte_len   (0 = missing/unknown token)
    char[byte_len]     UTF-8 bytes

Usage:
    python export_vocab.py [tokenizer.json] [output.bin]
"""

import json
import struct
import sys
import os


def _build_unicode_to_byte():
    """Reverse of GPT-2 bytes_to_unicode(): maps BPE char back to the byte it represents."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_UNICODE_TO_BYTE = _build_unicode_to_byte()


def decode_bpe_token(s):
    """Convert a BPE vocab string (with Ġ, Ċ, etc.) to real UTF-8 bytes."""
    raw = bytes(_UNICODE_TO_BYTE[c] for c in s if c in _UNICODE_TO_BYTE)
    return raw.decode("utf-8", errors="replace")


def main():
    hf_snapshot = (
        os.path.expanduser("~/.cache/huggingface/hub")
        + "/models--mlx-community--Qwen3.5-397B-A17B-4bit"
        "/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/tokenizer.json"
    )
    tok_path = sys.argv[1] if len(sys.argv) > 1 else hf_snapshot
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vocab.bin"

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    # Build id -> string map from vocab + added_tokens
    id_to_str = {}

    vocab = t["model"]["vocab"]  # str -> int
    for s, i in vocab.items():
        id_to_str[i] = decode_bpe_token(s)

    # Added tokens (e.g. <|im_end|>) are already plain text, not BPE-encoded
    for entry in t.get("added_tokens", []):
        id_to_str[entry["id"]] = entry["content"]

    max_id = max(id_to_str.keys())
    num_entries = max_id + 1

    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", num_entries, max_id))
        for i in range(num_entries):
            s = id_to_str.get(i, "")
            encoded = s.encode("utf-8")
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)

    print(f"[vocab] Wrote {num_entries} tokens (max_id={max_id}) to {out_path}")

if __name__ == "__main__":
    main()
