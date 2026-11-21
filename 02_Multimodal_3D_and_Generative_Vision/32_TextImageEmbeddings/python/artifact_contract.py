"""Versioned, strict Module 32 model-bundle contract for reusable inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from production import CLASSES, EMBED_DIM, MAX_TOKENS, PROMPTS

FILES = ("image.onnx", "text.onnx")
SCHEMA_VERSION = 1
PREPROCESSING = "RGB; resize to 32x32; float32 CHW in [0,1]"
TOKENIZER = "printable ASCII codepoint minus 31; pad 0 to 32"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(directory: Path) -> dict[str, Any]:
    """Fail closed on incompatible metadata, changed files, or missing artifacts."""
    manifest_path = directory / "manifest.json"
    data: dict[str, Any] = json.loads(manifest_path.read_text())
    expected: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "format": "onnx",
        "opset": 17,
        "image_shape": [1, 3, 32, 32],
        "text_shape": [len(CLASSES), MAX_TOKENS],
        "embedding_width": EMBED_DIM,
        "logit_scale": 10.0,
        "labels": list(CLASSES),
        "prompts": list(PROMPTS),
        "image_preprocessing": PREPROCESSING,
        "tokenizer": TOKENIZER,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"artifact contract mismatch: {key}")
    hashes = data.get("files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(FILES):
        raise ValueError("artifact model-file set mismatch")
    for filename in FILES:
        recorded = hashes[filename]
        if not isinstance(recorded, str) or len(recorded) != 64:
            raise ValueError(f"invalid checksum: {filename}")
        if sha256(directory / filename) != recorded:
            raise ValueError(f"model checksum mismatch: {filename}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_bundle(args.export_dir)
    print("Module 32 artifact contract: PASS")


if __name__ == "__main__":
    main()
