"""Versioned artifact checks, including rejection of tampered model bytes."""
import json
from pathlib import Path

import pytest
from artifact_contract import (
    FILES,
    PREPROCESSING,
    SCHEMA_VERSION,
    TOKENIZER,
    sha256,
    validate_bundle,
)
from production import CLASSES, EMBED_DIM, MAX_TOKENS, PROMPTS


def make_bundle(directory: Path) -> None:
    for filename in FILES:
        (directory / filename).write_bytes(b"test-only " + filename.encode())
    data = {
        "schema_version": SCHEMA_VERSION,
        "format": "onnx", "opset": 17,
        "image_shape": [1, 3, 32, 32],
        "text_shape": [len(CLASSES), MAX_TOKENS],
        "embedding_width": EMBED_DIM,
        "logit_scale": 10.0,
        "labels": list(CLASSES),
        "prompts": list(PROMPTS),
        "image_preprocessing": PREPROCESSING,
        "tokenizer": TOKENIZER,
        "files_sha256": {f: sha256(directory / f) for f in FILES},
    }
    (directory / "manifest.json").write_text(json.dumps(data))


def test_valid_bundle(tmp_path: Path) -> None:
    make_bundle(tmp_path)
    assert validate_bundle(tmp_path)["schema_version"] == 1


def test_tampered_model_rejected(tmp_path: Path) -> None:
    make_bundle(tmp_path)
    (tmp_path / "image.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_bundle(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2),
    ("image_shape", [1, 3, 224, 224]),
    ("tokenizer", "unknown"),
    ("logit_scale", 1.0),
    ("labels", ["changed"]),
])
def test_bad_metadata_rejected(tmp_path: Path, field: str, value: object) -> None:
    make_bundle(tmp_path)
    path = tmp_path / "manifest.json"
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_bundle(tmp_path)
