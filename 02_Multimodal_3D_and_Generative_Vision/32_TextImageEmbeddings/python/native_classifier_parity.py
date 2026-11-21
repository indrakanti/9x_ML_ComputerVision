"""Prove complete C++ image→class native inference matches Python ORT.

Uses a real lossless PNG and exported models, not synthetic embedding text files.
Also verifies native code fails closed when a model is tampered with.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from infer_benchmark import infer, load_image


def check(binary: Path, directory: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="cv9x-module32-") as location:
        root = Path(location)
        picture = root / "sample.png"
        # Lossless 32×32 RGB makes PyTorch/PIL and OpenCV preprocessing directly comparable.
        rgb = np.random.default_rng(32).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(picture)
        args = [
            str(binary.resolve()),
            "--export-dir", str(directory.resolve()),
            "--image", str(picture),
            "--warmup", "1",
            "--repeats", "3",
        ]
        actual = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
        expected = infer(directory.resolve(), load_image(picture), warmup=1, repeats=3)
        assert actual["prediction"] == expected["class"]
        actual_labels = [item["label"] for item in actual["ranked"]]
        expected_labels = [item["label"] for item in expected["ranked"]]
        assert actual_labels == expected_labels
        np.testing.assert_allclose(
            [item["probability"] for item in actual["ranked"]],
            [item["probability"] for item in expected["ranked"]],
            rtol=1e-4,
            atol=1e-5,
        )
        assert actual["benchmark"]["latency_p50_ms"] >= 0.0
        assert actual["benchmark"]["latency_p95_ms"] >= 0.0

        tampered = root / "tampered"
        tampered.mkdir()
        for filename in ("image.onnx", "text.onnx", "manifest.json"):
            shutil.copy2(directory / filename, tampered / filename)
        with (tampered / "image.onnx").open("ab") as stream:
            stream.write(b"tampered")
        args[2] = str(tampered)
        denied = subprocess.run(args, capture_output=True, text=True, check=False)
        if denied.returncode == 0 or "SHA256 mismatch" not in denied.stderr:
            raise AssertionError("native classifier accepted a tampered ONNX model")
    print("Module 32: real PNG native classification, Python parity, tamper rejection PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("export_dir", type=Path)
    args = parser.parse_args()
    check(args.binary, args.export_dir)


if __name__ == "__main__":
    main()
