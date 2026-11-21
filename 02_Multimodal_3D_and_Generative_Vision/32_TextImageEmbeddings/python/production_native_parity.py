"""End-to-end parity: exported ONNX embeddings via Python and C++17 runtime."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: production_native_parity.py <native-binary> <export-dir>")
    binary = str(Path(sys.argv[1]).resolve())
    directory = Path(sys.argv[2]).resolve()
    for kind, rows, width, dtype in [
        ("image", 1, 3 * 32 * 32, np.float32),
        ("text", 10, 32, np.int64),
    ]:
        model_path = directory / f"{kind}.onnx"
        input_path = directory / f"{kind}_input.txt"
        flat = np.loadtxt(input_path, dtype=dtype).reshape(rows, width)
        tensor = flat.reshape(rows, 3, 32, 32) if kind == "image" else flat
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        expected = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        result = subprocess.run(
            [binary, str(model_path), kind, str(input_path), str(rows)],
            check=True, capture_output=True, text=True,
        )
        actual = np.array(
            [[float(number) for number in line.split()] for line in result.stdout.splitlines()],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        print(f"Module 32 {kind}: native ONNX embeddings agree with Python")
    print("Module 32 deployment parity: PASS")


if __name__ == "__main__":
    main()
