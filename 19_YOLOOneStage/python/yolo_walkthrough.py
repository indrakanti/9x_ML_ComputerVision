"""Walkthrough for Module 19 YOLO-style decode."""

from __future__ import annotations

from pathlib import Path

from yolo_decode import decode, final_detections, load_artifact


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "yolo_head.txt")

    print("Decoded candidates:")
    for detection in decode(artifact):
        print(detection)

    print("\nAfter NMS:")
    for detection in final_detections(artifact):
        print(detection)


if __name__ == "__main__":
    main()
